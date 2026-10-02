#!/usr/bin/env python3
"""QAT (Quantization-Aware Training) per Gemma-2-9B usando LoRA e torchao."""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from torchao.quantization import Int4WeightOnlyConfig, quantize_

MODEL_IDS = {"gemma": "google/gemma-2-9b"}

# Manteniamo la logica di Copilot per simulare i 4-bit durante il forward
# ma la applicheremo solo per rendere il modello "consapevole" della quantizzazione
class FakeQuantLinear(torch.nn.Linear):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scale = self.weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 7.0
        fake_weight = (self.weight / scale).round().clamp(-8, 7) * scale
        return torch.nn.functional.linear(inputs, fake_weight.to(inputs.dtype), self.bias)

def apply_fake_quant_to_base_layers(module: torch.nn.Module) -> None:
    for name, child in tuple(module.named_children()):
        # Quantizziamo solo i layer base, non gli adattatori LoRA
        if isinstance(child, torch.nn.Linear) and "lora" not in name.lower():
            replacement = FakeQuantLinear(
                child.in_features, child.out_features, child.bias is not None,
                device=child.weight.device, dtype=child.weight.dtype
            )
            replacement.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                replacement.bias.data.copy_(child.bias.data)
            setattr(module, name, replacement)
        else:
            apply_fake_quant_to_base_layers(child)

def restore_linear(module: torch.nn.Module) -> None:
    for name, child in tuple(module.named_children()):
        if isinstance(child, FakeQuantLinear):
            replacement = torch.nn.Linear(
                child.in_features, child.out_features, child.bias is not None,
                device=child.weight.device, dtype=child.weight.dtype
            )
            replacement.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                replacement.bias.data.copy_(child.bias.data)
            setattr(module, name, replacement)
        else:
            restore_linear(child)

class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, prompts: list[str], tokenizer, max_length: int):
        self.items = []
        for prompt in prompts:
            item = tokenizer(prompt, truncation=True, max_length=max_length)
            item["labels"] = item["input_ids"].copy()
            self.items.append(item)
    def __len__(self): return len(self.items)
    def __getitem__(self, index): return self.items[index]

def collate(tokenizer, batch):
    # "labels" ha lunghezza non paddata: va tolta prima di tokenizer.pad() e ricreata dopo
    batch = [{key: value for key, value in item.items() if key != "labels"} for item in batch]
    padded = tokenizer.pad(batch, padding=True, return_tensors="pt")
    padded["labels"] = padded["input_ids"].clone()
    padded["labels"][padded["attention_mask"] == 0] = -100
    return padded

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QAT (LoRA + Fake Quant) per Gemma-2")
    parser.add_argument("--model_path", type=str, default=MODEL_IDS["gemma"])
    parser.add_argument("--output_dir", type=Path, required=True)
    # Alpaca è nettamente superiore a Wikitext per il training instruct/QAT
    parser.add_argument("--dataset", default="yahma/alpaca-cleaned")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=200) # Limitiamo per la tesi
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    
    if not torch.cuda.is_available():
        raise RuntimeError("QAT richiede una GPU CUDA")
        
    print("Caricamento Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Caricamento Modello Base (BF16)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16,
        local_files_only=True, 
        device_map={"": 0}
    )
    model.config.use_cache = False
    
    # 1. Applichiamo LoRA per evitare OOM su A100
    print("Configurazione LoRA (Low-Rank Adaptation)...")
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "v_proj"], 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    
    # 2. Applichiamo la Fake Quantization ai layer base per renderli QAT-aware
    print("Applicazione Fake Quantization (4-bit simmetrica) per il training...")
    apply_fake_quant_to_base_layers(model)
    
    print("Preparazione Dataset Alpaca...")
    raw_dataset = load_dataset(args.dataset, split="train", download_mode="reuse_cache_if_exists")
    # Prendiamo le istruzioni da Alpaca
    prompts = [text for text in raw_dataset["instruction"] if text.strip()]
    dataset = PromptDataset(prompts, tokenizer, args.max_length)
    
    print("Inizio Addestramento QAT...")
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs, 
        max_steps=args.max_steps, 
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4, # Ottimizza la memoria
        learning_rate=args.learning_rate, 
        bf16=True, 
        gradient_checkpointing=True,
        save_strategy="no", 
        logging_steps=10, 
        report_to=[]
    )
    
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=dataset,
        data_collator=lambda batch: collate(tokenizer, batch)
    )
    trainer.train()
    
    # 3. Post-Training: Ricomponiamo e quantizziamo REALE a 4-bit!
    print("Training completato. Pulizia dei layer Fake Quant...")
    restore_linear(model)
    
    print("Fusione dei pesi LoRA nel modello base...")
    model = model.merge_and_unload()
    model.config.use_cache = True
    
    print("Applicazione vera quantizzazione torchao 4-bit per l'esportazione...")
    # Stesso packing format usato da evaluate.py per il caricamento (calibration != "gptq")
    quantize_(model, Int4WeightOnlyConfig(group_size=128, int4_packing_format="tile_packed_to_4d"))
    
    print("Salvataggio del modello QAT quantizzato...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Salviamo esattamente come in quantize_ptq.py per compatibilità con evaluate.py
    torch.save(model.state_dict(), args.output_dir / "quantized_model.pt")
    tokenizer.save_pretrained(args.output_dir)
    
    config_data = {
        "format": "torchao_int4",
        "base_model": MODEL_IDS["gemma"],
        "model_key": "gemma",
        "calibration": "qat",
        "dataset": args.dataset,
        "max_steps": args.max_steps
    }
    (args.output_dir / "ptq_config.json").write_text(json.dumps(config_data, indent=2), encoding="utf-8")
    
    print("Operazione conclusa con successo!")
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()