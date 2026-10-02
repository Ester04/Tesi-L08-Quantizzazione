#!/usr/bin/env python3
"""QAT LoRA riproducibile e allineata alla valutazione C4 per Gemma-2-9B."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from torchao.quantization import Int4WeightOnlyConfig, quantize_
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


MODEL_ID = "google/gemma-2-9b"
GROUP_SIZE = 128
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GroupwiseFakeQuantLinear(torch.nn.Linear):
    """Simula int4 weight-only per gruppi sull'asse degli input."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original_shape = self.weight.shape
        if original_shape[1] % GROUP_SIZE != 0:
            raise ValueError(
                f"in_features={original_shape[1]} non divisibile per group_size={GROUP_SIZE}"
            )

        grouped_weight = self.weight.reshape(-1, GROUP_SIZE)
        scale = grouped_weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 7.0
        quantized_weight = (grouped_weight / scale).round().clamp(-8, 7) * scale
        # Straight-through estimator: il forward vede int4, LoRA riceve gradienti utili.
        fake_weight = self.weight + (quantized_weight.reshape(original_shape) - self.weight).detach()
        return torch.nn.functional.linear(inputs, fake_weight.to(inputs.dtype), self.bias)


def apply_groupwise_fake_quant(module: torch.nn.Module) -> None:
    for name, child in tuple(module.named_children()):
        if "lora" in name.lower() or name == "lm_head":
            continue
        if isinstance(child, torch.nn.Linear):
            replacement = GroupwiseFakeQuantLinear(
                child.in_features,
                child.out_features,
                child.bias is not None,
                device=child.weight.device,
                dtype=child.weight.dtype,
            )
            replacement.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                replacement.bias.data.copy_(child.bias.data)
            replacement.weight.requires_grad_(child.weight.requires_grad)
            setattr(module, name, replacement)
        else:
            apply_groupwise_fake_quant(child)


def restore_linear_layers(module: torch.nn.Module) -> None:
    for name, child in tuple(module.named_children()):
        if isinstance(child, GroupwiseFakeQuantLinear):
            replacement = torch.nn.Linear(
                child.in_features,
                child.out_features,
                child.bias is not None,
                device=child.weight.device,
                dtype=child.weight.dtype,
            )
            replacement.weight.data.copy_(child.weight.data)
            replacement.weight.requires_grad_(child.weight.requires_grad)
            if child.bias is not None:
                replacement.bias.data.copy_(child.bias.data)
            setattr(module, name, replacement)
        else:
            restore_linear_layers(child)


class CausalLMDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int):
        self.items = [
            tokenizer(text, truncation=True, max_length=max_length)
            for text in texts
            if text.strip()
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def collate(tokenizer, batch: list[dict]) -> dict[str, torch.Tensor]:
    padded = tokenizer.pad(batch, padding=True, return_tensors="pt")
    labels = padded["input_ids"].clone()
    labels[padded["attention_mask"] == 0] = -100
    padded["labels"] = labels
    return padded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QAT LoRA int4 group-wise allineata a C4")
    parser.add_argument("--model_path", default=MODEL_ID)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_train_samples", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_c4_train() -> list[str]:
    local_dataset_path = os.path.expandvars("$SCRATCH/tesi_quantizzazione/c4_dataset")
    if os.path.exists(local_dataset_path):
        dataset = load_from_disk(local_dataset_path)
    else:
        dataset = load_dataset("allenai/c4", "en", split="train")
    return [text.strip() for text in dataset["text"] if text.strip()]


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("QAT richiede una GPU CUDA")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    apply_groupwise_fake_quant(model)

    texts = load_c4_train()[: args.max_train_samples]
    train_dataset = CausalLMDataset(texts, tokenizer, args.max_length)
    if not train_dataset:
        raise RuntimeError("C4 non contiene esempi di training validi")

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        gradient_checkpointing=True,
        save_strategy="no",
        logging_steps=10,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda batch: collate(tokenizer, batch),
    )
    trainer.train()

    restore_linear_layers(model)
    model = model.merge_and_unload()
    model.config.use_cache = True
    quantize_(
        model,
        Int4WeightOnlyConfig(
            group_size=GROUP_SIZE,
            int4_packing_format="tile_packed_to_4d",
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "quantized_model.pt")
    tokenizer.save_pretrained(args.output_dir)
    metadata = {
        "format": "torchao_int4",
        "base_model": args.model_path,
        "model_key": "gemma",
        "calibration": "qat_c4_groupwise",
        "dataset": "allenai/c4-en",
        "max_steps": args.max_steps,
        "max_train_samples": args.max_train_samples,
        "max_length": args.max_length,
        "group_size": GROUP_SIZE,
        "lora_target_modules": LORA_TARGET_MODULES,
        "seed": args.seed,
    }
    (args.output_dir / "ptq_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()