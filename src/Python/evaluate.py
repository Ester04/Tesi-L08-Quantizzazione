#!/usr/bin/env python3
"""Benchmark offline di latenza, throughput e memoria per fase 1/2."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Int4WeightOnlyConfig, quantize_


def verifica_hardware_e_quantizzazione(model):
    print("\n--- INIZIO CHECK HARDWARE E TORCHAO ---")
    assert torch.cuda.is_available(), "ERRORE: CUDA non disponibile, stai andando su CPU!"
    device = next(model.parameters()).device
    print(f"Dispositivo del modello: {device}")
    assert device.type == 'cuda', f"ERRORE: Il modello è su {device}, non su GPU!"

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            peso = module.weight
            nome_classe_peso = peso.__class__.__name__
            print(f"Analisi del layer: {name} | Classe: {nome_classe_peso}")
            if nome_classe_peso in ['Parameter', 'Tensor'] and peso.dtype not in [torch.int8, torch.uint8]:
                print("ALLARME: Il tensore è FP16. La quantizzazione PTQ potrebbe aver fallito.")
            else:
                print("SUCCESSO: Il tensore è quantizzato!")
            break
    print("--- FINE CHECK ---\n")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valutazione LLM su una sola GPU")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--mode", choices=("base", "ptq", "qat"), required=True)
    parser.add_argument("--results_file", type=Path, default=Path("results_phase1.csv"))
    parser.add_argument("--num_prompts", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset_config", default="wikitext-2-raw-v1")
    parser.add_argument("--backend", choices=("auto", "torchao", "bitsandbytes"), default="auto",
                        help="Backend 4-bit opzionale per modelli base; PTQ torchao usa sempre torchao")
    parser.add_argument("--no_compile", action="store_true",
                        help="Disabilita torch.compile, utile durante il debugging")
    parser.add_argument("--perplexity_samples", type=int, default=20,
                        help="Numero di prompt usati per calcolare la perplexity (0 per disabilitare)")
    return parser.parse_args()


def load_model(args: argparse.Namespace):
    if args.mode == "ptq":
        config = json.loads((args.model_path / "ptq_config.json").read_text(encoding="utf-8"))
        base_id = config["base_model"]
    else:
        config = {}
        base_id = str(args.model_path)
    tokenizer_path = args.model_path if args.mode in ("ptq", "qat") else base_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                               local_files_only=True)
    load_kwargs = {"local_files_only": True, "low_cpu_mem_usage": True, "device_map": {"": 0}}
    if args.mode == "ptq" and config["calibration"] == "gptq":
        # Carica direttamente con GPTQModel: il quantizer GPTQ di transformers/optimum
        # richiede gptqmodel>=7.0.0, non disponibile in questo ambiente.
        from gptqmodel import GPTQModel
        wrapper = GPTQModel.load(str(args.model_path), device="cuda:0")
        model = wrapper.model
        model.eval()
        return model, tokenizer

    backend = "torchao" if args.mode == "ptq" else args.backend
    if backend == "bitsandbytes":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(base_id, **load_kwargs)
    if args.mode == "ptq":
        quantization_config = Int4WeightOnlyConfig(group_size=128)
        if config["calibration"] != "gptq":
            quantization_config = Int4WeightOnlyConfig(
                group_size=128,
                int4_packing_format="tile_packed_to_4d",
            )
        quantize_(model, quantization_config)
        state = torch.load(args.model_path / "quantized_model.pt", map_location="cuda:0", weights_only=True)
        model.load_state_dict(state, strict=True)
    model.eval()
    return model, tokenizer


def generate_one(model, tokenizer, prompt: str, max_new_tokens: int) -> tuple[float, float, int]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True)
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
        cache = outputs.past_key_values
    torch.cuda.synchronize()
    ttft = time.perf_counter() - start
    generated = 1
    decode_start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(max_new_tokens - 1):
            outputs = model(input_ids=next_token, past_key_values=cache, use_cache=True)
            cache = outputs.past_key_values
            next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
            generated += 1
            if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                break
    torch.cuda.synchronize()
    return ttft, generated / max(time.perf_counter() - decode_start, 1e-9), generated


def compute_perplexity(model, tokenizer, texts: list[str], max_length: int = 512) -> float:
    total_nll, total_tokens = 0.0, 0
    with torch.inference_mode():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = inputs["input_ids"].to("cuda:0")
            if input_ids.shape[1] < 2:
                continue
            outputs = model(input_ids=input_ids, labels=input_ids)
            num_tokens = input_ids.shape[1] - 1
            total_nll += outputs.loss.item() * num_tokens
            total_tokens += num_tokens
    if total_tokens == 0:
        return float("nan")
    return float(np.exp(total_nll / total_tokens))


def append_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Il benchmark richiede esattamente una GPU CUDA visibile")
    
    local_dataset_path = os.path.expandvars("$SCRATCH/tesi_quantizzazione/wikitext_dataset")
    if os.path.exists(local_dataset_path):
        dataset = load_from_disk(local_dataset_path)["test"]
    else:
        dataset = load_dataset(args.dataset, args.dataset_config, split="test", download_mode="reuse_cache_if_exists")
    prompts = [text.strip() for text in dataset["text"] if text.strip()][:args.num_prompts]
    if len(prompts) < args.num_prompts:
        raise RuntimeError(f"Il dataset contiene solo {len(prompts)} prompt validi, richiesti {args.num_prompts}")
    
    model, tokenizer = load_model(args)
    
    # 1. Eseguiamo il check hardware
    verifica_hardware_e_quantizzazione(model)
    
    # 2. Compiliamo il modello per fondere i kernel CUDA (Fondamentale per le prestazioni)
    if args.no_compile:
        print("torch.compile disabilitato per il debug")
    else:
        print("Compilazione del modello con torch.compile in corso (richiederà qualche minuto)...")
        model = torch.compile(model)
        print("Compilazione completata!")

    print(f"Inizio warm-up su {args.warmup} prompt...")
    for prompt in prompts[:args.warmup]:
        generate_one(model, tokenizer, prompt, args.max_new_tokens)
    
    print("Inizio benchmark reale...")
    torch.cuda.reset_peak_memory_stats()
    measurements = [generate_one(model, tokenizer, prompt, args.max_new_tokens) for prompt in prompts]
    
    ttft = np.array([item[0] for item in measurements])
    throughput = np.array([item[1] for item in measurements])

    perplexity = float("nan")
    if args.perplexity_samples > 0:
        print(f"Calcolo perplexity su {args.perplexity_samples} prompt...")
        perplexity = compute_perplexity(model, tokenizer, prompts[:args.perplexity_samples])
        print(f"Perplexity: {perplexity:.4f}")

    append_result(args.results_file, {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_path": str(args.model_path), 
        "mode": args.mode, 
        "num_prompts": len(measurements),
        "ttft_mean_s": float(ttft.mean()), 
        "ttft_std_s": float(ttft.std(ddof=1)),
        "tokens_sec_mean": float(throughput.mean()), 
        "tokens_sec_std": float(throughput.std(ddof=1)),
        "peak_vram_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "perplexity": perplexity
    })
    print(f"Risultati salvati in {args.results_file}")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
