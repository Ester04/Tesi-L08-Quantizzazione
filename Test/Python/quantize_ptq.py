#!/usr/bin/env python3
"""PTQ offline a 4-bit con torchao per Llama-3.1-8B e Gemma-2-9B."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from torchao.quantization import Int4WeightOnlyConfig, quantize_


MODEL_IDS = {"llama": "meta-llama/Meta-Llama-3.1-8B", "gemma": "google/gemma-2-9b"}


def local_model_snapshot(model_id: str) -> Path:
    """Restituisce la snapshot locale del modello nella cache Hugging Face."""
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    snapshots_dir = hf_home / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots"
    snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir()) if snapshots_dir.exists() else []
    if not snapshots:
        raise FileNotFoundError(f"Snapshot locale non trovata per {model_id}: {snapshots_dir}")
    return snapshots[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantizzazione PTQ int4 offline")
    parser.add_argument("--model", choices=MODEL_IDS, required=True)
    parser.add_argument("--calib", choices=("minmax", "gptq"), required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_calib_samples", type=int, default=128)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset_config", default="wikitext-2-raw-v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if not torch.cuda.is_available():
        raise RuntimeError("La PTQ richiede una GPU CUDA, attesa su Leonardo")

    print(f"Caricamento dataset {args.dataset} per calibrazione...")
    local_dataset_path = os.path.expandvars("$SCRATCH/tesi_quantizzazione/wikitext_dataset")

    if os.path.exists(local_dataset_path):
        dataset = load_from_disk(local_dataset_path)["train"]
    else:
        dataset = load_dataset(args.dataset, args.dataset_config, split="train")

    texts = [text.strip() for text in dataset["text"] if text.strip()][: args.max_calib_samples]

    if not texts:
        raise RuntimeError("Il dataset di calibrazione non contiene testo valido")

    # --- ESECUZIONE QUANTIZZAZIONE ---
    if args.calib == "minmax":
        snapshot_path = local_model_snapshot(MODEL_IDS[args.model])
        print(f"Caricamento tokenizzatore e modello {args.model} da {snapshot_path}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS[args.model], local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_IDS[args.model],
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            device_map={"": 0},
        )
        print("Applicazione PTQ 4-bit: Symmetric Min-Max (Weight-Only)...")
        quantize_(
            model,
            Int4WeightOnlyConfig(
                group_size=128,
                int4_packing_format="tile_packed_to_4d",
            ),
        )

    elif args.calib == "gptq":
        try:
            from gptqmodel import GPTQModel, QuantizeConfig
        except ImportError as error:
            raise RuntimeError(
                "La calibrazione GPTQ su A100 richiede GPTQModel. Installa 'gptqmodel' "
                "nell'ambiente virtuale del progetto."
            ) from error

        print(f"Caricamento modello GPTQ {args.model}...")
        model = GPTQModel.load(
            local_model_snapshot(MODEL_IDS[args.model]),
            QuantizeConfig(bits=4, group_size=128),
        )
        print("Applicazione PTQ 4-bit: GPTQ Data-Driven in corso...")
        model.quantize(texts, batch_size=1, calibration_data_min_length=1)

    print("Salvataggio del modello e dei metadati...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.calib == "gptq":
        model.save(args.output_dir)
    else:
        torch.save(model.state_dict(), args.output_dir / "quantized_model.pt")
        tokenizer.save_pretrained(args.output_dir)

    config_data = {
        "format": "gptqmodel" if args.calib == "gptq" else "torchao_int4",
        "backend": "gptqmodel" if args.calib == "gptq" else "torchao",
        "base_model": MODEL_IDS[args.model],
        "model_key": args.model,
        "calibration": args.calib,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "max_calib_samples": len(texts),
    }

    (args.output_dir / "ptq_config.json").write_text(json.dumps(config_data, indent=2), encoding="utf-8")

    print("Operazione completata con successo. Pulizia memoria GPU...")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()