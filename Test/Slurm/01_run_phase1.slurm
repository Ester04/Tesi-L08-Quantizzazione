#!/bin/bash
#SBATCH --job-name=phase1_ptq
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# Assicurati che le cartelle usate da SLURM e dallo script esistano
mkdir -p logs outputs "$SCRATCH/tesi_quantizzazione/hf_cache"

# ---------------------------------------------------------
# 1. INIZIALIZZAZIONE AMBIENTE CINECA
# ---------------------------------------------------------
module purge
module load profile/deeplrn
module load python

source $SCRATCH/tesi_quantizzazione/env_tesi/bin/activate

# ---------------------------------------------------------
# 2. CONFIGURAZIONE CACHE OFFLINE
# ---------------------------------------------------------
export HF_HOME="$SCRATCH/tesi_quantizzazione/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "SCRATCH=$SCRATCH"
echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
for repo in models--meta-llama--Meta-Llama-3.1-8B models--google--gemma-2-9b; do
  if [[ ! -d "$HF_HUB_CACHE/$repo/snapshots" ]] || [[ -z "$(ls -A "$HF_HUB_CACHE/$repo/snapshots" 2>/dev/null)" ]]; then
    echo "ERRORE: cache HF mancante per $repo in $HF_HUB_CACHE/$repo/snapshots" >&2
    echo "Scarica il modello da un login node (con accesso a internet) prima di rilanciare il job." >&2
    exit 1
  fi
done

# ---------------------------------------------------------
# 3. FASE PTQ (Generazione Modelli Quantizzati)
# ---------------------------------------------------------
echo "=== INIZIO FASE PTQ ==="
for model in llama gemma; do
  for calib in minmax gptq; do
    echo "-> Esecuzione PTQ su $model con calibrazione $calib"
    python quantize_ptq.py --model "$model" --calib "$calib" \
      --output_dir "outputs/${model}_ptq_${calib}"
  done
done

# ---------------------------------------------------------
# 4. FASE VALUTAZIONE (Benchmark TTFT, Token/sec, VRAM)
# ---------------------------------------------------------
echo "=== INIZIO FASE VALUTAZIONE ==="
RESULTS="results_phase1.csv"

for model in llama gemma; do
  if [[ "$model" == "llama" ]]; then
    BASE_PATH="meta-llama/Meta-Llama-3.1-8B"
  else
    BASE_PATH="google/gemma-2-9b"
  fi
  
  echo "-> Valutazione Modello Base: $model"
  python evaluate.py --model_path "$BASE_PATH" --mode base \
    --results_file "$RESULTS"
  
  for calib in minmax gptq; do
    echo "-> Valutazione Modello PTQ: $model ($calib)"
    python evaluate.py --model_path "outputs/${model}_ptq_${calib}" --mode ptq \
      --results_file "$RESULTS"
  done
done

echo "=== TUTTI I JOB DELLA FASE 1 SONO COMPLETATI ==="
