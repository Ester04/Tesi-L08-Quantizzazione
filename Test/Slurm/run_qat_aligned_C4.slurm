#!/bin/bash
#SBATCH --job-name=qat_aligned
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs outputs

module purge
module load profile/deeplrn
module load python/3.11
source "$SCRATCH/tesi_quantizzazione/env_tesi/bin/activate"

export HF_HOME="$SCRATCH/tesi_quantizzazione/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=42

QAT_OUTPUT="outputs/gemma_qat_aligned_seed42_C4"
RESULTS_FILE="results_qat_aligned_C4.csv"

echo "=== INIZIO QAT ALLINEATA SU C4 ==="
python train_qat_aligned.py \
  --model_path "google/gemma-2-9b" \
  --output_dir "$QAT_OUTPUT" \
  --max_steps 1000 \
  --max_train_samples 8000 \
  --max_length 512 \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 \
  --seed 42

echo "=== INIZIO VALUTAZIONE QAT ALLINEATA SU C4 - test su WIKITEXT ==="
python evaluate.py \
  --model_path "$QAT_OUTPUT" \
  --mode ptq \
  --results_file "$RESULTS_FILE"

echo "=== QAT ALLINEATA COMPLETATA SU C4 - test su WIKITEXT ==="
