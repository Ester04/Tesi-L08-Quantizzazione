#!/bin/bash
#SBATCH --job-name=phase2_qat
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs outputs

# ---------------------------------------------------------
# 1. INIZIALIZZAZIONE AMBIENTE CINECA
# ---------------------------------------------------------
module purge
module load profile/deeplrn
module load python/3.11


# Attiva il tuo ambiente virtuale
source $SCRATCH/tesi_quantizzazione/env_tesi/bin/activate

# ---------------------------------------------------------
# 2. CONFIGURAZIONE CACHE OFFLINE
# ---------------------------------------------------------
export HF_HOME="$SCRATCH/tesi_quantizzazione/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# ---------------------------------------------------------
# 3. ADDESTRAMENTO QAT
# ---------------------------------------------------------
echo "=== INIZIO FASE 2: TRAINING QAT SU GEMMA-2 ==="
QAT_OUTPUT="outputs/gemma_qat"

# Usiamo l'ID standard di Hugging Face, ci penserà la cache offline a trovarlo
python train_qat.py \
  --model_path "google/gemma-2-9b" \
  --output_dir "$QAT_OUTPUT"

# ---------------------------------------------------------
# 4. VALUTAZIONE QAT
# ---------------------------------------------------------
echo "=== INIZIO VALUTAZIONE QAT ==="

# Usiamo --mode ptq perché il nostro train_qat.py esporta il modello 
# in formato torchao_int4 compresso (quantized_model.pt e ptq_config.json)
python evaluate.py \
  --model_path "$QAT_OUTPUT" \
  --mode ptq \
  --results_file "results_phase2.csv"

echo "=== FASE 2 COMPLETATA CON SUCCESSO ==="