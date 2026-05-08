#!/bin/bash
#SBATCH --job-name=emotion_steer
#SBATCH --account=a127
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# #SBATCH --environment=  # Set path to your SLURM environment TOML file

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

EMOTIONS=("furious" "depressed" "gloomy" "panicked" "anxious" "calm" "serene" "excited" "joyful" "ecstatic")

echo "=============================================="
echo "[slurm] Job: $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "[slurm] Node: $(hostname)"
echo "[slurm] GPUs: $SLURM_GPUS_PER_NODE"
echo "=============================================="

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun --export=ALL bash -lc "
  cd ${SCRIPT_DIR}

  python3 steer_emotion_vectors.py \
    --model-name   swiss-ai/Apertus-8B-Instruct-2509 \
    --vectors-dir  ${SCRIPT_DIR}/output_apertus_apstories/emotion_vectors \
    --output-dir   ${SCRIPT_DIR}/output_steering/apertus_steering \
    --layers       1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 \
    --emotions     "${EMOTIONS[@]}" \
    --alpha        20 \
    --n-samples    5 \
    --n-new-tokens 150 \
"

echo "[steering] Done. Results in ${SCRIPT_DIR}/output_steering/apertus_steering"
