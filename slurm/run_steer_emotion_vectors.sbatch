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

set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)

echo "=============================================="
echo "[slurm] Job: $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "[slurm] Node: $(hostname)"
echo "[slurm] GPUs: $SLURM_GPUS_PER_NODE"
echo "=============================================="

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun --export=ALL bash -lc "
  cd ${SCRIPT_DIR}

  python3 steer_emotion_vectors.py \
    --vectors-dir  ${SCRIPT_DIR}/output/emotion_vectors \
    --sae-config   ${SCRIPT_DIR}/sae_config.json \
    --output-dir   ${SCRIPT_DIR}/output/steering \
    --all-layers \
    --alpha        20 \
    --n-samples    10 \
    --n-new-tokens 80
"

echo "[steering] Done. Results in ${SCRIPT_DIR}/output/steering/"
