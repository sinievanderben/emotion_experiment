#!/bin/bash
#SBATCH --job-name=analyze_cross_model_apstories
#SBATCH --account=a127
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)

echo "=============================================="
echo "[slurm] Job: $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "[slurm] Node: $(hostname)"
echo "=============================================="

srun --export=ALL bash -lc "
  cd ${SCRIPT_DIR}
  python3 analyze_cross_model_geometry.py \
    --gemma-dir   ${SCRIPT_DIR}/output_gemma/emotion_vectors \
    --apertus-dir ${SCRIPT_DIR}/output_apertus/emotion_vectors \
    --output-dir  ${SCRIPT_DIR}/output_cross_model_apstories/emotion_vectors
"

echo "[analysis] Done. Figures in ${SCRIPT_DIR}/output_cross_model_apstories/emotion_vectors"
