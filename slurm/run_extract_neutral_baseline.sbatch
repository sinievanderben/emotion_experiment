#!/bin/bash
#SBATCH --job-name=neutral_basis
#SBATCH --account=a127
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output_apertus/neutral_basis"
LOGS_DIR="${SCRIPT_DIR}/logs"

MODEL="swiss-ai/Apertus-8B-Instruct-2509"
LAYERS="12 16 18 20 22 24 26 28 30"
BATCH_SIZE=4
MAX_LENGTH=512

echo "=============================================="
echo "[slurm] Job: $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "[slurm] Node: $(hostname)"
echo "[slurm] GPUs: $SLURM_GPUS_PER_NODE"
echo "=============================================="

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOGS_DIR}"

echo "[config] Model:      ${MODEL}"
echo "[config] Output dir: ${OUTPUT_DIR}"
echo "[config] Layers:     ${LAYERS}"

export PYTHONPATH="${SCRIPT_DIR}/pip_transformers_new:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun --export=ALL bash -lc "
  cd ${SCRIPT_DIR}

  python3 extract_neutral_basis.py \
    --output-dir ${OUTPUT_DIR} \
    --layers     ${LAYERS} \
    --batch-size ${BATCH_SIZE} \
    --max-length ${MAX_LENGTH} \
    --model      ${MODEL}
"

echo ""
echo "[extract] Neutral baseline saved to: ${OUTPUT_DIR}"
