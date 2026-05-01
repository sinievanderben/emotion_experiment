#!/bin/bash
#SBATCH --job-name=emotion_gemma
#SBATCH --account=a127
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
OUTPUT_DIR="${SCRIPT_DIR}/output_gemma/emotion_vectors"
LOGS_DIR="${SCRIPT_DIR}/logs"

STORIES_FILE="${SCRIPT_DIR}/output/stories_dedup.jsonl"

MODEL="google/gemma-4-E4B-it"

# Residual-stream layers to extract (no SAEs for Gemma).
# Gemma-4-E4B-it has 42 layers (indices 0-41); adjust if needed.
LAYERS="17 19 27 28 29"

BATCH_SIZE=4
MAX_LENGTH=512

# ============================================================================
# SLURM Info
# ============================================================================

echo "=============================================="
echo "[slurm] Job: $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "[slurm] Node: $(hostname)"
echo "[slurm] GPUs: $SLURM_GPUS_PER_NODE"
echo "[slurm] Time limit: $SLURM_TIMELIMIT"
echo "=============================================="

# ============================================================================
# Setup
# ============================================================================

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOGS_DIR}"

echo "[config] Model:         ${MODEL}"
echo "[config] Stories file:  ${STORIES_FILE}"
echo "[config] Output dir:    ${OUTPUT_DIR}"
echo "[config] Layers:        ${LAYERS}"
echo "[config] Batch size:    ${BATCH_SIZE}"
echo "[config] Max length:    ${MAX_LENGTH}"

# ============================================================================
# Extract emotion vectors (residual stream only, no SAEs)
# ============================================================================

echo ""
echo "[extract] Starting activation extraction"

# Newer transformers in its own dir — avoids the aarch64 numpy in pip_packages
export PYTHONPATH="${SCRIPT_DIR}/pip_transformers_new:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun --export=ALL bash -lc "
  cd ${SCRIPT_DIR}

  python3 extract_emotion_vectors.py \
    --model        ${MODEL} \
    --stories-file ${STORIES_FILE} \
    --output-dir   ${OUTPUT_DIR} \
    --layers       ${LAYERS} \
    --batch-size   ${BATCH_SIZE} \
    --max-length   ${MAX_LENGTH}
"

echo ""
echo "[extract] Completed"
echo "[extract] Vectors saved to: ${OUTPUT_DIR}"
