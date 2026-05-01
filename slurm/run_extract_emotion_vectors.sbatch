#!/bin/bash
#SBATCH --job-name=emotion_12
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
OUTPUT_DIR="${SCRIPT_DIR}/output/emotion_vectors"
LOGS_DIR="${SCRIPT_DIR}/logs"

STORIES_FILE="${SCRIPT_DIR}/output/stories_dedup.jsonl"

# SAE layers to extract (must have trained SAEs in sae_apertus/)
# we do one layer at a time to save memory; can be parallelized if needed
LAYERS="13 14 15 21 22 23 24 30" 

# Stories are processed in batches; reduce if you hit OOM
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

echo "[config] Stories file:  ${STORIES_FILE}"
echo "[config] Output dir:    ${OUTPUT_DIR}"
echo "[config] Layers:        ${LAYERS}"
echo "[config] Batch size:    ${BATCH_SIZE}"
echo "[config] Max length:    ${MAX_LENGTH}"

# ============================================================================
# Extract emotion vectors
# ============================================================================

echo ""
echo "[extract] Starting activation extraction"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun --export=ALL bash -lc "
  cd ${SCRIPT_DIR}

  python3 extract_emotion_vectors.py \
    --stories-file ${STORIES_FILE} \
    --output-dir   ${OUTPUT_DIR} \
    --layers       ${LAYERS} \
    --batch-size   ${BATCH_SIZE} \
    --max-length   ${MAX_LENGTH}
"

echo ""
echo "[extract] Completed"
echo "[extract] Vectors saved to: ${OUTPUT_DIR}"
