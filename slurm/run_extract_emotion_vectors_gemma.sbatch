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
# #SBATCH --environment=  # Set path to your SLURM environment TOML file

set -eo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output_gemma/emotion_vectors"
LOGS_DIR="${SCRIPT_DIR}/logs"

STORIES_FILE="${SCRIPT_DIR}/output_gemma_stories/stories_dedup.jsonl"

# Gemma-2-9b-it has 42 layers (0-41)
LAYERS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41"

# Stories are processed in batches; reduce if you hit OOM
BATCH_SIZE=4
MAX_LENGTH=512
MODEL="google/gemma-4-E4B-it"

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
    --max-length   ${MAX_LENGTH} \
    --model ${MODEL}
"

echo ""
echo "[extract] Completed"
echo "[extract] Vectors saved to: ${OUTPUT_DIR}"


# srun --export=ALL bash -lc "
#   cd ${SCRIPT_DIR}

#   python3 extract_emotion_vectors.py \
#     --stories-file ${STORIES_FILE} \
#     --output-dir   ${OUTPUT_DIR} \
#     --layers       ${LAYERS} \
#     --batch-size   ${BATCH_SIZE} \
#     --max-length   ${MAX_LENGTH}
# "

# echo ""
# echo "[extract] Completed"
# echo "[extract] Vectors saved to: ${OUTPUT_DIR}"
