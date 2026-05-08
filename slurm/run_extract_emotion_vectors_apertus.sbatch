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
# #SBATCH --environment=  # Set path to your SLURM environment TOML file

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output_apertus/emotion_vectors"
LOGS_DIR="${SCRIPT_DIR}/logs"

STORIES_FILE="${SCRIPT_DIR}/output_apertus_stories/stories_dedup.jsonl"

# we do one layer at a time to save memory; can be parallelized if needed
LAYERS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 "

# Stories are processed in batches; reduce if you hit OOM
BATCH_SIZE=4
MAX_LENGTH=512

echo "=============================================="
echo "[slurm] Job: $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "[slurm] Node: $(hostname)"
echo "[slurm] GPUs: $SLURM_GPUS_PER_NODE"
echo "[slurm] Time limit: $SLURM_TIMELIMIT"
echo "=============================================="

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOGS_DIR}"

echo "[config] Stories file:  ${STORIES_FILE}"
echo "[config] Output dir:    ${OUTPUT_DIR}"
echo "[config] Layers:        ${LAYERS}"
echo "[config] Batch size:    ${BATCH_SIZE}"
echo "[config] Max length:    ${MAX_LENGTH}"

echo ""
echo "[extract] Starting activation extraction"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


srun --export=ALL bash -lc "
  cd ${SCRIPT_DIR}

  python3 extract_emotion_vectors.py \
    --stories-file ${STORIES_FILE} \
    --output-dir   ${OUTPUT_DIR} \
    --layers       1 2 3 4 5 6 7 8 9 \
    --batch-size   ${BATCH_SIZE} \
    --max-length   ${MAX_LENGTH} \
    --model swiss-ai/Apertus-8B-Instruct-2509
    --debug-pooled-sim
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
