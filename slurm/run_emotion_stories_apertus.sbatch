#!/bin/bash
#SBATCH --job-name=emotion_stories
#SBATCH --account=a127
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# #SBATCH --environment=  # Set path to your SLURM environment TOML file

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"
LOGS_DIR="${SCRIPT_DIR}/logs"

PROMPT_FILE="${SCRIPT_DIR}/prompts/story_prompt.txt"
EMOTIONS_FILE="${SCRIPT_DIR}/prompts/emotions.txt"
TOPICS_FILE="${SCRIPT_DIR}/prompts/topics.txt"

MODEL_NAME="swiss-ai/Apertus-8B-Instruct-2509"

STORIES_PER_PROMPT=3
PROMPTS_PER_EMOTION=3
MAX_NEW_TOKENS=2048
TEMPERATURE=0.8

EMOTION_FLAG="--all_emotions"

echo "[slurm] Job: $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "[slurm] Node: $(hostname)"
echo "[slurm] GPUs: $SLURM_GPUS_PER_NODE"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOGS_DIR}"

echo "[config] Model: $MODEL_NAME"
echo "[config] Output: $OUTPUT_DIR"

echo ""
echo "[generation] Starting emotion story generation"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun --export=ALL bash -lc "
  cd ${SCRIPT_DIR}
  python3 generate_emotion_stories.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name ${MODEL_NAME} \
    --prompt_file ${PROMPT_FILE} \
    --emotions_file ${EMOTIONS_FILE} \
    --topics_file ${TOPICS_FILE} \
    --stories_per_prompt ${STORIES_PER_PROMPT} \
    --prompts_per_emotion ${PROMPTS_PER_EMOTION} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature ${TEMPERATURE} \
    ${EMOTION_FLAG} \
    --resume
"

echo "[generation] Completed"
echo "[generation] Output saved to: ${OUTPUT_DIR}"

echo ""
echo "[stats] Dataset statistics:"
srun --export=ALL bash -lc "
  cd ${SCRIPT_DIR}
  python3 generate_emotion_stories.py --stats ${OUTPUT_DIR}/stories.jsonl
"
