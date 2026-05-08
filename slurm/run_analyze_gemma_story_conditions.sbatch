#!/bin/bash
#SBATCH --job-name=analyze_gemma_story_cond
#SBATCH --account=a127
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# #SBATCH --environment=  # Set path to your SLURM environment TOML file

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=============================================="
echo "[slurm] Job: $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "[slurm] Node: $(hostname)"
echo "=============================================="

srun --export=ALL bash -lc "
  cd ${SCRIPT_DIR}
  python3 analyze_story_conditions.py \
    --model-name  'Gemma 4 8B' \
    --dir-a       ${SCRIPT_DIR}/output_gemma_apstories/emotion_vectors \
    --label-a     'Apertus stories' \
    --dir-b       ${SCRIPT_DIR}/output_gemma_gemstories/emotion_vectors \
    --label-b     'Gemma stories' \
    --layers      2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 \
    --output-dir  ${SCRIPT_DIR}/output_compare_gemma_story_conditions_all_layers
"

echo "[analysis] Done. Figures in ${SCRIPT_DIR}/output_compare_gemma_story_conditions_all_layers"
