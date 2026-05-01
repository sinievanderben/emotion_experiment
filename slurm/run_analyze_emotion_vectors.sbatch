#!/bin/bash
#SBATCH --job-name=gemma_emotion_analysis_resid
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

OUTPUT_DIR="${SCRIPT_DIR}/output_gemma/analysis_resid_projected_higher_layers"


srun --export=ALL bash -lc "
  cd ${SCRIPT_DIR}

  python3 analyze_emotion_vectors.py \
    --vectors-dir  ${SCRIPT_DIR}/output_gemma/emotion_vectors \
    --output-dir   ${OUTPUT_DIR} \
    --vad-csv      ${SCRIPT_DIR}/emotion_valence_arousal_nrc.csv \
    --layers       35 36 37 38 39 40\
    --heatmap-layer 35 \
    --kmeans-k     10 \
    --use-resid
"

echo "[analysis] Done. Figures in ${OUTPUT_DIR}"
