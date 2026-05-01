#!/bin/bash
#SBATCH --job-name=token_viz
#SBATCH --account=a127
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
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

# ── Apertus 8B  ·  layer 26 (valence peak, r=0.687) ─────────────────────────
echo ""
echo "[viz] Apertus 8B — layer 26"
srun --export=ALL bash -lc "
  cd ${SCRIPT_DIR}
  python3 visualize_token_activations.py \
    --model       swiss-ai/Apertus-8B-Instruct-2509 \
    --vectors-dir ${SCRIPT_DIR}/output_apertus/analysis_resid_projected_higher_layers/contrast_vectors \
    --layer       26 \
    --sentences   ${SCRIPT_DIR}/sentences.json \
    --output-dir  ${SCRIPT_DIR}/output_token_viz/apertus_l26
"

# ── Gemma 4 8B  ·  layer 16 ──────────────────────────────────────────────────
echo ""
echo "[viz] Gemma 4 8B — layer 16"
export PYTHONPATH="${SCRIPT_DIR}/pip_transformers_new:${PYTHONPATH}"
srun --export=ALL bash -lc "
  cd ${SCRIPT_DIR}
  python3 visualize_token_activations.py \
    --model       google/gemma-4-E4B-it \
    --vectors-dir ${SCRIPT_DIR}/output_gemma/analysis_resid_projected/contrast_vectors \
    --layer       16 \
    --sentences   ${SCRIPT_DIR}/sentences.json \
    --output-dir  ${SCRIPT_DIR}/output_token_viz/gemma_l16
"

echo ""
echo "[viz] Done. Figures in ${SCRIPT_DIR}/output_token_viz/"
