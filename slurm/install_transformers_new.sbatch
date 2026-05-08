#!/bin/bash
#SBATCH --job-name=install_transformers
#SBATCH --account=a127
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# #SBATCH --environment=  # Set path to your SLURM environment TOML file

set -eo pipefail

TARGET="${SCRIPT_DIR}/pip_transformers_new"

echo "Installing latest transformers into ${TARGET} ..."

srun --export=ALL bash -lc "
  # Clean slate
  rm -rf ${TARGET}
  mkdir -p ${TARGET}

  # Install transformers + pure-Python deps; accelerate is needed for device_map
  pip install --target=${TARGET} --upgrade 'transformers>=4.51' accelerate

  # Remove torch (and related) — the container's torch must be used instead,
  # otherwise it conflicts with the container's torchvision C extensions.
  rm -rf ${TARGET}/torch         ${TARGET}/torch-*.dist-info
  rm -rf ${TARGET}/torchvision   ${TARGET}/torchvision-*.dist-info
  rm -rf ${TARGET}/torchaudio    ${TARGET}/torchaudio-*.dist-info
  rm -rf ${TARGET}/triton        ${TARGET}/triton-*.dist-info
  rm -rf ${TARGET}/nvidia*       ${TARGET}/nvidia_*
  rm -rf ${TARGET}/cuda*         ${TARGET}/cuda_*

  echo 'Remaining top-level packages:'
  ls ${TARGET}
"

echo "Done."
