#!/bin/bash
# Install a newer transformers (pure Python) into a dedicated directory so it
# doesn't conflict with the aarch64 numpy in pip_packages.
set -eo pipefail

TARGET="$(cd "$(dirname "$0")"; pwd)/pip_transformers_new"
mkdir -p "${TARGET}"

echo "Installing latest transformers into ${TARGET} ..."
python3 -m pip install --target="${TARGET}" --upgrade "transformers>=4.51" accelerate

echo "Done."
echo "Add to PYTHONPATH: export PYTHONPATH=${TARGET}:\${PYTHONPATH}"
