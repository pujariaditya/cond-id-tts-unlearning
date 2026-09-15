#!/usr/bin/env bash
#
# setup_indextts.sh -- create the isolated IndexTTS-1.5 environment.
#
# IndexTTS-1.5 has dependencies that conflict with the Coqui stack used by XTTS /
# Tortoise, so it runs in its own virtualenv. This script creates that venv,
# installs the IndexTTS package + this repo (editable), and downloads the
# IndexTTS-1.5 checkpoint.
#
# Point run_eval.py at this interpreter for the indextts config:
#   models/index_tts/.venv/bin/python scripts/run_eval.py --config configs/indextts.yaml
#
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${REPO}/models/index_tts/.venv"

python3 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install --upgrade pip

# PyTorch (CUDA 12.4) first, then the shared eval deps, then IndexTTS-1.5 LAST so its
# own dependency pins (notably its required transformers version) win over ours.
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r "${REPO}/requirements-indextts.txt"
pip install -e "${REPO}" --no-deps
# IndexTTS-1.5 is not on PyPI; install it from its official source. Pin the git ref to
# the exact commit/tag your checkpoint matches before a public release -- newer
# releases change the library API and the transformers pin.
pip install "indextts @ git+https://github.com/index-tts/index-tts.git"

# IndexTTS-1.5 checkpoint.
python "${REPO}/scripts/download_models.py" --model indextts

echo "IndexTTS venv ready: ${VENV}"
echo "Run: ${VENV}/bin/python ${REPO}/scripts/run_eval.py --config configs/indextts.yaml"
