#!/usr/bin/env bash
#
# reproduce.sh -- reproduce the paper's results for one backbone.
#
# End to end: assumes the environment is set up (requirements-core.txt for
# xtts/tortoise, scripts/setup_indextts.sh for indextts) and the dataset is
# prepared. Downloads the checkpoint if needed, then runs the full pipeline
# (unlearning fine-tune -> relearn adversary [XTTS] -> held-out scoring).
#
# Usage:
#   bash scripts/reproduce.sh xtts
#   bash scripts/reproduce.sh tortoise
#   bash scripts/reproduce.sh indextts   # auto-uses the IndexTTS venv (setup_indextts.sh first)
#
# Override the interpreter with PYTHON=/path/to/python if needed.
#
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-xtts}"
SEED="${2:-0}"
PY="${PYTHON:-python3}"

case "${MODEL}" in
  xtts|tortoise|indextts) ;;
  *) echo "usage: reproduce.sh {xtts|tortoise|indextts} [seed]"; exit 1 ;;
esac

# IndexTTS-1.5's ``indextts`` library lives only in its isolated venv, so run this
# backbone under that interpreter (unless PYTHON was set explicitly).
if [ "${MODEL}" = "indextts" ] && [ -z "${PYTHON:-}" ]; then
  VENV_PY="${REPO}/models/index_tts/.venv/bin/python"
  if [ -x "${VENV_PY}" ]; then
    PY="${VENV_PY}"
  else
    echo "warning: IndexTTS venv not found at ${VENV_PY}; run scripts/setup_indextts.sh first" >&2
  fi
fi

# 1. Data (idempotent; rebuild if either the map or the audio dir is missing).
if [ ! -f "${REPO}/data/speaker_map_20.json" ] || [ ! -d "${REPO}/data/audio20" ]; then
  bash "${REPO}/scripts/download_vctk.sh"
  "${PY}" "${REPO}/scripts/prepare_vctk.py"
fi

# 2. Checkpoint (XTTS / IndexTTS need an explicit fetch; Tortoise self-downloads).
case "${MODEL}" in
  xtts)     [ -f "${REPO}/models/xtts_v2/model.pth" ] || "${PY}" "${REPO}/scripts/download_models.py" --model xtts ;;
  indextts) [ -f "${REPO}/models/index_tts/gpt.pth" ] || "${PY}" "${REPO}/scripts/download_models.py" --model indextts ;;
esac

# 3. Full pipeline -> results/<model>.json
export CD_XTTS_DIR="${REPO}/models/xtts_v2"
export CD_INDEXTTS_DIR="${REPO}/models/index_tts"
export CD_EVAL_SIZE=20        # pin the canonical 20-speaker pool (audio20/ + speaker_map_20.json)
# Expose the pip cuDNN 9 so the WavLM verifier (and optional GPU Faster-Whisper)
# find it at process start. Harmless if the package is absent.
CUDNN_LIB="$("${PY}" -c 'import nvidia.cudnn,os;print(os.path.dirname(nvidia.cudnn.__file__)+"/lib")' 2>/dev/null || true)"
if [ -n "${CUDNN_LIB}" ] && [ -d "${CUDNN_LIB}" ]; then
  export LD_LIBRARY_PATH="${CUDNN_LIB}:${LD_LIBRARY_PATH:-}"
fi
"${PY}" "${REPO}/scripts/run_eval.py" --config "${REPO}/configs/${MODEL}.yaml" \
        --seed "${SEED}" --out "${REPO}/results/${MODEL}.json"

echo "Done. Results: ${REPO}/results/${MODEL}.json"
