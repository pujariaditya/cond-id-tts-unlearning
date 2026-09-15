"""Shared configuration for cond-ID.

Self-contained: all data and model paths resolve inside the repository (or an
explicit override via environment variables). Run ``scripts/download_vctk.sh``
then ``scripts/prepare_vctk.py`` to populate ``data/`` before evaluation.

Environment variables read here (all optional; defaults make the repo runnable
as-is). The complete cross-module list is documented in ``README.md``:
  COND_ID_DATA_DIR   Root of the prepared VCTK data (default: <repo>/data)
  CD_EVAL_SIZE        Speaker-subset size: 20 (default), 10, or 5
  CD_FORGET           Comma-separated forget speakers (default: p272)
  CD_RETAIN           Comma-separated retain speakers (default: rest of subset)
  CD_CPU              Set to 1 to force CPU
  ECAPA_SOURCE        speechbrain source id or local dir for the ECAPA encoder
  COND_ID_ECAPA_DIR  Local cache dir for the downloaded ECAPA checkpoint

Other modules read additional variables -- asset paths ``CD_XTTS_DIR`` /
``CD_INDEXTTS_DIR`` (models), WER backend ``CD_FW_DEVICE`` / ``CD_FW_MODEL``
(metrics), and train-slice size ``CD_TRAIN_REF_N`` (datasets). See
``README.md`` ("Configuration") for the full reference.
"""

import os
from pathlib import Path

# Repository root = parent of this package directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("COND_ID_DATA_DIR", str(PROJECT_ROOT / "data")))

# Canonical VCTK speaker subsets. The 5- and 10-speaker sets are subsets of the
# 20-speaker pool used for the cross-validated results.
SUBSETS = {
    "5": {
        "audio_subdir": "audio",
        "speaker_map": "speaker_map.json",
        "speakers": ["p231", "p238", "p272", "p294", "p318"],
    },
    "10": {
        "audio_subdir": "audio10",
        "speaker_map": "speaker_map_10.json",
        "speakers": [
            "p231",
            "p238",
            "p246",
            "p272",
            "p275",
            "p280",
            "p294",
            "p302",
            "p303",
            "p318",
        ],
    },
    "20": {
        "audio_subdir": "audio20",
        "speaker_map": "speaker_map_20.json",
        "speakers": [
            "p225",
            "p231",
            "p238",
            "p243",
            "p246",
            "p252",
            "p260",
            "p261",
            "p263",
            "p272",
            "p273",
            "p275",
            "p280",
            "p283",
            "p294",
            "p302",
            "p303",
            "p318",
            "p336",
            "p376",
        ],
    },
}

_eval_size = os.environ.get("CD_EVAL_SIZE", "20")
# Any unrecognized size falls back to the 20-speaker SUPERSET (not the 5-subset), to
# match datasets.pool(), which also defaults to speaker_map_20.json. The SPEAKER_MAP /
# AUDIO_DIR selected here must contain every pooled speaker or ref_paths() would raise
# KeyError (e.g. 'p243' is absent from the 5-speaker map); speaker_map_20.json + audio20/
# are the only assets that contain the full pool.
_subset = SUBSETS.get(_eval_size, SUBSETS["20"])
AUDIO_DIR = DATA_DIR / _subset["audio_subdir"]
SPEAKER_MAP = DATA_DIR / _subset["speaker_map"]
EVAL_SPEAKERS = list(_subset["speakers"])

# ECAPA-TDNN speaker encoder: pulled from the HuggingFace Hub by default and
# cached inside the repo, so no machine-specific path is required.
ECAPA_SOURCE = os.environ.get("ECAPA_SOURCE", "speechbrain/spkrec-ecapa-voxceleb")
ECAPA_DIR = os.environ.get("COND_ID_ECAPA_DIR", str(PROJECT_ROOT / ".cache" / "ecapa"))

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if os.environ.get("CD_CPU") != "1" else "cpu"

_forget_env = os.environ.get("CD_FORGET", "p272")
FORGET_SPEAKERS = [s.strip() for s in _forget_env.split(",") if s.strip()]
_retain_env = os.environ.get("CD_RETAIN", "")
if _retain_env:
    RETAIN_SPEAKERS = [s.strip() for s in _retain_env.split(",") if s.strip()]
    EVAL_SPEAKERS = sorted(set(FORGET_SPEAKERS + RETAIN_SPEAKERS))
else:
    RETAIN_SPEAKERS = [s for s in EVAL_SPEAKERS if s not in FORGET_SPEAKERS]

# Verify harness: minimal sentences to keep wall-clock under ~30s/iter on an RTX 8000.
VERIFY_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning has transformed countless industries.",
    "Climate change requires immediate global action.",
]


def set_determinism(seed: int = 0, strict: bool = False) -> None:
    """Pin every RNG the synthesis + evaluation pipeline draws from, for reproducible results.

    Without this, results vary run-to-run: XTTS decodes with temperature sampling and nothing seeds torch/numpy
    globally otherwise. Call ONCE at the top of an
    evaluation, before building hosts or importing heavy modules.

    Args:
        seed: the master seed.
        strict: also request fully deterministic CUDA kernels
            (``torch.use_deterministic_algorithms``). Off by default because some XTTS /
            Tortoise / IndexTTS GPU ops
            lack deterministic implementations; ``warn_only`` keeps it non-fatal when enabled.
    """
    import random as _random

    import numpy as _np
    import torch as _torch

    # NOTE: PYTHONHASHSEED only affects child processes / interpreters started AFTER this point;
    # the method takes its seed as an explicit argument, so it does not rely on this.
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    _random.seed(seed)
    _np.random.seed(seed % (2**31 - 1))
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)
    _torch.backends.cudnn.benchmark = False
    _torch.backends.cudnn.deterministic = True
    if strict:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            _torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
