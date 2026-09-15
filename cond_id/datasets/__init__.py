"""VCTK speaker/reference access with a train / held-out clip partition.

Clip partition (positions within ``smap[speaker]``, ``N = len(smap[sp])``):

* TRAIN (unlearning-visible): the first ``T`` clips, ``T = train_ref_n()`` (default 32).
  :func:`ref_paths` returns only these -- the method fine-tunes on the train slice.
* CENTROID (evaluation identity reference): ``[floor(0.45*N) : +8]``.
* HELDOUT-REF (conditioning for baseline / edited clones): ``[floor(0.65*N) : +6]``.
* ATTACK-REF (relearn adversary): ``[floor(0.82*N) : +6]``, disjoint from HELDOUT-REF.

Erasure, retention and relearn are all scored on the HELD-OUT slices, which the
unlearning method never sees, so a method cannot pass by overfitting the train refs.
"""

import json
import math
import os
from pathlib import Path

from ..config import ARTIFACT_DIR

# On-disk transcript cache for VCTK reference clips (keyed by clip path). VCTK
# ships no transcripts, so each clip is transcribed once with Faster-Whisper and
# cached to ``artifacts/ref_text_cache.json``.
_REF_TEXT_CACHE: dict = {}
_REF_CACHE_PATH = ARTIFACT_DIR / "ref_text_cache.json"

#: Default size of the train-partition prefix per speaker (``CD_TRAIN_REF_N``).
_DEFAULT_TRAIN_REF_N = 32


def train_ref_n() -> int:
    """Number of train-partition clips per speaker (``CD_TRAIN_REF_N``, default 32).

    This is the size ``T`` of the train prefix. :func:`ref_paths` clamps to this
    many clips; :func:`held_clips` draws from strictly higher positions.
    """
    try:
        t = int(os.environ.get("CD_TRAIN_REF_N", str(_DEFAULT_TRAIN_REF_N)))
    except (TypeError, ValueError):
        t = _DEFAULT_TRAIN_REF_N
    return max(1, t)


def pool(n: int = 20):
    """Return the first ``n`` speakers of the configured VCTK subset (deterministic)."""
    from ..config import SUBSETS

    size = os.environ.get("CD_EVAL_SIZE", "20")
    return list(SUBSETS.get(size, SUBSETS["20"])["speakers"])[:n]


def _smap() -> dict:
    """Load the VCTK ``{speaker: [global_clip_index, ...]}`` map."""
    from ..config import SPEAKER_MAP

    return json.loads(Path(SPEAKER_MAP).read_text())


def _clip_path(idx) -> str:
    """Map a global VCTK clip index to its ``.flac`` path."""
    from ..config import AUDIO_DIR

    return str(AUDIO_DIR / (str(idx) + ".flac"))


def train_indices(speaker: str):
    """Return the train-partition global clip indices for ``speaker``
    (the first ``train_ref_n()`` positions of ``smap[speaker]``)."""
    return list(_smap()[speaker][: train_ref_n()])


def train_paths(speaker: str):
    """Return the train-partition clip paths for ``speaker``."""
    return [_clip_path(i) for i in train_indices(speaker)]


def ref_paths(speaker: str, k: int):
    """Return up to ``k`` train-partition reference paths for ``speaker``
    (clamped to ``train_ref_n()`` so it can never reach a held-out clip)."""
    kk = max(0, min(int(k), train_ref_n()))
    return train_paths(speaker)[:kk]


def _held_window(kind: str, n_total: int):
    """Return the ``(start, length)`` window for a held-out partition ``kind``."""
    if kind == "centroid":
        return int(math.floor(0.45 * n_total)), 8
    if kind == "heldout":
        return int(math.floor(0.65 * n_total)), 6
    if kind == "attack":
        return int(math.floor(0.82 * n_total)), 6
    raise ValueError(
        f"held_clips kind must be one of {{'centroid','heldout','attack'}}, got {kind!r}"
    )


def held_clips(speaker: str, kind: str, n=None, seed: int = 0):
    """Return held-out clip paths for ``speaker`` and partition ``kind``.

    The returned clips are strictly disjoint from the train slice and from each
    other (across ``kind``), so evaluation (identity centroid / clone conditioning /
    relearn attack) never touches audio the unlearning method trained on.

    Args:
        speaker: VCTK speaker id.
        kind: one of ``"centroid"``, ``"heldout"``, ``"attack"``.
        n: how many clip paths to return (``None`` = the whole window).
        seed: reserved for deterministic sub-sampling; the window is returned in
            map order (deterministic) today.

    Returns:
        List of ``.flac`` paths.
    """
    idxs = _smap()[speaker]
    n_total = len(idxs)
    start, length = _held_window(kind, n_total)
    start = min(start, max(0, n_total - length))
    window = list(idxs[start : start + length])
    if n is not None:
        window = window[: max(0, int(n))]
    return [_clip_path(i) for i in window]


def _load_ref_cache():
    global _REF_TEXT_CACHE
    if not _REF_TEXT_CACHE and _REF_CACHE_PATH.exists():
        _REF_TEXT_CACHE = json.loads(_REF_CACHE_PATH.read_text())
    return _REF_TEXT_CACHE


def _store_ref_cache(key: str, text: str) -> str:
    _REF_TEXT_CACHE[key] = text
    _REF_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REF_CACHE_PATH.write_text(json.dumps(_REF_TEXT_CACHE, indent=2))
    return text


def _transcribe_path(path) -> str:
    """Transcribe a wav/flac ``path`` with Faster-Whisper (no caching here)."""
    import soundfile as sf
    import torch

    from ..metrics.intelligibility import load_faster_whisper

    fw = load_faster_whisper()
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import torchaudio

        audio_t = torch.from_numpy(audio).unsqueeze(0)
        audio_t = torchaudio.functional.resample(audio_t, sr, 16000)
        audio = audio_t.squeeze(0).numpy()
    segments, _ = fw.transcribe(audio.astype("float32"), language="en", beam_size=5)
    return " ".join(s.text.strip() for s in segments).strip() or "Hello."


def ref_text(path: str) -> str:
    """Return the reference transcript for any VCTK clip ``path`` (transcribe once,
    cache by path). Resolves for both train and held-out clips."""
    cache = _load_ref_cache()
    key = "path:" + str(path)
    if key in cache:
        return cache[key]
    return _store_ref_cache(key, _transcribe_path(path))
