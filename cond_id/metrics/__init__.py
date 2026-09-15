"""Evaluation metrics for speaker unlearning.

Submodules:

* :mod:`~cond_id.metrics.speaker` -- speaker encoders (ECAPA-TDNN in-loop,
  WavLM-Base-Plus-SV held-out) and L2-normalized embeddings.
* :mod:`~cond_id.metrics.forgetting` -- spk-ZRF and cosine-similarity scoring.
* :mod:`~cond_id.metrics.intelligibility` -- Faster-Whisper WER.
* :mod:`~cond_id.metrics.audio` -- resampling / length helpers.
"""

from .audio import fix_len, load_wav_16k, to_16k
from .forgetting import cosine_to_centroid, spk_zrf
from .intelligibility import load_faster_whisper, transcribe, word_error_rate
from .speaker import ecapa_embed, load_ecapa, load_wavlm, wavlm_embed

__all__ = [
    "load_ecapa",
    "ecapa_embed",
    "load_wavlm",
    "wavlm_embed",
    "spk_zrf",
    "cosine_to_centroid",
    "load_faster_whisper",
    "transcribe",
    "word_error_rate",
    "to_16k",
    "fix_len",
    "load_wav_16k",
]
