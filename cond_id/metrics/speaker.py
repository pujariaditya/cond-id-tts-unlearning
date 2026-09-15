"""Speaker encoders for unlearning evaluation.

Two encoders are used, by design:

* **ECAPA-TDNN** (SpeechBrain) -- the *in-loop* encoder. Cheap and standard; used
  during a run but not reported by ``run_eval`` (the reported erasure comes from the
  held-out WavLM verifier below).
* **WavLM-Base-Plus-SV** (Microsoft) -- the *held-out* verifier, never used to
  drive the edit. Forgetting that transfers to this encoder is the headline
  metric, because a method can drive the in-loop encoder's similarity to zero
  yet leave the identity recoverable by a different verifier.

Both return L2-normalized embeddings so cosine similarity reduces to a dot
product.
"""

import numpy as np
import torch
import torch.nn.functional as F

from ..config import DEVICE, ECAPA_DIR, ECAPA_SOURCE
from .audio import fix_len

_ECAPA = None
_WAVLM = None
_WAVLM_FE = None
_SHIM_APPLIED = False


def _apply_transformers_shim():
    """Add ``isin_mps_friendly`` to transformers if a version combo lacks it.

    Some transformers/speechbrain combinations miss this helper; add it before
    loading either encoder. Imported lazily (not at module top level) so
    ``cond_id.metrics`` -- e.g. the numeric helpers in :mod:`forgetting` -- stays
    importable with only torch, without transformers installed.
    """
    global _SHIM_APPLIED
    if _SHIM_APPLIED:
        return
    import transformers.pytorch_utils as _ptu

    if not hasattr(_ptu, "isin_mps_friendly"):
        _ptu.isin_mps_friendly = lambda e, t: torch.isin(e, t)
    _SHIM_APPLIED = True


# Pin to an exact revision for reproducibility before a public release; ``None`` tracks
# the latest upload of ``microsoft/wavlm-base-plus-sv`` (not reproducible).
_WAVLM_REVISION = None  # TODO(release): pin to the evaluated wavlm-base-plus-sv commit


def load_ecapa():
    """Load the SpeechBrain ECAPA-TDNN speaker verifier once (cached)."""
    global _ECAPA
    if _ECAPA is None:
        _apply_transformers_shim()
        from speechbrain.inference.speaker import EncoderClassifier

        _ECAPA = EncoderClassifier.from_hparams(
            source=ECAPA_SOURCE, savedir=ECAPA_DIR, run_opts={"device": DEVICE}
        )
    return _ECAPA


def ecapa_embed(wavs: list[torch.Tensor]) -> torch.Tensor:
    """L2-normalized ECAPA-TDNN embeddings for a batch of 16 kHz waveforms.

    Args:
        wavs: List of 1-D 16 kHz waveform tensors.

    Returns:
        Tensor of shape ``(len(wavs), D)`` of unit-norm embeddings.
    """
    ecapa = load_ecapa()
    batch = torch.stack([fix_len(w.unsqueeze(0)).squeeze(0) for w in wavs]).to(DEVICE)
    with torch.no_grad():
        emb = ecapa.encode_batch(batch).squeeze(1)
    return F.normalize(emb, dim=-1)


def load_wavlm():
    """Load the held-out WavLM-Base-Plus-SV verifier once (cached).

    Returns:
        ``(model, feature_extractor)`` on :data:`~cond_id.config.DEVICE`.
    """
    global _WAVLM, _WAVLM_FE
    if _WAVLM is None:
        _apply_transformers_shim()
        from transformers import AutoFeatureExtractor, WavLMForXVector

        src = "microsoft/wavlm-base-plus-sv"
        rev = {} if _WAVLM_REVISION is None else {"revision": _WAVLM_REVISION}
        _WAVLM_FE = AutoFeatureExtractor.from_pretrained(src, **rev)
        _WAVLM = WavLMForXVector.from_pretrained(src, use_safetensors=True, **rev).to(DEVICE).eval()
    return _WAVLM, _WAVLM_FE


def wavlm_embed(wavs: list[torch.Tensor]) -> torch.Tensor:
    """L2-normalized WavLM-SV embeddings for a batch of 16 kHz waveforms.

    Args:
        wavs: List of 1-D 16 kHz waveform tensors.

    Returns:
        Tensor of shape ``(len(wavs), D)`` of unit-norm embeddings.
    """
    model, fe = load_wavlm()
    # Cap verifier input length: WavLM self-attention is O(L^2); long references
    # (e.g. LibriTTS utterances) otherwise OOM. 15 s @ 16 kHz is ample for speaker
    # identity and leaves short VCTK clips / clones untouched.
    _MAX = 15 * 16000
    audios = [w.detach().cpu().numpy().astype(np.float32)[:_MAX] for w in wavs]
    inputs = fe(audios, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        emb = model(**inputs).embeddings
    return F.normalize(emb, dim=-1)
