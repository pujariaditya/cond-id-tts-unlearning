"""Tortoise-TTS backend adapter.

Tortoise-TTS is an autoregressive codec language model (the architecture XTTS-v2
is derived from). Speaker identity is carried by the autoregressive conditioning
path of its ``UnifiedVoice`` sub-net:

* ``autoregressive.conditioning_encoder(mel) -> (b, 1024, s)`` -- a per-token
  conditioning latent (the analogue of XTTS ``gpt.get_style_emb``).
* ``autoregressive.get_conditioning(...)`` pools that latent (mean over the token
  axis, plus token 0) into the ``(b, 1024)`` ``auto_conditioning`` vector that the
  autoregressive sampler consumes.

Because identity is separable in that latent, the same model-independent unlearning
recipe (:func:`cond_id.methods.cond_id.applied_unlearning`) fine-tunes only the
conditioning encoder and leaves the rest of the model frozen. Including Tortoise
alongside XTTS-v2 and IndexTTS-1.5 is what demonstrates the method's independence
from any single architecture.

The host exposes the uniform conditioning interface the method trains against:

* :attr:`token_axis` = ``-1``               -- pool the ``(1, 1024, s)`` latent over
  the token axis.
* :meth:`cond_latents` (differentiable)     -- ``(1, 1024, s)`` latent for a reference
  waveform, so gradients flow to :meth:`cond_params`.
* :meth:`cond_params`                       -- the conditioning-encoder parameters
  (the only weights that are trained; the rest of the AR body stays frozen).

Checkpoint source: the Coqui TTS ``ModelManager`` downloads and caches
``tts_models/en/multi-dataset/tortoise-v2`` (autoregressive + diffusion + vocoder
weights) on first construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..hosts import TTSHost, _named_cond_params


@dataclass
class TortoiseSynthConfig:
    """Tortoise inference and conditioning settings (explicit; no environment reads).

    Attributes:
        num_autoregressive_samples: Number of autoregressive candidate clips the
            sampler draws before the diffusion decoder re-ranks them. Lower is
            faster; the paper's evaluation uses 16.
        diffusion_iterations: Diffusion decoder steps. Lower is faster.
        cond_free: Enable classifier-free diffusion guidance.
        temperature: Autoregressive sampling temperature.
        top_p: Nucleus-sampling cutoff for the autoregressive sampler.
        half: Run inference in fp16.
        pool: How :func:`cond_id.methods.cond_id` pools the conditioning latent
            for the identity vector -- ``"mean"`` (mean over the token axis, the
            default) or ``"tok0"`` (token 0, matching Tortoise's inference-time
            ``get_conditioning``, which consumes ``latent[:, :, 0]``). Empirically the
            mean transfers to token 0 because the attention encoder couples all tokens.
    """

    num_autoregressive_samples: int = 16
    diffusion_iterations: int = 20
    cond_free: bool = True
    temperature: float = 0.7
    top_p: float = 0.8
    half: bool = True
    pool: str = "mean"

    def inference_kwargs(self) -> dict:
        """Keyword arguments for ``Tortoise.inference`` derived from this config."""
        return dict(
            num_autoregressive_samples=int(self.num_autoregressive_samples),
            diffusion_iterations=int(self.diffusion_iterations),
            cond_free=bool(self.cond_free),
            temperature=float(self.temperature),
            top_p=float(self.top_p),
            verbose=False,
            half=bool(self.half),
        )


#: Tortoise conditioning sample rate (Hz). Reference audio is resampled to this
#: rate before mel extraction, regardless of the model's 24 kHz output rate.
COND_SR = 22050

#: Fixed conditioning-clip length in samples (Tortoise crops/pads references to this
#: before mel extraction). 132300 = 6 s at 22050 Hz.
COND_LENGTH = 132300


def _load_wav(path, sr_target: int = COND_SR) -> torch.Tensor:
    """Load a mono float32 waveform ``(1, T)`` resampled to ``sr_target`` Hz."""
    import soundfile as sf

    w, sr = sf.read(str(path), dtype="float32")
    if w.ndim > 1:
        w = w.mean(axis=1)
    t = torch.from_numpy(np.asarray(w, dtype=np.float32)).reshape(1, -1)
    if sr != sr_target:
        import torchaudio

        t = torchaudio.functional.resample(t, sr, sr_target)
    return t


class _CondMel:
    """Cached differentiable Tortoise conditioning-mel extractor (22050 Hz, 80 mel).

    Unlike Tortoise's inference-time ``format_conditioning`` (which runs under
    ``no_grad`` and crops the reference at a random offset), this extractor keeps the
    graph intact and crops deterministically, so the conditioning latent is
    differentiable with respect to the conditioning-encoder weights.
    """

    def __init__(self, model, dev):
        import os

        from TTS.tts.layers.tortoise.arch_utils import TorchMelSpectrogram

        # mel_norm_file=None avoids a network fetch; load the model's own mel_norms.
        self.tmel = TorchMelSpectrogram(sampling_rate=COND_SR, mel_norm_file=None).to(dev)
        norms = None
        mel_norm_path = getattr(model, "mel_norm_path", None)
        if mel_norm_path and os.path.exists(mel_norm_path):
            norms = torch.load(mel_norm_path, map_location="cpu")
        self.tmel.mel_norms = None if norms is None else norms.to(dev)
        self.dev = dev

    def mel(self, wav22k, cond_length: int = COND_LENGTH) -> torch.Tensor:
        """Conditioning mel ``(1, 80, s)`` from a 22050 Hz waveform (differentiable)."""
        w = torch.as_tensor(wav22k, dtype=torch.float32).reshape(1, -1).to(self.dev)
        gap = w.shape[-1] - cond_length
        if gap < 0:
            w = torch.nn.functional.pad(w, (0, -gap))
        elif gap > 0:
            w = w[:, :cond_length]  # deterministic crop
        mel = self.tmel(w.unsqueeze(0))  # (1, 1, T) -> (1, 80, s)
        if mel.dim() == 2:  # (80, s) -> add batch dim
            mel = mel.unsqueeze(0)
        return mel


def load_tortoise(dev: str = "cuda"):
    """Load Tortoise-TTS once and return ``(model, model_dir)``.

    The checkpoint (``tts_models/en/multi-dataset/tortoise-v2``) is downloaded and
    cached by the Coqui TTS ``ModelManager`` on first call.

    Args:
        dev: Torch device to move the model to.

    Returns:
        A tuple ``(model, model_dir)`` where ``model`` is a loaded ``Tortoise`` and
        ``model_dir`` is its cache directory.
    """
    import os

    import TTS
    from TTS.tts.configs.tortoise_config import TortoiseConfig
    from TTS.tts.models.tortoise import Tortoise
    from TTS.utils.manage import ModelManager

    mm = ModelManager(
        os.path.join(os.path.dirname(TTS.__file__), ".models.json"),
        progress_bar=False,
    )
    mdir, _, _ = mm.download_model("tts_models/en/multi-dataset/tortoise-v2")
    cfg = TortoiseConfig()
    cfg.load_json(os.path.join(mdir, "config.json"))
    cfg.model_dir = mdir
    model = Tortoise.init_from_config(cfg)
    model.load_checkpoint(cfg, checkpoint_dir=mdir, eval=True)
    model.to(dev)  # model.device is derived from parameters
    return model, mdir


class TortoiseHost(TTSHost):
    """Tortoise-TTS host (autoregressive codec language model).

    Implements the uniform speaker-conditioning interface
    (:attr:`token_axis`, :meth:`cond_latents`, :meth:`cond_params`) so the
    model-independent unlearning recipe runs unchanged, plus a cloning
    :meth:`tts` that synthesizes from an explicit reference clip.
    """

    #: The Tortoise conditioning latent is ``(1, 1024, s)``; pool over the token axis.
    token_axis = -1

    def __init__(self, config: TortoiseSynthConfig | None = None):
        """Load Tortoise-TTS and prepare the conditioning interface.

        Args:
            config: Inference/conditioning settings. Defaults to
                :class:`TortoiseSynthConfig`.
        """
        from ..config import DEVICE

        self.config = config or TortoiseSynthConfig()
        self._dev = DEVICE
        self._m, self._model_dir = load_tortoise(DEVICE)
        self._sr = 24000  # Tortoise inference output sample rate
        self._condmel = None  # lazily-built differentiable mel extractor

    @property
    def sr(self) -> int:
        return self._sr

    @property
    def model(self):
        """The autoregressive (weight-bearing) sub-net.

        Unlearning fine-tunes only ``model.autoregressive.conditioning_encoder``.
        """
        return self._m.autoregressive

    def _cm(self) -> _CondMel:
        if self._condmel is None:
            self._condmel = _CondMel(self._m, next(self.model.parameters()).device)
        return self._condmel

    def cond_latents(self, wav) -> torch.Tensor:
        """Differentiable conditioning latent ``(1, 1024, s)`` for a reference clip.

        Args:
            wav: A ``(1, T)`` float waveform at :attr:`sr` (24 kHz). It is resampled
                to the 22050 Hz conditioning rate, mel-extracted, and passed through
                the autoregressive conditioning encoder, so gradients flow to
                :meth:`cond_params`.

        Returns:
            The ``(1, 1024, s)`` conditioning latent.
        """
        import torchaudio

        src = wav.detach().cpu() if hasattr(wav, "detach") else wav
        w = torch.as_tensor(np.asarray(src, dtype=np.float32)).reshape(1, -1)
        w22 = torchaudio.functional.resample(w, self._sr, COND_SR)
        return self.model.conditioning_encoder(self._cm().mel(w22))

    def cond_params(self):
        """Trainable conditioning-encoder parameters (the rest of the AR body frozen).

        The unlearning context manager trains only these weights and snapshots and
        restores them exactly on exit.
        """
        params = []
        for p in self.model.conditioning_encoder.parameters():
            p.requires_grad_(True)
            params.append(p)
        return params

    def named_cond_params(self):
        """Named conditioning-encoder parameters (for patch save/load)."""
        return _named_cond_params([("conditioning_encoder", self.model.conditioning_encoder)])

    def _voice(self, ref_wav_path):
        """A ``[[wav_22050, wav_24000]]`` voice sample for ``get_conditioning_latents``.

        Loads the reference directly (Tortoise's ``clone_voice`` rejects ``.flac``).
        """
        import torchaudio

        w = _load_wav(ref_wav_path, COND_SR)
        w24 = torchaudio.functional.resample(w, COND_SR, self._sr)
        return [[w.clamp(-1, 1), w24.clamp(-1, 1)]]

    def _resolve_ref(self, speaker: str) -> str:
        """Resolve a reference-audio path for ``speaker`` via the dataset resolvers."""
        from ..datasets import ref_paths

        return str(ref_paths(speaker, 1)[0])

    def tts(self, text, speaker, ref_override=None) -> np.ndarray:
        """Synthesize ``text`` in the voice of ``speaker``.

        Builds conditioning latents from the *current* conditioning encoder (pristine
        or unlearned), so an identity redirect applied by the unlearning method drives
        the cloned speaker directly.

        Args:
            text: The sentence to synthesize.
            speaker: Speaker id used to resolve a reference clip when ``ref_override``
                is not supplied.
            ref_override: Optional path to a specific (e.g. held-out) reference clip
                to clone from, giving reproducible identity without depending on the
                dataset resolver's ordering.

        Returns:
            A float32 waveform at :attr:`sr` (24 kHz).
        """
        ref = str(ref_override) if ref_override is not None else self._resolve_ref(speaker)
        voice = self._voice(ref)
        auto_c, diff_c = self._m.get_conditioning_latents(voice)
        out = self._m.inference(
            text, conditioning_latents=(auto_c, diff_c), **self.config.inference_kwargs()
        )
        return np.asarray(out["wav"], dtype=np.float32).reshape(-1)
