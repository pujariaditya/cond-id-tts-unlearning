"""XTTS-v2 backend adapter.

XTTS-v2 is an autoregressive codec language model: a GPT-2-style transformer
predicts discrete audio (DVAE) codes conditioned on text and a reference clip.
It is one of three backbones cond-ID runs on: the same conditioning-path
unlearning fine-tune drives this codec-LM, Tortoise-TTS and IndexTTS-1.5 without
modification.

The checkpoint is not bundled. Fetch it with ``scripts/download_models.py`` (or
set ``CD_XTTS_DIR`` to an existing XTTS-v2 directory containing ``config.json``,
the checkpoint, ``dvae.pth`` and ``mel_stats.pth``).
"""

import os

import torch

from ..config import PROJECT_ROOT
from ..hosts import TTSHost, _named_cond_params

#: XTTS conditions on the first ``_CLONE_SECONDS`` of the reference clip, resampled
#: to the mel-cloning rate ``_MEL_CLONING_SR`` (distinct from the 24 kHz output rate).
_MEL_CLONING_SR = 22050
_CLONE_SECONDS = 6


class XTTSWrapper:
    """Minimal XTTS-v2 synthesis surface (clones from a reference clip)."""

    #: Repository-relative default location for the XTTS-v2 checkpoint.
    DEFAULT_DIR = str(PROJECT_ROOT / "models" / "xtts_v2")

    def __init__(self, model_dir=None):
        """Load XTTS-v2 from ``model_dir`` (or ``CD_XTTS_DIR`` / the default).

        Args:
            model_dir: Path to the XTTS-v2 directory. Falls back to
                ``$CD_XTTS_DIR`` then :attr:`DEFAULT_DIR`.
        """
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        d = model_dir or os.environ.get("CD_XTTS_DIR", self.DEFAULT_DIR)
        self.ckpt_dir = d  # the actual directory the checkpoint was loaded from
        cfg = XttsConfig()
        cfg.load_json(os.path.join(d, "config.json"))
        self.model = Xtts.init_from_config(cfg)
        self.model.load_checkpoint(cfg, checkpoint_dir=d, eval=True)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.dev)
        self.config = cfg
        self.sr = 24000

    def gpt(self):
        """Return the GPT backbone (the module whose weights are edited)."""
        return self.model.gpt

    def _seed(self, seed):
        """Seed the RNGs the XTTS GPT decode samples from, for reproducible synthesis.

        XTTS decodes autoregressively with multinomial sampling (``temperature``); without a
        pinned seed the same input yields a different waveform each call, so speaker-similarity
        drop is non-reproducible. Re-seeding here makes a single synthesis call deterministic.
        """
        if seed is None:
            return
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    def tts(self, text, ref_wav_path, seed=None, temperature=None):
        """Synthesize ``text`` cloning the voice in ``ref_wav_path``.

        Args:
            seed: if given, re-seed before decoding so the output is reproducible.
            temperature: override the GPT sampling temperature (None = XTTS config default).
        Returns:
            float32 waveform at 24 kHz.
        """
        self._seed(seed)
        kw = {} if temperature is None else {"temperature": float(temperature)}
        out = self.model.synthesize(
            text, self.config, speaker_wav=ref_wav_path, language="en", **kw
        )
        return out["wav"]

    def tts_uncond(self, text, ref_wav_path, seed=None, temperature=0.7):
        """UNCONDITIONED generation (drop the speaker conditioning).

        Zeroes the speaker conditioning (gpt_cond_latent + speaker_embedding) so the
        GPT produces intelligible speech in a GENERIC voice with no speaker identity.
        ``ref_wav_path`` is used only to obtain conditioning-tensor shapes (its
        identity is zeroed out).

        Args:
            seed: if given, re-seed before decoding for reproducible synthesis.
            temperature: GPT sampling temperature (0.0 = greedy/deterministic).
        """
        import numpy as np
        import torch

        self._seed(seed)
        gpt_cond, spk_emb = self.model.get_conditioning_latents(audio_path=ref_wav_path)
        out = self.model.inference(
            text,
            "en",
            torch.zeros_like(gpt_cond),
            torch.zeros_like(spk_emb),
            temperature=float(temperature),
            enable_text_splitting=False,
        )
        return np.asarray(out["wav"], dtype=np.float32)


class XTTSHost(TTSHost):
    """XTTS-v2 host (autoregressive codec language model).

    Clones from a reference clip (no speaker-preset shortcut), so the model is
    scored on the same speakers as any other host.
    """

    #: XTTS cond latent has shape ``(1, 1024, 32)``; pool over the 32-token axis.
    token_axis = -1

    def __init__(self):
        import json
        from pathlib import Path

        from ..config import AUDIO_DIR, SPEAKER_MAP

        self._w = XTTSWrapper()
        self._smap = json.loads(Path(SPEAKER_MAP).read_text()) if Path(SPEAKER_MAP).exists() else {}
        self._audio = AUDIO_DIR

    @property
    def model(self):
        return self._w.gpt()

    def cond_latents(self, wav):
        """XTTS differentiable cond latent ``(1, 1024, 32)`` via ``gpt.get_style_emb``.

        Gradients flow to the conditioning encoder and perceiver while the GPT
        body stays frozen (preserving intelligibility). ``wav`` is a 24 kHz mono
        waveform, resampled here to the mel-cloning rate and cropped to the first
        ``_CLONE_SECONDS`` seconds.
        """
        import numpy as np
        import torchaudio
        from TTS.tts.models.xtts import wav_to_mel_cloning

        gpt = self._w.gpt()
        model = self._w.model
        dev = next(gpt.parameters()).device
        w = torch.as_tensor(np.asarray(wav, dtype=np.float32)).reshape(1, -1)
        audio = torchaudio.functional.resample(w, 24000, _MEL_CLONING_SR)[
            :, : _MEL_CLONING_SR * _CLONE_SECONDS
        ]
        mel = wav_to_mel_cloning(
            audio,
            mel_norms=model.mel_stats.cpu(),
            n_fft=2048,
            hop_length=256,
            win_length=1024,
            power=2,
            normalized=False,
        )
        return gpt.get_style_emb(mel.to(dev))

    def cond_params(self):
        """The conditioning encoder + perceiver parameters (the GPT body is frozen)."""
        gpt = self._w.gpt()
        params = []
        for mod in (gpt.conditioning_encoder, gpt.conditioning_perceiver):
            for p in mod.parameters():
                p.requires_grad_(True)
                params.append(p)
        return params

    def named_cond_params(self):
        """Named conditioning encoder + perceiver parameters (for patch save/load)."""
        gpt = self._w.gpt()
        return _named_cond_params(
            [
                ("conditioning_encoder", gpt.conditioning_encoder),
                ("conditioning_perceiver", gpt.conditioning_perceiver),
            ]
        )

    def _ref_for(self, speaker):
        """Return the VCTK reference-audio path for ``speaker``."""
        idx = self._smap[speaker][0]
        return str(self._audio / (str(idx) + ".flac"))

    def tts(self, text, speaker, ref_override=None):
        """Synthesize ``text`` in the voice of ``speaker``.

        When ``ref_override`` is given, clone from that clip path instead of the
        speaker's default reference (used to evaluate against held-out clips, and
        by :meth:`synth` for arbitrary references).
        """
        ref = str(ref_override) if ref_override is not None else self._ref_for(speaker)
        return self._w.tts(text, ref)

    def tts_uncond(self, text, dur_speaker):
        return self._w.tts_uncond(text, self._ref_for(dur_speaker))

    @property
    def sr(self):
        return self._w.sr
