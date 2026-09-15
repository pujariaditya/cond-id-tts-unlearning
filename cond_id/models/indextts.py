"""IndexTTS-1.5 backend adapter.

IndexTTS-1.5 is an autoregressive TTS model in the ``conformer_perceiver``
conditioning regime. Speaker identity flows through two distinct paths:

* A GPT-style ``UnifiedVoice`` predicts discrete mel codes from text and a
  differentiable speaker-conditioning latent. That latent is produced by a
  ConformerEncoder (``conditioning_encoder``) followed by a PerceiverResampler
  (``perceiver_encoder``) that compresses a reference log-mel into a fixed
  ``(1, 32, 1024)`` tensor -- 32 perceiver tokens of width 1024. This is the
  direct analog of the XTTS ``get_style_emb`` path and is what the unlearning
  method edits.
* A BigVGAN vocoder turns the GPT latent into a waveform. BigVGAN carries its
  own ECAPA speaker encoder that re-derives an embedding from the *raw*
  reference mel, so it forms a second, independent identity path. Evaluation
  therefore scores erasure on the FINAL synthesized audio, which exposes any
  identity that leaks through the vocoder rather than the GPT conditioning.

Including IndexTTS-1.5 alongside XTTS-v2 and Tortoise-TTS is what lets cond-ID
demonstrate architecture independence: the same recipe runs against an
autoregressive codec-LM and this conformer/perceiver conditioner without
modification. The only per-model difference is the
conditioning-latent shape, absorbed by pooling over :attr:`IndexTTSHost.token_axis`
(axis ``1``, the 32 perceiver tokens).

The checkpoint is not bundled. Fetch the IndexTTS-1.5 release (``config.yaml``,
``gpt.pth``, ``bigvgan_generator.pth``, ``dvae.pth``, ``bpe.model``) into a
checkpoints directory and point ``model_dir`` (or ``CD_INDEXTTS_DIR``) at it.
Because the IndexTTS-1.5 stack pins its own dependency versions, this adapter is
intended to be imported inside an isolated virtual environment that has the
``indextts`` library installed; see the package docs for the environment recipe.
"""

import os

import numpy as np
import torch

from ..config import DEVICE, PROJECT_ROOT
from ..hosts import TTSHost, _named_cond_params


class IndexTTSWrapper:
    """Minimal IndexTTS-1.5 synthesis surface (clones from a reference clip).

    Loads the GPT ``UnifiedVoice``, the BigVGAN vocoder and the BPE tokenizer,
    and exposes the differentiable conditioning path used by the unlearning
    method (``get_conditioning`` -> ``(1, 32, 1024)``) alongside a ``clone``
    entry point that synthesizes text in a reference speaker's voice.
    """

    #: Repository-relative default location for the IndexTTS-1.5 checkpoint
    #: (mirrors XTTS's ``DEFAULT_DIR``; override with ``CD_INDEXTTS_DIR``).
    DEFAULT_DIR = str(PROJECT_ROOT / "models" / "index_tts")

    def __init__(self, model_dir=None, device=None):
        """Load IndexTTS-1.5 from a checkpoints directory.

        Args:
            model_dir: Directory holding ``config.yaml``, ``gpt.pth``,
                ``bigvgan_generator.pth``, ``dvae.pth`` and ``bpe.model``.
                Falls back to ``$CD_INDEXTTS_DIR``.
            device: Torch device string (defaults to the package device).
        """
        from indextts.BigVGAN.models import BigVGAN as Generator
        from indextts.gpt.model import UnifiedVoice
        from indextts.utils.checkpoint import load_checkpoint
        from indextts.utils.feature_extractors import MelSpectrogramFeatures
        from indextts.utils.front import TextTokenizer
        from omegaconf import OmegaConf

        ckpt = model_dir or os.environ.get("CD_INDEXTTS_DIR", self.DEFAULT_DIR)
        if not os.path.exists(os.path.join(ckpt, "config.yaml")):
            raise ValueError(
                f"IndexTTS-1.5 checkpoint not found at {ckpt!r}; pass model_dir or set "
                "CD_INDEXTTS_DIR to a directory containing config.yaml, gpt.pth, "
                "bigvgan_generator.pth, dvae.pth and bpe.model."
            )
        self.ckpt_dir = ckpt
        self.device = device or DEVICE
        self.cfg = OmegaConf.load(os.path.join(ckpt, "config.yaml"))
        # fp32 throughout for gradient stability during the conditioning-path edit.
        self.stop_mel_token = self.cfg.gpt.stop_mel_token
        self.mel_fn = MelSpectrogramFeatures().to(self.device)

        gpt_ckpt = self.cfg.get("gpt_checkpoint", "gpt.pth")
        bigvgan_ckpt = self.cfg.get("bigvgan_checkpoint", "bigvgan_generator.pth")
        self.dvae_ckpt = self.cfg.get("dvae_checkpoint", "dvae.pth")

        self.gpt = UnifiedVoice(**self.cfg.gpt)
        load_checkpoint(self.gpt, os.path.join(ckpt, gpt_ckpt))
        self.gpt = self.gpt.to(self.device).eval()
        self.gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=False, half=False)

        self.bigvgan = Generator(self.cfg.bigvgan, use_cuda_kernel=False)
        vd = torch.load(os.path.join(ckpt, bigvgan_ckpt), map_location="cpu")
        self.bigvgan.load_state_dict(vd["generator"])
        self.bigvgan = self.bigvgan.to(self.device)
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()

        # config's bpe_model path is checkpoints-relative; resolve by basename.
        bpe = os.path.join(ckpt, os.path.basename(self.cfg.dataset["bpe_model"]))
        self.tokenizer = TextTokenizer(bpe, normalizer=None)  # raw English tokens
        self.sr = 24000

    def cond_mel(self, audio24k):
        """Return the ``(1, 100, frames)`` log-mel of a mono 24 kHz waveform.

        Args:
            audio24k: Mono waveform tensor at 24 kHz, shape ``(T,)`` or ``(1, T)``.
        """
        a = audio24k.to(self.device).float()
        if a.dim() == 1:
            a = a.unsqueeze(0)
        return self.mel_fn(a)

    def cond_latents(self, cond_mel):
        """Differentiable speaker-conditioning latent ``(1, 32, 1024)``.

        Runs the ConformerEncoder + PerceiverResampler over ``cond_mel`` WITHOUT
        ``no_grad``, so gradients flow into the conditioning encoders while the
        rest of the GPT stays frozen. This is the identity path the unlearning
        method edits.

        Args:
            cond_mel: A ``(1, 100, frames)`` log-mel from :meth:`cond_mel`.
        """
        cml = torch.tensor([cond_mel.shape[-1]], device=self.device)
        return self.gpt.get_conditioning(cond_mel, cml)

    def cond_params(self, scope="both"):
        """Return the trainable conditioning-path parameters.

        Args:
            scope: ``"both"`` trains the ConformerEncoder and the
                PerceiverResampler (the XTTS-exact analog); ``"perceiver"`` trains
                only the PerceiverResampler -- the lighter, less-entangled head
                that avoids perturbing the deep conformer that also shapes the
                codes the GPT decodes.

        Returns:
            A list of parameters with ``requires_grad`` enabled; every other GPT
            parameter is left untouched.
        """
        mods = (
            (self.gpt.perceiver_encoder,)
            if scope == "perceiver"
            else (self.gpt.conditioning_encoder, self.gpt.perceiver_encoder)
        )
        params = []
        for mod in mods:
            for p in mod.parameters():
                p.requires_grad_(True)
                params.append(p)
        return params

    @torch.no_grad()
    def clone(self, text, cond_mel, max_mel_tokens=600):
        """Synthesize ``text`` in the voice described by ``cond_mel``.

        The GPT samples mel codes conditioned on the reference, re-runs the GPT to
        obtain the latent, and BigVGAN vocodes it (reading the raw ``cond_mel`` for
        its own speaker embedding).

        Args:
            text: The sentence to synthesize.
            cond_mel: A ``(1, 100, frames)`` reference log-mel.
            max_mel_tokens: Cap on the number of generated mel tokens.

        Returns:
            A float32 waveform at 24 kHz.
        """
        toks = self.tokenizer.encode(text.strip())
        tt = torch.tensor(toks, dtype=torch.int32, device=self.device).unsqueeze(0)
        cml = torch.tensor([cond_mel.shape[-1]], device=self.device)
        codes = self.gpt.inference_speech(
            cond_mel,
            tt,
            cond_mel_lengths=cml,
            do_sample=True,
            top_p=0.8,
            top_k=30,
            temperature=1.0,
            num_return_sequences=1,
            length_penalty=0.0,
            num_beams=3,
            repetition_penalty=10.0,
            max_generate_length=max_mel_tokens,
        )
        code_lens = torch.tensor([codes.shape[-1]], device=self.device, dtype=codes.dtype)
        latent = self.gpt(
            cond_mel,
            tt,
            torch.tensor([tt.shape[-1]], device=self.device),
            codes,
            code_lens * self.gpt.mel_length_compression,
            cond_mel_lengths=cml,
            return_latent=True,
            clip_inputs=False,
        )
        wav, _ = self.bigvgan(latent, cond_mel.transpose(1, 2))
        return wav.squeeze(1).squeeze(0).float().cpu().numpy()


def _load_wav_24k(path):
    """Load ``path`` as a mono float32 ``(1, T)`` waveform resampled to 24 kHz."""
    import soundfile as sf

    w, sr = sf.read(str(path), dtype="float32")
    if w.ndim > 1:
        w = w.mean(axis=1)
    t = torch.from_numpy(np.asarray(w, dtype=np.float32)).reshape(1, -1)
    if sr != 24000:
        import torchaudio

        t = torchaudio.functional.resample(t, sr, 24000)
    return t


class IndexTTSHost(TTSHost):
    """IndexTTS-1.5 host (conformer/perceiver conditioning + BigVGAN vocoder).

    Implements the model-independent conditioning interface so the same
    unlearning recipe that runs on XTTS-v2 and Tortoise-TTS also runs here without
    modification:

    * :attr:`model`          -- the GPT ``UnifiedVoice`` backbone (frozen /
      snapshotted / restored by the method).
    * :attr:`token_axis`     -- ``1``; the pooled identity is the mean over the 32
      perceiver tokens of the ``(1, 32, 1024)`` conditioning latent.
    * :meth:`cond_latents`   -- differentiable latent for a reference waveform.
    * :meth:`cond_params`    -- the conditioning-encoder parameters to train.
    * :meth:`tts`            -- clone a speaker from its dataset reference clip.

    Clones VCTK references so it is scored on the same speakers as the other
    hosts (no speaker-preset shortcut).
    """

    #: Pool the conditioning latent over the 32 perceiver tokens (axis 1).
    token_axis = 1

    def __init__(self, model_dir=None, cond_scope="both", max_mel_tokens=600):
        """Build the host.

        Args:
            model_dir: IndexTTS-1.5 checkpoints directory (see
                :class:`IndexTTSWrapper`).
            cond_scope: Conditioning-path training scope, ``"both"`` (default) or
                ``"perceiver"`` (see :meth:`IndexTTSWrapper.cond_params`).
            max_mel_tokens: Cap on generated mel tokens per synthesis.
        """
        self._w = IndexTTSWrapper(model_dir=model_dir)
        self._cond_scope = cond_scope
        self._max_mel_tokens = int(max_mel_tokens)
        self._cond_mel_cache = {}

    @property
    def model(self):
        """The GPT ``UnifiedVoice`` backbone (the module whose weights are edited)."""
        return self._w.gpt

    def cond_latents(self, wav):
        """Differentiable conditioning latent ``(1, 32, 1024)`` for a waveform.

        Args:
            wav: A mono waveform tensor at :attr:`sr` (24 kHz).
        """
        return self._w.cond_latents(self._w.cond_mel(wav))

    def cond_params(self):
        """Return the trainable conditioning-encoder parameters (rest frozen)."""
        return self._w.cond_params(scope=self._cond_scope)

    def named_cond_params(self):
        """Named conditioning-path parameters (for patch save/load), honoring ``cond_scope``."""
        gpt = self._w.gpt
        if self._cond_scope == "perceiver":
            pairs = [("perceiver_encoder", gpt.perceiver_encoder)]
        else:
            pairs = [
                ("conditioning_encoder", gpt.conditioning_encoder),
                ("perceiver_encoder", gpt.perceiver_encoder),
            ]
        return _named_cond_params(pairs)

    def _cond_mel_for(self, ref_path):
        """Return the cached reference log-mel for ``ref_path``."""
        if ref_path not in self._cond_mel_cache:
            self._cond_mel_cache[ref_path] = self._w.cond_mel(_load_wav_24k(ref_path))
        return self._cond_mel_cache[ref_path]

    def tts(self, text, speaker, ref_override=None):
        """Synthesize ``text`` cloning ``speaker``.

        ``ref_override`` clones from a specific reference clip path instead of
        resolving one from the speaker id -- the reproducible evaluation path
        supplies held-out reference clips this way.

        Returns:
            A float32 waveform at 24 kHz.
        """
        if ref_override is not None:
            ref = str(ref_override)
        else:
            from ..datasets import ref_paths

            ref = ref_paths(speaker, 1)[0]
        return self._w.clone(text, self._cond_mel_for(ref), max_mel_tokens=self._max_mel_tokens)

    @property
    def sr(self):
        """Output sample rate in Hz (24000)."""
        return self._w.sr
