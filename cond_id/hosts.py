"""TTS host abstraction for model-independent speaker unlearning.

A *host* wraps a zero-shot TTS backend behind one uniform, differentiable
speaker-conditioning interface, so the same unlearning recipe runs unchanged on
architecturally different models:

* :class:`cond_id.models.xtts.XTTSHost`        -- XTTS-v2, an autoregressive
  codec language model.
* :class:`cond_id.models.tortoise.TortoiseHost` -- Tortoise-TTS, autoregressive.
* :class:`cond_id.models.indextts.IndexTTSHost` -- IndexTTS-1.5, a GPT /
  perceiver-conditioned model (runs in its own environment).

Construct one with :func:`build_host`.

The contract a host must satisfy is intentionally small:

* ``host.model``               -- the backbone ``nn.Module`` (an autoregressive
  transformer; frozen while only the conditioning path is trained).
* ``host.token_axis``          -- axis of the conditioning latent to pool for identity.
* ``host.cond_latents(wav)``   -- the differentiable speaker-conditioning latent.
* ``host.cond_params()``       -- the trainable parameters of the conditioning path.
* ``host.named_cond_params()`` -- the same parameters as an ordered ``{name: tensor}``
  mapping, so an edit can be saved/loaded as a patch (see :mod:`cond_id.adapter`).
* ``host.tts(text, spk)``      -- synthesize ``text`` in the voice of ``spk``.
* ``host.synth(text, ref)``    -- synthesize ``text`` cloning an arbitrary reference
  clip path (the package/API entry point; independent of any dataset).
* ``host.sr``                  -- output sample rate (Hz).
* ``host.tts_uncond(...)``     -- OPTIONAL: synthesize with the speaker identity
  dropped (implemented by :class:`~cond_id.models.xtts.XTTSHost` only; used by
  its relearn adversary).

Constructing a host loads the backend lazily, so importing this module is cheap
and free of side effects.
"""


def _named_cond_params(module_pairs):
    """Build an ordered ``{qualified_name: parameter}`` map from ``(prefix, module)``.

    Each parameter is named ``f"{prefix}.{param_name}"`` and has ``requires_grad``
    enabled. The ordering is stable for a fixed module structure, so a saved patch
    reloads onto the same parameters (see :mod:`cond_id.adapter`).
    """
    out = {}
    for prefix, mod in module_pairs:
        for name, p in mod.named_parameters():
            p.requires_grad_(True)
            out[f"{prefix}.{name}"] = p
    return out


class TTSHost:
    """Uniform interface over a zero-shot TTS backend.

    Subclasses load a specific model and implement :attr:`model`, :meth:`tts`,
    :attr:`sr`, and the speaker-conditioning interface (:attr:`token_axis`,
    :meth:`cond_latents`, :meth:`cond_params`, :meth:`named_cond_params`).
    :meth:`synth` and :meth:`tts_uncond` have defaults here.
    """

    #: Axis of the conditioning latent to average over to obtain the pooled
    #: per-channel speaker identity vector (model-specific; see subclasses).
    token_axis = -1

    @property
    def model(self):
        """The backbone ``nn.Module`` (an autoregressive transformer) of the host."""
        raise NotImplementedError

    def tts(self, text, speaker, ref_override=None):
        """Synthesize ``text`` in the voice of ``speaker``; return a waveform.

        Args:
            text: The text to synthesize.
            speaker: Speaker identifier resolved to a reference clip by the host
                (may be ``None`` when ``ref_override`` is supplied).
            ref_override: Optional path to a reference clip to clone from instead
                of the speaker's default reference. ``None`` uses the speaker's
                default reference.
        """
        raise NotImplementedError

    def synth(self, text, reference):
        """Synthesize ``text`` cloning the voice in an arbitrary reference clip.

        This is the dataset-independent entry point used by the public API and CLI
        (:func:`cond_id.unlearn`). It defaults to cloning ``reference`` directly
        via :meth:`tts`.

        Args:
            text: The text to synthesize.
            reference: Path to a reference audio clip to clone the voice from.

        Returns:
            A waveform (numpy array) at :attr:`sr`.
        """
        return self.tts(text, None, ref_override=reference)

    def tts_uncond(self, text, dur_speaker):
        """Synthesize ``text`` UNCONDITIONED (generic voice, no speaker identity).

        Optional: only :class:`~cond_id.models.xtts.XTTSHost` implements it (its
        relearn adversary uses it). ``dur_speaker`` supplies only the
        conditioning-tensor shapes needed to decode, not any identity.
        """
        raise NotImplementedError

    @property
    def sr(self):
        """Output sample rate in Hz."""
        raise NotImplementedError

    # --- unified speaker-conditioning interface (model-independent unlearning) ---
    # An unlearning method redirects the pooled speaker identity carried by this
    # differentiable conditioning latent. Writing the method against
    # ``host.cond_latents`` / ``host.cond_params`` lets one recipe run unchanged
    # across backbones -- the only per-model difference is the latent's shape,
    # absorbed by pooling over ``token_axis``.

    def cond_latents(self, wav):
        """Differentiable speaker-conditioning latent ``z`` for a reference waveform.

        This is the separable identity path of the backbone. The returned shape
        is model-specific; :attr:`token_axis` is the axis to average over to get
        the pooled per-channel identity vector. ``wav`` is a mono waveform at the
        host's expected reference sample rate.
        """
        raise NotImplementedError

    def cond_params(self):
        """Trainable parameters of the conditioning path (rest of the model frozen).

        An unlearning method fine-tunes only these parameters; the eval path
        snapshots and restores them so the host is left unchanged after the edit.
        """
        raise NotImplementedError

    def named_cond_params(self):
        """Ordered ``{qualified_name: parameter}`` for the conditioning path.

        The same tensors as :meth:`cond_params`, but named, so a persisted edit can
        be saved and reloaded as a patch (see :mod:`cond_id.adapter`). Build one
        with :func:`_named_cond_params`.
        """
        raise NotImplementedError


def build_host(host_tag: str = "xtts") -> TTSHost:
    """Construct a host by tag.

    Backends are imported lazily inside this function so that using one never
    depends on its optional sibling modules being importable.

    Args:
        host_tag: ``"xtts"`` (default), ``"tortoise"``, or ``"indextts"``.

    Returns:
        The corresponding :class:`TTSHost` instance.

    Raises:
        ValueError: If ``host_tag`` names an unknown backend.
    """
    tag = str(host_tag).lower()
    if tag == "xtts":
        from .models.xtts import XTTSHost

        return XTTSHost()
    if tag == "tortoise":
        from .models.tortoise import TortoiseHost

        return TortoiseHost()
    if tag == "indextts":
        from .models.indextts import IndexTTSHost

        return IndexTTSHost()
    raise ValueError(f"unknown host tag {host_tag!r}; expected 'xtts', 'tortoise', or 'indextts'.")


def _resolve_host(host) -> TTSHost:
    """Resolve a host object or tag to a :class:`TTSHost`.

    Args:
        host: A :class:`TTSHost`, or a string tag (``"xtts"`` / ``"tortoise"`` /
            ``"indextts"``).

    Returns:
        A :class:`TTSHost` instance.

    Raises:
        TypeError: If ``host`` is neither a host nor a known tag string.
    """
    if isinstance(host, TTSHost):
        return host
    if isinstance(host, str):
        return build_host(host)
    raise TypeError(
        "expected a TTSHost or a tag string ('xtts' / 'tortoise' / 'indextts') "
        f"as the first argument; got {type(host).__name__}. "
        "Build one with cond_id.build_host(...)."
    )
