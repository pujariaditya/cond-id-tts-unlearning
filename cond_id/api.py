"""Public unlearning API: erase speakers and get a persisted, edited model.

    import cond_id as ve

    edited = ve.unlearn(
        "xtts",
        forget={"alice": ["a1.wav", "a2.wav"]},   # or forget="forget_dir/"
        retain={"bob": ["b1.wav"]},                # or retain="retain_dir/"
        out="edited_xtts/",                        # writes a small adapter patch
    )
    wav = edited.synth("Hello, world.", reference="some_clip.wav")

Later, rebuild it onto your own base checkpoint (nothing but the patch is shared):

    m = ve.load_edited("edited_xtts/", base="xtts")
    wav = m.synth("...", reference="ref.wav")

The edit only touches the model's conditioning path, so the exported patch is a few
megabytes and never contains base weights. On IndexTTS the edit misdirects the GPT
conditioning path only; BigVGAN's own speaker encoder is a residual-leak path, so
erasure there is partial.

``unlearn`` mutates the host's conditioning weights **in place and permanently**
(there is no restore); the returned model and saved patch are that persisted edit.
The reversible, snapshot-and-restore variant used only by the reproducible
evaluation is :func:`cond_id.methods.cond_id.applied_unlearning`.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from collections.abc import Mapping
from pathlib import Path

from . import adapter as _adapter
from .hosts import TTSHost, _resolve_host, build_host
from .methods.cond_id import UnlearnConfig, apply_cond_id

_log = logging.getLogger(__name__)

_AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
_TAGS = ("xtts", "tortoise", "indextts")
_CKPT_ENV = {"xtts": "CD_XTTS_DIR", "indextts": "CD_INDEXTTS_DIR"}


def _refs_from(spec) -> dict[str, list[str]]:
    """Normalize a forget/retain spec into ``{label: [clip_path, ...]}``.

    ``spec`` is either a mapping ``{label: [paths]}`` or a directory whose immediate
    subdirectories are speaker labels containing audio clips (``<label>/*.wav``).
    """
    if spec is None:
        return {}
    if isinstance(spec, Mapping):

        def _clips(v):
            # A scalar path (str / PathLike) is one clip, not an iterable of chars.
            if isinstance(v, str | os.PathLike):
                return [str(v)]
            return [str(p) for p in v]

        return {str(k): _clips(v) for k, v in spec.items()}
    root = Path(spec)
    if not root.is_dir():
        raise ValueError(
            f"expected a mapping or a directory of <label>/*.wav folders; got {spec!r}"
        )
    out: dict[str, list[str]] = {}
    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            clips = sorted(str(p) for p in sub.iterdir() if p.suffix.lower() in _AUDIO_EXTS)
            if clips:
                out[sub.name] = clips
    if not out:
        raise ValueError(f"no <label>/*.wav speaker folders found under {root}")
    return out


def _load_unlearn_config(config) -> UnlearnConfig:
    """Coerce ``config`` (None / UnlearnConfig / dict / YAML path) into an UnlearnConfig."""
    if config is None:
        return UnlearnConfig()
    if isinstance(config, UnlearnConfig):
        return config
    if isinstance(config, Mapping):
        return UnlearnConfig(**config.get("unlearn", config))
    import yaml

    data = yaml.safe_load(Path(config).read_text())
    return UnlearnConfig(**data.get("unlearn", {}))


def _host_tag(host: TTSHost) -> str:
    """Map a host instance to its backbone tag (raises on an unrecognized host type)."""
    tags = {"XTTSHost": "xtts", "TortoiseHost": "tortoise", "IndexTTSHost": "indextts"}
    name = type(host).__name__
    if name not in tags:
        raise TypeError(
            f"cannot infer a backbone tag from host type {name!r}; pass model= as a "
            "tag string ('xtts' / 'tortoise' / 'indextts')"
        )
    return tags[name]


class EditedModel:
    """A speaker-unlearned model: synthesize with it, or save it as an adapter patch."""

    def __init__(self, host: TTSHost, meta: dict):
        self._host = host
        self.meta = dict(meta)

    @property
    def host(self) -> TTSHost:
        return self._host

    @property
    def sr(self):
        """Output sample rate in Hz."""
        return self._host.sr

    @property
    def forget_labels(self) -> list[str]:
        """The forget-speaker labels this edit erased."""
        return list(self.meta.get("forget_labels", []))

    def synth(self, text: str, reference: str):
        """Synthesize ``text`` cloning the voice in ``reference`` (an audio clip path)."""
        return self._host.synth(text, reference)

    def save(self, out_dir) -> Path:
        """Write this edit to ``out_dir`` as an adapter patch (weights + metadata)."""
        return _adapter.save_patch(self._host, out_dir, self.meta)


def _build_meta(tag: str, host: TTSHost, forget_labels, cfg: UnlearnConfig, seed: int) -> dict:
    """Assemble the JSON-serializable patch metadata."""
    # The actual base-checkpoint directory: XTTS/IndexTTS wrappers record ckpt_dir;
    # TortoiseHost keeps its Coqui ModelManager cache dir on _model_dir.
    wrapper = getattr(host, "_w", None)
    base_dir = getattr(wrapper, "ckpt_dir", None) or getattr(host, "_model_dir", None)
    return {
        "cond_id_version": _version(),
        "model": tag,
        "forget_labels": list(forget_labels),
        "seed": int(seed),
        "config": dataclasses.asdict(cfg),
        "base_checkpoint_dir": str(base_dir) if base_dir else None,
    }


def unlearn(model, forget, retain, *, config=None, seed: int = 0, out=None) -> EditedModel:
    """Unlearn ``forget`` speakers from ``model`` while preserving ``retain`` speakers.

    This edits the host's conditioning path **in place and permanently**: it mutates
    the passed-in model's weights and does not snapshot or restore them, so the same
    host object is erased for the rest of the process (that is what makes the returned
    :class:`EditedModel` -- and the saved adapter patch -- a persisted edit). If you
    instead want a *reversible* edit that restores the original weights on exit (used
    by the reproducible evaluation), use
    :func:`cond_id.methods.cond_id.applied_unlearning` as a context manager.

    Args:
        model: A backbone tag (``"xtts"`` / ``"tortoise"`` / ``"indextts"``) or a
            :class:`~cond_id.hosts.TTSHost`.
        forget: Speakers to erase -- ``{label: [clip_path, ...]}`` or a directory of
            ``<label>/*.wav`` folders.
        retain: Speakers to preserve -- same forms as ``forget``.
        config: cond-ID hyperparameters: an :class:`UnlearnConfig`, a dict, or a path
            to a config YAML with an ``unlearn:`` block. Defaults to model-agnostic.
        seed: Random seed for the edit.
        out: Optional directory to write the adapter patch to.

    Returns:
        An :class:`EditedModel` (the edit persists on the underlying host).
    """
    host = _resolve_host(model)
    tag = model if isinstance(model, str) else _host_tag(host)
    cfg = _load_unlearn_config(config)
    forget_refs = _refs_from(forget)
    retain_refs = _refs_from(retain)
    if not forget_refs:
        raise ValueError("no forget speakers supplied")
    if not retain_refs:
        raise ValueError("no retain speakers supplied (cond-ID pins retain identity)")

    edited_labels = apply_cond_id(host, forget_refs, retain_refs, seed=seed, config=cfg)
    if not edited_labels:
        raise RuntimeError(
            "cond-ID edit was a no-op: the host may lack a differentiable conditioning "
            "path, or the reference audio could not be loaded. Check the input clips."
        )
    # A silently un-erased forget speaker is worse than a crash for a privacy tool:
    # fail loudly if any requested forget speaker had no usable reference audio.
    dropped = [label for label in forget_refs if label not in set(edited_labels)]
    if dropped:
        raise RuntimeError(
            f"cond-ID could not erase {dropped}: no usable reference audio loaded for "
            "them. Check those speakers' clips (this is a hard failure rather than a "
            "silent partial edit)."
        )

    edited = EditedModel(host, _build_meta(tag, host, edited_labels, cfg, seed))
    if out is not None:
        edited.save(out)
        _log.info("wrote adapter patch to %s", out)
    return edited


def load_edited(patch_dir, base=None) -> EditedModel:
    """Rebuild an edited model: construct the base host and apply the saved patch.

    Args:
        patch_dir: Directory written by :func:`unlearn` / :meth:`EditedModel.save`.
        base: How to obtain the base model -- a :class:`TTSHost`, a backbone tag, a
            base-checkpoint directory, or ``None`` to use the tag recorded in the
            patch metadata (resolving the checkpoint via the usual ``CD_*`` env vars).

    Returns:
        An :class:`EditedModel` ready to :meth:`~EditedModel.synth`.
    """
    meta = _adapter.read_meta(patch_dir)
    tag = meta.get("model", "xtts")

    if isinstance(base, TTSHost):
        host = base
    elif base is None:
        host = build_host(tag)
    elif isinstance(base, str) and base in _TAGS:
        tag = base
        host = build_host(tag)
    elif isinstance(base, str) and Path(base).is_dir():
        # A base-checkpoint directory: point the tag's asset env var at it, then build.
        if tag not in _CKPT_ENV:
            raise ValueError(
                f"{tag!r} does not support a base-checkpoint directory (it resolves its "
                f"checkpoint from a cache); pass a TTSHost or a tag {_TAGS} instead."
            )
        # Confine the env override to just this construction so we don't leak a
        # process-global CD_*_DIR that would silently redirect later host builds.
        env_key = _CKPT_ENV[tag]
        prev = os.environ.get(env_key)
        os.environ[env_key] = str(base)
        try:
            host = build_host(tag)
        finally:
            if prev is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = prev
    else:
        raise ValueError(
            f"base must be a TTSHost, a tag {_TAGS}, a checkpoint directory, or None; got {base!r}"
        )

    loaded = _adapter.load_patch(host, patch_dir)
    return EditedModel(host, loaded or meta)


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("cond-id")
    except Exception:
        return "unknown"
