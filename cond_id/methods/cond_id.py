"""Conditioning-Space Identity Redirection (cond-ID) speaker unlearning.

One method, three backbones. The edit is written once against a small host
conditioning interface (``host.cond_latents`` / ``host.cond_params`` /
``host.token_axis``), so the same code fine-tunes XTTS-v2, Tortoise-TTS, and
IndexTTS-1.5 -- the only per-model difference is the conditioning-latent shape,
absorbed by pooling over ``host.token_axis``.

Mechanism. Decompose the differentiable speaker-conditioning latent
``z = host.cond_latents(wav)`` into a pooled identity ``v = mean_axis(z)`` and a
per-token structure residual ``r = z - v``. Redirect each forget speaker's pooled
identity onto its own dispersed orthonormal anchor -- a cosine redirect toward
``c_dir - alpha * f_dir + gamma * anchor`` (``c_dir`` = retain centroid direction,
``f_dir`` = the speaker's own identity direction) -- while keeping the residual
close to frozen (intelligibility) and pinning retain latents to the frozen model.
Only ``host.cond_params()`` (the conditioning encoder) is trained; the rest of the
model is frozen.

Two entry points:

* :func:`apply_cond_id` applies the edit IN PLACE and does NOT restore -- the basis
  for a persisted edited model (see :func:`cond_id.unlearn`). It takes explicit
  reference clips, so it is independent of any dataset.
* :func:`applied_unlearning` is a context manager that snapshots the conditioning
  weights, applies the edit for the duration of the ``with`` block, then restores
  them exactly on exit. The reproducible evaluation uses it (it resolves VCTK
  speaker ids to their train-partition reference clips).

Hyperparameters are supplied per model via :class:`UnlearnConfig` (see ``configs/``).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

_log = logging.getLogger(__name__)


@dataclass
class UnlearnConfig:
    """cond-ID hyperparameters (per-model; loaded from ``configs/<model>.yaml``)."""

    alpha: float = 1.0  # strength of self-identity subtraction in the target
    gamma: float = 0.35  # strength of the dispersed anchor in the target
    redirect_w: float = 3.0  # weight of the cosine redirect loss
    structure_w: float = 0.5  # weight of the per-token residual preservation loss
    retain_w: float = 0.75  # weight of the retain-latent preservation loss
    k: int = 6  # reference clips per speaker
    steps: int = 25  # optimizer steps
    lr: float = 5e-4  # Adam learning rate


def _as_list(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def _load_wav(path, sr_target: int = 24000):
    """Load a wav/flac clip as mono float32 ``(1, T)`` resampled to ``sr_target``."""
    import soundfile as sf

    w, sr = sf.read(str(path), dtype="float32")
    if w.ndim > 1:
        w = w.mean(axis=1)
    t = torch.from_numpy(np.asarray(w, dtype=np.float32)).reshape(1, -1)
    if sr != sr_target:
        import torchaudio

        t = torchaudio.functional.resample(t, sr, sr_target)
    return t


def _refs_to_batches(host, refs: Mapping[str, Sequence], k: int) -> dict[str, list]:
    """Load ``{label: [wav_tensor, ...]}`` (<= ``k`` clips each) from reference paths.

    ``refs`` maps a speaker label to a sequence of reference-clip paths. Clips that
    fail to load are skipped; a label whose clips all fail is dropped with a warning
    (an empty forget label means that speaker would be silently un-erased).
    """
    sr = getattr(host, "sr", 24000)
    out: dict[str, list] = {}
    for label, paths in refs.items():
        wavs = []
        for p in list(paths)[: max(1, int(k))]:
            try:
                wavs.append(_load_wav(p, sr))
            except Exception:
                _log.warning("cond-ID: could not load reference clip %s for %r", p, label)
        if wavs:
            out[label] = wavs
        else:
            _log.warning("cond-ID: no usable reference clips for %r; skipping it", label)
    return out


def _pooled_identity(z, axis):
    """Pooled per-channel identity vector: mean of ``z`` over the token/time axis."""
    return z.mean(dim=axis)


def _dispersed_anchors(dim: int, n: int, device, seed: int):
    """``n`` mutually-orthonormal unit anchors (QR of a random matrix), one per forget
    speaker, so the erased voices stay distinct (no collapse onto a single decoy)."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    a = torch.randn(dim, max(n, 1), generator=g).to(device)
    q, _ = torch.linalg.qr(a)
    return q[:, :n].T.contiguous()  # (n, dim), unit orthonormal rows


def _snapshot(params):
    return [(p, p.detach().clone()) for p in params]


def _restore(snap):
    with torch.no_grad():
        for p, v in snap:
            p.copy_(v)


def apply_cond_id(
    host,
    forget_refs: Mapping[str, Sequence],
    retain_refs: Mapping[str, Sequence],
    *,
    seed: int = 0,
    config: UnlearnConfig | None = None,
) -> list[str]:
    """Apply the cond-ID edit to ``host`` IN PLACE (no snapshot / no restore).

    Fine-tunes only ``host.cond_params()`` (the conditioning path); the rest of the
    model is frozen. Unlike :func:`applied_unlearning`, the edit **persists** -- this
    is what :func:`cond_id.unlearn` uses to produce an edited model. ``forget_refs``
    and ``retain_refs`` map a speaker label to a list of reference-clip paths (the
    method reads only the audio; no transcripts are needed).

    Args:
        host: A :class:`~cond_id.hosts.TTSHost`.
        forget_refs: ``{label: [clip_path, ...]}`` for the speakers to erase.
        retain_refs: ``{label: [clip_path, ...]}`` for the speakers to preserve.
        seed: Seed for the dispersed anchors and the optimizer's clip sampling.
        config: Hyperparameters (defaults to model-agnostic :class:`UnlearnConfig`).

    Returns:
        The list of forget labels actually edited (empty if the edit was a no-op,
        e.g. the host has no differentiable conditioning path or no usable clips).
    """
    cfg = config or UnlearnConfig()
    axis = int(getattr(host, "token_axis", -1))
    try:
        params = host.cond_params()
    except NotImplementedError:
        return []
    if not params:
        return []

    # Freeze the whole model; train only the conditioning path.
    model = getattr(host, "model", None)
    if model is not None:
        for p in model.parameters():
            p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)

    fb = _refs_to_batches(host, forget_refs, cfg.k)
    forget = list(fb.keys())
    rb_by_label = _refs_to_batches(host, retain_refs, cfg.k)
    rb = [w for wavs in rb_by_label.values() for w in wavs]
    if not forget or not rb:
        return []

    dev = params[0].device
    with torch.no_grad():
        f_resid0, proto = {}, {}
        for sp in forget:
            zs = [host.cond_latents(w).detach() for w in fb[sp]]
            f_resid0[sp] = [z - z.mean(dim=axis, keepdim=True) for z in zs]
            proto[sp] = torch.stack([_pooled_identity(z, axis) for z in zs], 0).mean(0)
        r0 = [host.cond_latents(w).detach() for w in rb]
        r_centroid = torch.stack([_pooled_identity(z, axis) for z in r0], 0).mean(0)
        id_norm = torch.stack([proto[sp].norm() for sp in forget]).mean()

    c_dir = r_centroid / (r_centroid.norm() + 1e-8)
    dim = c_dir.shape[-1]
    anchors = _dispersed_anchors(dim, len(forget), dev, seed)
    targets = {}
    for i, sp in enumerate(forget):
        f_dir = proto[sp] / (proto[sp].norm() + 1e-8)
        t = c_dir - cfg.alpha * f_dir + cfg.gamma * anchors[i]
        targets[sp] = t / (t.norm() + 1e-8) * id_norm

    opt = torch.optim.Adam(params, lr=cfg.lr)
    rng = np.random.default_rng(seed)
    for _ in range(cfg.steps):
        opt.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=dev)
        for sp in forget:
            fi = int(rng.integers(len(fb[sp])))
            z = host.cond_latents(fb[sp][fi])
            v = _pooled_identity(z, axis).flatten()
            tgt = targets[sp].flatten()
            cos = torch.nn.functional.cosine_similarity(v, tgt, dim=0)
            r = z - z.mean(dim=axis, keepdim=True)
            r_frozen = f_resid0[sp][fi] if fi < len(f_resid0[sp]) else f_resid0[sp][0]
            loss = (
                loss + cfg.redirect_w * (1.0 - cos) + cfg.structure_w * (r - r_frozen).pow(2).mean()
            )
        ri = int(rng.integers(len(rb)))
        zr = host.cond_latents(rb[ri])
        loss = loss + cfg.retain_w * (zr - r0[ri]).pow(2).mean()
        if torch.isfinite(loss):
            loss.backward()
            opt.step()
    return forget


def _vctk_refs(speakers: Sequence[str], k: int) -> dict[str, list]:
    """Resolve VCTK speaker ids to their train-partition reference-clip paths."""
    from cond_id.datasets import ref_paths

    refs: dict[str, list] = {}
    for sp in speakers:
        try:
            refs[sp] = ref_paths(sp, max(1, int(k)))
        except Exception:
            refs[sp] = []
    return refs


@contextmanager
def applied_unlearning(
    host, forget_set, retain_set, *, seed: int = 0, config: UnlearnConfig | None = None
):
    """Erase ``forget_set`` identities on ``host`` for the duration of the ``with`` block.

    Snapshots ``host.cond_params()``, applies the cond-ID edit
    (:func:`apply_cond_id`) resolving the VCTK speaker ids to their train-partition
    reference clips, then restores the weights on exit: the trained values are
    restored exactly (bitwise), and the parameters are left with
    ``requires_grad=False``. This is the reproducible-evaluation entry point; use
    :func:`cond_id.unlearn` to produce a persisted edited model instead.

    ``config`` supplies the per-model hyperparameters (defaults to model-agnostic
    values).
    """
    cfg = config or UnlearnConfig()
    forget = _as_list(forget_set)
    retain = _as_list(retain_set)
    if not forget:
        yield
        return

    try:
        params = host.cond_params()
    except NotImplementedError:
        yield
        return
    if not params:
        yield
        return

    model = getattr(host, "model", None)
    snap = _snapshot(params)
    try:
        apply_cond_id(
            host, _vctk_refs(forget, cfg.k), _vctk_refs(retain, cfg.k), seed=seed, config=cfg
        )
        yield
    finally:
        _restore(snap)
        if model is not None:
            for p in model.parameters():
                p.requires_grad_(False)
