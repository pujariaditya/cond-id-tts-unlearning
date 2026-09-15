"""Save / load a cond-ID edit as a lightweight adapter patch.

A patch captures only the edited conditioning-path weights (a few MB) plus
metadata, written to a directory:

    <dir>/cond_id_patch.pt   # {qualified_name: tensor} from host.named_cond_params()
    <dir>/unlearn_meta.json      # model tag, forget labels, config, seed, base info

Applying a patch loads those tensors back onto a freshly-built base host, so an
edited model is reproduced **without redistributing any base checkpoint** (the base
weights -- e.g. XTTS-v2 under the non-commercial CPML -- never leave the user's own
copy). See :mod:`cond_id.api`.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

#: File names inside a patch directory.
PATCH_FILE = "cond_id_patch.pt"
META_FILE = "unlearn_meta.json"


def save_patch(host, out_dir, meta: dict) -> Path:
    """Write ``host.named_cond_params()`` and ``meta`` to ``out_dir`` as a patch.

    Args:
        host: An edited :class:`~cond_id.hosts.TTSHost` whose conditioning path
            carries the cond-ID edit.
        out_dir: Destination directory (created if needed).
        meta: JSON-serializable metadata (see :func:`cond_id.api.unlearn`).

    Returns:
        The output directory path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = {name: p.detach().cpu().clone() for name, p in host.named_cond_params().items()}
    torch.save(state, out / PATCH_FILE)
    (out / META_FILE).write_text(json.dumps(meta, indent=2, default=str))
    return out


def read_meta(patch_dir) -> dict:
    """Return the patch metadata dict (``{}`` if no metadata file is present)."""
    p = Path(patch_dir) / META_FILE
    return json.loads(p.read_text()) if p.exists() else {}


def load_patch(host, patch_dir) -> dict:
    """Load the patch in ``patch_dir`` onto ``host`` in place; return its metadata.

    Args:
        host: A freshly-built base host to apply the edit onto (its conditioning
            params must match the ones the patch was saved from).
        patch_dir: Directory written by :func:`save_patch`.

    Returns:
        The patch metadata dict.

    Raises:
        ValueError: If the patch's parameter names do not match the host's
            conditioning parameters (a base-model or conditioning-scope mismatch).
    """
    d = Path(patch_dir)
    state = torch.load(d / PATCH_FILE, map_location="cpu")
    named = host.named_cond_params()
    missing = set(named) - set(state)
    unexpected = set(state) - set(named)
    mis_shape = [n for n in named if n in state and tuple(named[n].shape) != tuple(state[n].shape)]
    if missing or unexpected or mis_shape:
        raise ValueError(
            "patch does not match this host's conditioning parameters "
            f"({len(missing)} missing, {len(unexpected)} unexpected, "
            f"{len(mis_shape)} shape-mismatched). The patch must be applied to the "
            "same base model (and conditioning scope) it was built from."
        )
    with torch.no_grad():
        for name, p in named.items():
            p.copy_(state[name].to(device=p.device, dtype=p.dtype))
    return read_meta(patch_dir)
