"""cond-ID: training-based speaker unlearning for zero-shot TTS.

Library use -- unlearn your own speakers and get an edited model:

* :func:`~cond_id.api.unlearn` -- erase forget speakers from a backbone, given
  their (and the retain speakers') audio, and return an :class:`~cond_id.api.EditedModel`.
* :func:`~cond_id.api.load_edited` -- rebuild an edited model from a saved patch.
* :class:`~cond_id.api.EditedModel` -- ``.synth(text, reference)`` / ``.save(dir)``.

Lower-level / evaluation API:

* :func:`~cond_id.methods.cond_id.apply_cond_id` -- apply the cond-ID edit in
  place (no restore); the basis of :func:`unlearn`.
* :func:`~cond_id.methods.cond_id.applied_unlearning` -- the restoring context
  manager the reproducible evaluation uses.
* :class:`~cond_id.methods.cond_id.UnlearnConfig` -- per-model hyperparameters.
* :func:`~cond_id.hosts.build_host` -- construct an XTTS-v2, Tortoise-TTS, or
  IndexTTS-1.5 host.
* :func:`~cond_id.config.set_determinism` -- pin all RNG for reproducible eval.
"""

from .api import EditedModel, load_edited, unlearn
from .config import set_determinism
from .hosts import TTSHost, build_host
from .methods.cond_id import UnlearnConfig, applied_unlearning, apply_cond_id
from .models.xtts import XTTSHost

__all__ = [
    "unlearn",
    "load_edited",
    "EditedModel",
    "apply_cond_id",
    "applied_unlearning",
    "UnlearnConfig",
    "set_determinism",
    "build_host",
    "TTSHost",
    "XTTSHost",
]
