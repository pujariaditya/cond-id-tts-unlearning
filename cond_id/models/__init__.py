"""Zero-shot TTS backend adapters.

Each backend implements the host interface in :mod:`cond_id.hosts` so the
unlearning method and evaluation harness drive any model the same way:

* :class:`cond_id.models.xtts.XTTSHost` -- XTTS-v2, an autoregressive codec LM.
* :class:`cond_id.models.tortoise.TortoiseHost` -- Tortoise-TTS.
* :class:`cond_id.models.indextts.IndexTTSHost` -- IndexTTS-1.5 (perceiver
  conditioning; runs in its own environment).

``XTTSHost`` is re-exported here for convenience; the Tortoise and IndexTTS hosts
are imported lazily (via :func:`cond_id.hosts.build_host`) so their optional
dependencies are only required when used.
"""

from .xtts import XTTSHost

__all__ = ["XTTSHost"]
