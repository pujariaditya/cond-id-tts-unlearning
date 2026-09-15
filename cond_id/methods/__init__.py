"""Speaker-unlearning method.

:func:`~cond_id.methods.cond_id.applied_unlearning` -- the cond-ID
training-based unlearning context manager (fine-tunes the conditioning path,
restores on exit); :class:`~cond_id.methods.cond_id.UnlearnConfig` supplies
per-model hyperparameters.
"""

from .cond_id import UnlearnConfig, applied_unlearning

__all__ = ["applied_unlearning", "UnlearnConfig"]
