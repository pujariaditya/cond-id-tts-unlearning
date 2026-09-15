"""Clone synthesis for the held-out-clone unlearning protocol (baseline + edited).

The unlearning protocol is evaluated on *clones*: short sentences synthesized in a
target speaker's voice from a held-out reference clip the method never trains on.
Two clone sets are produced per speaker split:

* **baseline** clones ``cB`` -- synthesized on the pristine model, before any
  unlearning. These fix the "before" identity of every speaker.
* **edited** clones ``cE`` -- synthesized while :func:`~cond_id.methods.cond_id.applied_unlearning` is
  active, i.e. on the unlearned conditioning path. For a *forget* speaker the
  identity should have collapsed toward the erasure floor; for a *retain* speaker
  it should be unchanged.

A third clone set ``cRL`` (relearn clones, produced by a white-box re-cloning
adversary) is left empty here and filled in by the relearn stage; scoring then
measures how much of the erased identity a re-clone attack can recover.

Both clone sets are synthesized from the *same* held-out references and the *same*
sentence bank, positionally aligned, so per-sentence WavLM/WER deltas are directly
comparable. Every clone is written to disk sentence-major, keyed by speaker.

Data contract
-------------
:func:`build_splits` returns a list of *splits*. Each split is a plain ``dict``::

    {
      "forget":        [speaker, ...],        # identities to erase
      "retain_seen":   [speaker, ...],        # retain, provided as unlearning anchors
      "retain_unseen": [speaker, ...],        # retain, held out from unlearning
      "impostor":      [wav_path, ...],       # different-speaker clips (erasure floor)
      "centroid_refs": {speaker: [wav_path]}, # held-out clips for the identity centroid
      "heldout_refs":  {speaker: [wav_path]}, # held-out clips used as cloning references
      "sentences":     [text, ...],           # positionally aligned with cB/cE lists
      "cB":  {speaker: [wav_path, ...]},      # baseline clones   (synth_baseline)
      "cE":  {speaker: [wav_path, ...]},      # edited clones     (synth_edited_clones)
      "cRL": {speaker: [wav_path, ...]},      # relearn clones    (relearn stage)
    }

``cB``/``cE``/``cRL`` are sentence-major and equal-length per speaker (one wav per
sentence). The score stage consumes ``cB``/``cE``/``cRL`` together with
``centroid_refs``/``heldout_refs``/``impostor``/``sentences``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from cond_id.datasets import held_clips
from cond_id.methods.cond_id import UnlearnConfig


@dataclass
class CloneConfig:
    """Where and how clone wavs are written.

    Attributes:
        out_dir: Directory the synthesized clone wavs are written into (created if
            absent).
        unlearn: Hyperparameters passed to :func:`~cond_id.methods.cond_id.applied_unlearning` when
            synthesizing the edited clones. ``None`` uses the model-agnostic
            defaults of :class:`~cond_id.methods.cond_id.UnlearnConfig`.
    """

    out_dir: str = "clones"
    unlearn: UnlearnConfig | None = None


def _save_wav(wav, sr: int, out_dir: str, name: str) -> str:
    """Write ``wav`` as ``<out_dir>/<name>.wav`` at ``sr`` Hz; return its path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name + ".wav")
    sf.write(path, np.asarray(wav, dtype=np.float32), sr)
    return path


def _synth_clone(host, speaker: str, ref: str, sentence: str) -> np.ndarray:
    """Synthesize ``sentence`` in ``speaker``'s voice from the held-out clip ``ref``.

    The reference is passed explicitly so cloning uses a held-out clip rather than
    the method-visible training references -- the intended reproducible path.
    """
    return np.asarray(host.tts(sentence, speaker, ref_override=ref), dtype=np.float32)


def _first_ref(refs: dict[str, list[str]], speaker: str) -> str | None:
    """Return the first held-out reference clip for ``speaker``, or ``None``."""
    clips = refs.get(speaker) or []
    return clips[0] if clips else None


def build_splits(
    host,
    forget: Sequence[str],
    retain_seen: Sequence[str],
    retain_unseen: Sequence[str],
    sentences: Sequence[str],
    seed: int,
):
    """Build one evaluation split for a fixed forget / retain assignment.

    Selects the held-out reference partitions for every speaker in the
    split -- ``centroid_refs`` (identity centroid), ``heldout_refs`` (cloning
    references) and the ``impostor`` clip set (different speakers, for the erasure
    floor) -- and returns a split dict with empty ``cB``/``cE``/``cRL`` ready for
    :func:`synth_baseline`, :func:`synth_edited_clones` and the relearn stage to fill.

    Args:
        host: A constructed TTS host (its speaker pool defines the impostor set).
        forget: Speakers whose identity is to be erased.
        retain_seen: Retain speakers provided to unlearning as anchors.
        retain_unseen: Retain speakers held out from unlearning.
        sentences: Sentence bank; each clone is synthesized once per sentence,
            positionally aligned with the ``cB``/``cE`` lists.
        seed: Determines the held-out reference partition for every speaker.

    Returns:
        A list with a single split dict following the shared data contract.
    """
    from cond_id.datasets import pool

    forget = list(forget)
    retain_seen = list(retain_seen)
    retain_unseen = list(retain_unseen)
    retain = retain_seen + retain_unseen
    speakers = forget + retain
    sentences = list(sentences)

    # Impostor clips come from every remaining pool speaker (disjoint from the
    # split), giving the WavLM erasure floor a same-corpus different-speaker
    # reference rather than an absolute cosine threshold. The pool is requested
    # generously so the impostor set is the full remainder, independent of where
    # the split speakers fall in the pool ordering.
    used = set(speakers)
    all_speakers = pool(len(used) + max(len(used), 5))
    impostor_speakers = [sp for sp in all_speakers if sp not in used]

    centroid_refs = {
        sp: held_clips(sp, "centroid", seed=seed) for sp in speakers + impostor_speakers
    }
    heldout_refs = {sp: held_clips(sp, "heldout", seed=seed) for sp in speakers}
    # Attack references for the relearn adversary: a held-out partition disjoint from
    # both the train slice and the cloning references, so the attack is honest.
    attack_refs = {sp: held_clips(sp, "attack", seed=seed) for sp in forget}
    impostor = [p for sp in impostor_speakers for p in centroid_refs.get(sp, [])]

    split = {
        "seed": int(seed),
        "split_index": 0,
        "forget": forget,
        "retain_seen": retain_seen,
        "retain_unseen": retain_unseen,
        "impostor": impostor,
        "centroid_refs": centroid_refs,
        "heldout_refs": heldout_refs,
        "attack_refs": attack_refs,
        "sentences": sentences,
        "cB": {},
        "cE": {},
        "cRL": {sp: [] for sp in forget},
    }
    return [split]


def synth_baseline(host, split, config: CloneConfig | None = None):
    """Synthesize the baseline clones ``cB`` for every speaker in ``split``.

    Runs on the pristine model (no unlearning), cloning each speaker's held-out
    reference through the whole sentence bank. Populates and returns ``split["cB"]``
    as ``{speaker: [wav_path, ...]}`` sentence-major.

    Args:
        host: The TTS host (used unmodified).
        split: A split dict from :func:`build_splits`.
        config: Output/hyperparameter configuration; defaults to :class:`CloneConfig`.

    Returns:
        The ``cB`` mapping (also stored back into ``split["cB"]``).
    """
    cfg = config or CloneConfig()
    sr = host.sr
    speakers = split["forget"] + split["retain_seen"] + split["retain_unseen"]
    heldout_refs = split["heldout_refs"]
    sentences = split["sentences"]

    cB: dict[str, list[str]] = {}
    for sp in speakers:
        ref = _first_ref(heldout_refs, sp)
        paths: list[str] = []
        if ref is not None:
            for i, sentence in enumerate(sentences):
                wav = _synth_clone(host, sp, ref, sentence)
                paths.append(_save_wav(wav, sr, cfg.out_dir, f"cB_{sp}_{i}"))
        cB[sp] = paths

    split["cB"] = cB
    return cB


def synth_edited_clones(host, split, config: CloneConfig | None = None):
    """Clone every speaker under the CURRENTLY-active unlearning (no context block).

    Assumes the caller has already entered
    :func:`~cond_id.methods.cond_id.applied_unlearning` -- so the same open
    unlearning block can be shared with the relearn adversary
    (:func:`cond_id.eval.relearn.relearn_attack`), which must run on the
    still-unlearned host. Populates and returns ``split["cE"]``.
    """
    cfg = config or CloneConfig()
    sr = host.sr
    forget = split["forget"]
    retain_seen = split["retain_seen"]
    speakers = forget + retain_seen + split["retain_unseen"]
    heldout_refs = split["heldout_refs"]
    sentences = split["sentences"]

    cE: dict[str, list[str]] = {}
    for sp in speakers:
        ref = _first_ref(heldout_refs, sp)
        paths: list[str] = []
        if ref is not None:
            for i, sentence in enumerate(sentences):
                wav = _synth_clone(host, sp, ref, sentence)
                paths.append(_save_wav(wav, sr, cfg.out_dir, f"cE_{sp}_{i}"))
        cE[sp] = paths

    split["cE"] = cE
    return cE
