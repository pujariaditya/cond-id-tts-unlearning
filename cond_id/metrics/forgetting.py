"""Forgetting metrics: speaker similarity and spk-ZRF.

This module provides the speaker zero-shot retention-forgetting (spk-ZRF) score, a
standalone similarity-based forgetting helper. (The metric actually reported by the
pipeline is ``forget_erasure_abs``, computed per-utterance in
``cond_id/eval/score.py``.) For a speaker, let ``s_base``
be the cosine similarity between the *original* model's synthesis and the
speaker's real reference centroid, and ``s_edit`` the same for the *edited*
model. With an impostor floor ``f`` (the similarity a different speaker scores
against this centroid)::

    spk-ZRF = clip((s_base - s_edit) / (s_base - f), 0, 1)

``spk-ZRF = 0`` means no forgetting; ``spk-ZRF = 1`` means the speaker is
de-identified down to the impostor floor.
"""

import torch
import torch.nn.functional as F


def cosine_to_centroid(embeddings: torch.Tensor, reference_embeddings: torch.Tensor) -> float:
    """Mean cosine similarity of ``embeddings`` to a reference centroid.

    Args:
        embeddings: ``(N, D)`` L2-normalized synthesis embeddings.
        reference_embeddings: ``(M, D)`` L2-normalized reference embeddings; their
            mean (re-normalized) is the speaker centroid.

    Returns:
        The mean cosine similarity as a float.
    """
    centroid = F.normalize(reference_embeddings.mean(0, keepdim=True), dim=-1)
    return float(F.cosine_similarity(embeddings, centroid, dim=-1).mean())


def spk_zrf(base_sim: float, edited_sim: float, impostor_floor: float) -> float:
    """Speaker zero-retain forgetting score, clipped to ``[0, 1]``.

    Args:
        base_sim: Similarity of the original model's synthesis to the speaker.
        edited_sim: Similarity of the edited model's synthesis to the speaker.
        impostor_floor: Similarity a different speaker scores against the centroid.

    Returns:
        ``clip((base_sim - edited_sim) / (base_sim - impostor_floor), 0, 1)``.
    """
    return max(0.0, min((base_sim - edited_sim) / max(base_sim - impostor_floor, 1e-6), 1.0))
