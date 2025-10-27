"""
Suggestion Generation
--------------------

Generate high-affinity non-edge recommendations based on semantic similarity.
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple

import numpy as np

from .semantic import _semantic_affinity


def generate_suggestions(
    nodes: List[str],
    mean_vec: Dict[str, np.ndarray],
    phrase_E: Dict[str, np.ndarray],
    phrase_w: Dict[str, np.ndarray],
    existing_edges: Set[Tuple[str, str]],
    cos_min: float = 0.25,
    suggest_k: int = 3,
    suggest_pool: int = 15
) -> List[Dict[str, any]]:
    """
    Generate semantic suggestions for non-edges with high affinity.

    Args:
        nodes: Ordered list of node IDs
        mean_vec: Dict mapping node ID to weighted mean embedding
        phrase_E: Dict mapping node ID to phrase embeddings
        phrase_w: Dict mapping node ID to phrase weights
        existing_edges: Set of (source, target) tuples representing existing edges
        cos_min: Minimum cosine similarity threshold for affinity
        suggest_k: Number of suggestions to generate per node
        suggest_pool: Number of candidates to consider (by mean similarity)

    Returns:
        List of suggestion dicts with keys: source, target, affinity
    """
    n = len(nodes)

    # Compute mean embedding similarity matrix
    M = np.stack([mean_vec[nid] for nid in nodes], axis=0)
    M_norm = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    mean_sim = np.clip(M_norm @ M_norm.T, -1.0, 1.0)

    suggestions = []
    for i, src in enumerate(nodes):
        # Find candidates by mean similarity (excluding existing edges)
        cands = [
            (j, mean_sim[i, j])
            for j in range(n)
            if j != i and (src, nodes[j]) not in existing_edges
        ]
        cands.sort(key=lambda t: t[1], reverse=True)
        cands = cands[:suggest_pool]

        # Refine with phrase-level affinity
        Ei, wi = phrase_E[src], phrase_w[src]
        refined = []
        for j, _ in cands:
            tgt = nodes[j]
            Ej, wj = phrase_E[tgt], phrase_w[tgt]
            aff = _semantic_affinity(Ei, wi, Ej, wj, cos_min=cos_min)
            refined.append((tgt, aff))

        refined.sort(key=lambda t: t[1], reverse=True)
        for tgt, aff in refined[:suggest_k]:
            if aff > 0:
                suggestions.append({
                    "source": src,
                    "target": tgt,
                    "affinity": float(aff)
                })

    return suggestions
