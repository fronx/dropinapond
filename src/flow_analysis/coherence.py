"""
Coherence Metrics
-----------------

Compute semantic region coherence based on precomputed clusters and
semantic fields (F, D, F_MB).
"""

from __future__ import annotations
from typing import Dict, List

import numpy as np


def _mean_pairwise(mat: np.ndarray, idxs: List[int]) -> float:
    if len(idxs) < 2:
        return 0.0
    vals = []
    for a in range(len(idxs)):
        ia = idxs[a]
        for b in range(a + 1, len(idxs)):
            ib = idxs[b]
            vals.append(float(mat[ia, ib]))
    return float(np.mean(vals)) if vals else 0.0


def _mean_cross(mat: np.ndarray, idxs_in: List[int], idxs_out: List[int]) -> float:
    if not idxs_in or not idxs_out:
        return 0.0
    vals = [float(mat[i, j]) for i in idxs_in for j in idxs_out]
    return float(np.mean(vals)) if vals else 0.0


def _cut_weight(mat: np.ndarray, idxs_in: List[int], idxs_out: List[int]) -> float:
    if not idxs_in or not idxs_out:
        return 0.0
    return float(np.sum(mat[np.ix_(idxs_in, idxs_out)]))


def _volume(mat: np.ndarray, idxs_in: List[int], n: int) -> float:
    if not idxs_in:
        return 0.0
    return float(np.sum(mat[np.ix_(idxs_in, list(range(n)))]))


def compute_coherence(
    nodes: List[str],
    clusters: List[List[str]],
    F: np.ndarray,
    D: np.ndarray,
    F_MB: np.ndarray,
) -> Dict:
    """
    Compute region- and node-level coherence using provided clusters.

    Uses F (global mutual similarity), F_MB (blanket similarity) and D (distance).

    Args:
        nodes: Ordered list of node IDs
        clusters: List of clusters (each is a list of node IDs)
        F: (n, n) predictability matrix (symmetric)
        D: (n, n) semantic distance matrix (symmetric)
        F_MB: (n, n) blanket predictability matrix (symmetric)

    Returns:
        Dict with region summaries and node fit metrics
    """
    n = len(nodes)
    name_to_idx = {nid: i for i, nid in enumerate(nodes)}

    # Select semantic weight for conductance: prefer F_MB if available
    use_MB = np.any(F_MB > 0)
    W = F_MB if use_MB else F

    region_idx_lists = [[name_to_idx[x] for x in region if x in name_to_idx] for region in clusters]

    regions_summary = []
    for r_ix, region in enumerate(clusters):
        idxs_in = region_idx_lists[r_ix]
        idxs_out = [k for k in range(n) if k not in idxs_in]

        internal_F = _mean_pairwise(F, idxs_in)
        internal_MB = _mean_pairwise(F_MB, idxs_in) if use_MB else 0.0
        internal_D = _mean_pairwise(D, idxs_in)

        external_F = _mean_cross(F, idxs_in, idxs_out)
        external_MB = _mean_cross(F_MB, idxs_in, idxs_out) if use_MB else 0.0
        external_D = _mean_cross(D, idxs_in, idxs_out)

        cut_W = _cut_weight(W, idxs_in, idxs_out)
        vol_W = _volume(W, idxs_in, n)
        conductance_W = (cut_W / (vol_W + 1e-12)) if vol_W > 0 else 0.0

        # Distance-based silhouette
        silhouettes = []
        for u in idxs_in:
            others_in = [v for v in idxs_in if v != u]
            a_u = float(np.mean([D[u, v] for v in others_in])) if others_in else 0.0
            b_candidates = []
            for ix2, other in enumerate(region_idx_lists):
                if ix2 == r_ix or not other:
                    continue
                b_candidates.append(float(np.mean([D[u, v] for v in other])))
            b_u = min(b_candidates) if b_candidates else a_u
            s_u = (1.0 - (a_u / (b_u + 1e-12))) if b_u > 0 else 0.0
            silhouettes.append(s_u)
        silhouette_D = float(np.mean(silhouettes)) if silhouettes else 0.0

        coherence_F = internal_F / (internal_F + external_F + 1e-12)
        coherence_MB = (internal_MB / (internal_MB + external_MB + 1e-12)) if use_MB else 0.0

        regions_summary.append({
            "nodes": region,
            "internal_F": internal_F,
            "external_F": external_F,
            "internal_MB": internal_MB,
            "external_MB": external_MB,
            "internal_D": internal_D,
            "external_D": external_D,
            "conductance_sem": conductance_W,
            "silhouette_D": silhouette_D,
            "coherence_F": coherence_F,
            "coherence_MB": coherence_MB,
        })

    node_fits = {}
    for r_ix, region in enumerate(clusters):
        idxs_in = region_idx_lists[r_ix]
        idxs_out = [k for k in range(n) if k not in idxs_in]
        for u in idxs_in:
            nid = nodes[u]
            others_in = [v for v in idxs_in if v != u]
            avg_in = float(np.mean([W[u, v] for v in others_in])) if others_in else 0.0
            avg_out = float(np.mean([W[u, v] for v in idxs_out])) if idxs_out else 0.0
            fit_diff = avg_in - avg_out
            fit_ratio = avg_in / (avg_out + 1e-12) if avg_out > 0 else float('inf') if avg_in > 0 else 0.0
            node_fits[nid] = {
                "region_index": r_ix,
                "avg_in": avg_in,
                "avg_out": avg_out,
                "fit_diff": fit_diff,
                "fit_ratio": fit_ratio,
            }

    return {
        "method": "metrics.clusters_based",
        "regions": regions_summary,
        "nodes": node_fits,
    }

