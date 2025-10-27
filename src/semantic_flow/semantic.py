"""
Semantic Affinity Computation
-----------------------------

Load phrase data and compute semantic affinity between nodes based on
phrase-level embedding similarity.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np


def _phrase_matrix(embedding_service, graph_name: str, node_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (E, w) where E:(m,d) unit-norm embeddings, w:(m,) phrase weights.

    Args:
        embedding_service: ChromaDB embedding service
        graph_name: Name of the ego graph
        node_id: ID of the node

    Returns:
        E: (m, d) array of unit-normalized phrase embeddings
        w: (m,) array of phrase weights
    """
    data = embedding_service.get_all_node_phrases(graph_name, node_id) or {}
    vecs, ws = [], []
    for _, p in data.items():
        v = np.asarray(p["embedding"], dtype=float)
        n = np.linalg.norm(v)
        if n > 0:
            vecs.append(v / n)
            ws.append(float(p.get("metadata", {}).get("weight", 1.0)))
    if not vecs:
        return np.zeros((0, 1)), np.zeros(0)
    E = np.stack(vecs, axis=0)
    w = np.asarray(ws)
    return E, w


def _weighted_mean_embedding(E: np.ndarray, w: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute weighted mean of phrase embeddings, normalized to unit length.

    Args:
        E: (m, d) phrase embeddings
        w: (m,) phrase weights
        eps: Small constant for numerical stability

    Returns:
        (d,) unit-normalized mean embedding
    """
    if E.size == 0:
        return np.zeros(1)
    ww = w / (w.sum() + eps)
    m = (ww[:, None] * E).sum(axis=0)
    n = np.linalg.norm(m)
    return m / (n + eps)


def _semantic_affinity(
    Ei: np.ndarray,
    wi: np.ndarray,
    Ej: np.ndarray,
    wj: np.ndarray,
    cos_min: float = 0.2,
    eps: float = 1e-12
) -> float:
    """
    Phrase-level cosine affinity, weighted by phrase weights.

    Computes weighted average of cosine similarities between phrase pairs,
    considering only pairs with similarity >= cos_min.

    Args:
        Ei: (mi, d) phrase embeddings for node i
        wi: (mi,) phrase weights for node i
        Ej: (mj, d) phrase embeddings for node j
        wj: (mj,) phrase weights for node j
        cos_min: Minimum cosine similarity threshold
        eps: Small constant for numerical stability

    Returns:
        Affinity score in [0, 1]
    """
    if Ei.size == 0 or Ej.size == 0:
        return 0.0
    S = Ei @ Ej.T
    mask = (S >= cos_min)
    if not np.any(mask):
        return 0.0
    W = np.outer(wi, wj)
    num = float((S * W * mask).sum())
    den = float((W * mask).sum() + eps)
    val = max(0.0, min(1.0, num / den))
    return val


def load_phrase_data(
    embedding_service,
    graph_name: str,
    nodes: List[str]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load phrase embeddings and weights for all nodes.

    Args:
        embedding_service: ChromaDB embedding service
        graph_name: Name of the ego graph
        nodes: List of node IDs

    Returns:
        Tuple of (phrase_E, phrase_w, mean_vec) where:
        - phrase_E: Dict mapping node ID to (m, d) phrase embeddings
        - phrase_w: Dict mapping node ID to (m,) phrase weights
        - mean_vec: Dict mapping node ID to (d,) weighted mean embedding
    """
    phrase_E, phrase_w, mean_vec = {}, {}, {}
    for nid in nodes:
        E, w = _phrase_matrix(embedding_service, graph_name, nid)
        phrase_E[nid] = E
        phrase_w[nid] = w
        mean_vec[nid] = _weighted_mean_embedding(E, w)
    return phrase_E, phrase_w, mean_vec


def compute_semantic_affinity_matrix(
    S: np.ndarray,
    nodes: List[str],
    phrase_E: Dict[str, np.ndarray],
    phrase_w: Dict[str, np.ndarray],
    cos_min: float = 0.25
) -> np.ndarray:
    """
    Compute semantic affinity matrix A on existing edges.

    Args:
        S: (n, n) structural matrix (used to identify existing edges)
        nodes: Ordered list of node IDs
        phrase_E: Dict mapping node ID to phrase embeddings
        phrase_w: Dict mapping node ID to phrase weights
        cos_min: Minimum cosine similarity threshold

    Returns:
        A: (n, n) semantic affinity matrix
    """
    n = len(nodes)
    A = np.zeros((n, n))
    for i, src in enumerate(nodes):
        Ei, wi = phrase_E[src], phrase_w[src]
        for j, tgt in enumerate(nodes):
            if i == j or S[i, j] <= 0.0:
                continue
            Ej, wj = phrase_E[tgt], phrase_w[tgt]
            A[i, j] = _semantic_affinity(Ei, wi, Ej, wj, cos_min=cos_min)
    return A
