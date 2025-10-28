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


def compute_phrase_contribution_breakdown(
    embedding_service,
    graph_name: str,
    focal_id: str,
    neighbor_id: str,
    cos_min: float = 0.25,
    top_k: int = 10
) -> List[dict]:
    """
    Compute phrase-level contribution breakdown showing which focal phrases
    contribute most to the semantic affinity with a neighbor.

    For each focal phrase, computes the weighted sum of similarities to all
    neighbor phrases above the threshold. Returns the top-k contributing phrases.

    Args:
        embedding_service: ChromaDB embedding service
        graph_name: Name of the ego graph
        focal_id: ID of the focal node
        neighbor_id: ID of the neighbor
        cos_min: Minimum cosine similarity threshold
        top_k: Number of top contributing phrases to return

    Returns:
        List of dicts with keys:
            - phrase: Focal phrase text
            - contribution: Total weighted contribution score
            - weight: Original phrase weight
            - top_matches: List of top 3 neighbor phrases that matched (with similarity)
    """
    # Load phrase data for both nodes
    focal_data = embedding_service.get_all_node_phrases(graph_name, focal_id) or {}
    neighbor_data = embedding_service.get_all_node_phrases(graph_name, neighbor_id) or {}

    if not focal_data or not neighbor_data:
        return []

    # Extract embeddings and weights for neighbor
    neighbor_phrases = []
    for phrase_id, p in neighbor_data.items():
        v = np.asarray(p["embedding"], dtype=float)
        n = np.linalg.norm(v)
        if n > 0:
            neighbor_phrases.append({
                'text': p['text'],
                'embedding': v / n,  # Normalize
                'weight': float(p.get("metadata", {}).get("weight", 1.0))
            })

    if not neighbor_phrases:
        return []

    # Stack neighbor embeddings for vectorized computation
    neighbor_E = np.stack([p['embedding'] for p in neighbor_phrases], axis=0)
    neighbor_w = np.array([p['weight'] for p in neighbor_phrases])

    # Compute contributions for each focal phrase
    contributions = []
    for phrase_id, focal_p in focal_data.items():
        focal_v = np.asarray(focal_p["embedding"], dtype=float)
        focal_n = np.linalg.norm(focal_v)
        if focal_n == 0:
            continue

        focal_embedding = focal_v / focal_n
        focal_weight = float(focal_p.get("metadata", {}).get("weight", 1.0))
        focal_text = focal_p['text']

        # Compute similarities to all neighbor phrases
        similarities = neighbor_E @ focal_embedding

        # Apply threshold mask
        mask = similarities >= cos_min

        if not np.any(mask):
            continue

        # Compute weighted contribution
        contribution = float((similarities * neighbor_w * mask).sum() * focal_weight)

        # Find top 3 matching neighbor phrases
        valid_indices = np.where(mask)[0]
        if len(valid_indices) > 0:
            # Get similarities for valid matches
            valid_sims = similarities[valid_indices]
            # Sort by similarity descending
            sorted_indices = valid_indices[np.argsort(-valid_sims)]

            top_matches = []
            for idx in sorted_indices[:3]:
                top_matches.append({
                    'phrase': neighbor_phrases[idx]['text'],
                    'similarity': float(similarities[idx])
                })
        else:
            top_matches = []

        contributions.append({
            'phrase': focal_text,
            'contribution': contribution,
            'weight': focal_weight,
            'top_matches': top_matches
        })

    # Sort by contribution and return top-k
    contributions.sort(key=lambda x: x['contribution'], reverse=True)
    return contributions[:top_k]


def compute_standout_phrases(
    embedding_service,
    graph_name: str,
    focal_id: str,
    neighbor_id: str,
    all_neighbor_ids: List[str],
    cos_min: float = 0.25,
    top_k: int = 10
) -> List[dict]:
    """
    Compute which of this neighbor's phrases make them stand out in the network.

    For each neighbor phrase, computes how much it resonates with the focal node
    compared to how much other neighbors' phrases typically resonate.

    This answers "what makes THIS person special?" by identifying their phrases
    that have unusually high affinity with you compared to the network baseline.

    Args:
        embedding_service: ChromaDB embedding service
        graph_name: Name of the ego graph
        focal_id: ID of the focal node
        neighbor_id: ID of the target neighbor
        all_neighbor_ids: List of all neighbor IDs (for computing network baseline)
        cos_min: Minimum cosine similarity threshold
        top_k: Number of top standout phrases to return

    Returns:
        List of dicts with keys:
            - phrase: Neighbor's phrase text that stands out
            - standout_score: How much more this phrase resonates vs baseline
            - target_affinity: Affinity between this phrase and focal node
            - mean_affinity: Average affinity of other neighbors' phrases with focal
            - weight: Neighbor phrase weight
            - top_matches: List of top 3 focal phrases that matched
    """
    # Load focal phrase data (YOUR phrases)
    focal_data = embedding_service.get_all_node_phrases(graph_name, focal_id) or {}
    if not focal_data:
        return []

    focal_phrases = []
    focal_texts = []
    focal_weights = []
    for _, p in focal_data.items():
        v = np.asarray(p["embedding"], dtype=float)
        n = np.linalg.norm(v)
        if n > 0:
            focal_phrases.append(v / n)
            focal_texts.append(p['text'])
            focal_weights.append(float(p.get("metadata", {}).get("weight", 1.0)))

    if not focal_phrases:
        return []

    focal_E = np.stack(focal_phrases, axis=0)
    focal_w = np.array(focal_weights)

    # Load target neighbor phrases (THEIR phrases - what makes them special)
    target_data = embedding_service.get_all_node_phrases(graph_name, neighbor_id) or {}
    if not target_data:
        return []

    target_phrases = []
    target_texts = []
    target_weights = []
    for _, p in target_data.items():
        v = np.asarray(p["embedding"], dtype=float)
        n = np.linalg.norm(v)
        if n > 0:
            target_phrases.append(v / n)
            target_texts.append(p['text'])
            target_weights.append(float(p.get("metadata", {}).get("weight", 1.0)))

    if not target_phrases:
        return []

    target_E = np.stack(target_phrases, axis=0)
    target_w = np.array(target_weights)

    # For each TARGET phrase, compute affinity with focal node
    # S_target: (n_target, m_focal) - rows are target phrases, cols are focal phrases
    S_target = target_E @ focal_E.T
    mask_target = (S_target >= cos_min)
    W_target = np.outer(target_w, focal_w)

    # Affinity of each target phrase with focal (sum over focal phrases)
    target_affinity = (S_target * W_target * mask_target).sum(axis=1)  # (n_target,)

    # Compute baseline: average affinity of other neighbors' phrases with focal
    other_neighbor_ids = [nid for nid in all_neighbor_ids if nid != neighbor_id]
    all_other_phrase_affinities = []

    for other_id in other_neighbor_ids:
        other_data = embedding_service.get_all_node_phrases(graph_name, other_id) or {}
        if not other_data:
            continue

        other_phrases = []
        other_weights = []
        for _, p in other_data.items():
            v = np.asarray(p["embedding"], dtype=float)
            n = np.linalg.norm(v)
            if n > 0:
                other_phrases.append(v / n)
                other_weights.append(float(p.get("metadata", {}).get("weight", 1.0)))

        if not other_phrases:
            continue

        other_E = np.stack(other_phrases, axis=0)
        other_w = np.array(other_weights)

        # Affinity of each other neighbor's phrases with focal
        S_other = other_E @ focal_E.T
        mask_other = (S_other >= cos_min)
        W_other = np.outer(other_w, focal_w)
        other_affinity = (S_other * W_other * mask_other).sum(axis=1)

        # Collect all phrase affinities from this neighbor
        all_other_phrase_affinities.extend(other_affinity.tolist())

    # Compute mean baseline affinity across all other neighbors' phrases
    if all_other_phrase_affinities:
        mean_baseline = np.mean(all_other_phrase_affinities)
    else:
        mean_baseline = 0.0

    # Compute standout scores: how much does this phrase exceed baseline?
    standout_scores = target_affinity - mean_baseline

    # Build results
    results = []
    for i, (text, standout, aff, weight) in enumerate(
        zip(target_texts, standout_scores, target_affinity, target_w)
    ):
        # Only include phrases with positive standout scores
        if standout <= 0:
            continue

        # Find top 3 matching focal phrases for this target phrase
        phrase_sims = S_target[i] * mask_target[i]
        top_indices = np.argsort(phrase_sims)[::-1][:3]

        top_matches = [
            {
                'phrase': focal_texts[j],
                'similarity': float(S_target[i, j])
            }
            for j in top_indices if S_target[i, j] >= cos_min
        ]

        results.append({
            'phrase': text,
            'standout_score': float(standout),
            'target_affinity': float(aff),
            'mean_affinity': float(mean_baseline),
            'weight': float(weight),
            'top_matches': top_matches
        })

    # Sort by standout score and return top-k
    results.sort(key=lambda x: x['standout_score'], reverse=True)
    return results[:top_k]
