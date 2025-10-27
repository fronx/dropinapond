"""
Blending and Field Computations
-------------------------------

Blend structural and semantic matrices, then compute various predictability
and distance fields for ego graph navigation.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np


def blend_matrices(S: np.ndarray, A: np.ndarray, alpha: float) -> np.ndarray:
    """
    Blend structural and semantic matrices into effective weight matrix.

    W = alpha * S + (1 - alpha) * A

    Args:
        S: (n, n) structural matrix
        A: (n, n) semantic affinity matrix
        alpha: Blending parameter (1.0 = structural only, 0.0 = semantic only)

    Returns:
        W: (n, n) blended effective weight matrix
    """
    return alpha * S + (1 - alpha) * A


def compute_fields(
    A: np.ndarray,
    mean_vec: Dict[str, np.ndarray],
    nodes: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute predictability and distance fields from semantic data.

    Returns four matrices:
    - F: Raw mutual predictability (symmetric sqrt of A * A.T)
    - D: Semantic distance (1 - cosine similarity of mean embeddings)
    - F_MB: Markov-blanket predictability (coupling given context)
    - E_MB: Exploration potential (F_MB * (1 - D))

    All matrices are normalized to [0, 1] for comparability.

    Args:
        A: (n, n) semantic affinity matrix
        mean_vec: Dict mapping node ID to weighted mean embedding
        nodes: Ordered list of node IDs

    Returns:
        Tuple of (F, D, F_MB, E_MB) matrices
    """
    n = len(nodes)

    # Stack mean embeddings and normalize
    M = np.stack([mean_vec[nid] for nid in nodes], axis=0)
    M_norm = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)

    # Predictability: symmetric mutual affinity
    F = np.sqrt(A * A.T)

    # Distance: 1 - cosine similarity between node mean embeddings
    D = 1 - np.clip(M_norm @ M_norm.T, -1.0, 1.0)

    # Normalize both to [0,1] for comparability
    F /= F.max() + 1e-12
    D /= D.max() + 1e-12

    # Markov-blanket coupling (mutual predictability given context)
    # Normalize rows of A to get local conditional probabilities, then
    # take elementwise product with its transpose to get symmetric coupling.
    row_sums = np.sum(A, axis=1, keepdims=True) + 1e-12
    A_norm = A / row_sums
    F_MB = A_norm * A_norm.T

    # Normalize for comparability
    F_MB /= F_MB.max() + 1e-12

    # Exploration potential: combines compatibility and contrast
    E_MB = F_MB * (1 - D)

    return F, D, F_MB, E_MB
