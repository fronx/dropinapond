"""
Structural Matrix Computation
-----------------------------

Build the structural matrix S from edges.json, representing the factual
topology of the ego graph.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np


def build_structural_matrix(
    edges: List[Tuple],
    nodes: List[str],
    idx: Dict[str, int]
) -> Tuple[np.ndarray, Set[Tuple[str, str]]]:
    """
    Build directed structural matrix S from edges.

    Args:
        edges: List of edge tuples from EgoData (u, v, dims) or (u, v, w) or (u, v)
        nodes: Ordered list of node IDs
        idx: Mapping from node ID to matrix index

    Returns:
        S: (n, n) structural weight matrix where S[i,j] is the edge weight from i to j
        existing_edges: Set of (source, target) tuples for existing edges
    """
    n = len(nodes)
    S = np.zeros((n, n))
    existing_edges = set()

    for edge in edges:
        if len(edge) == 2:
            src, tgt = edge
            w = 0.3  # Default weight
        elif len(edge) == 3:
            src, tgt, dims = edge
            if dims is None:
                w = 0.3
            elif isinstance(dims, dict):
                w = float(dims.get("actual", 0.3) or 0.0)
            else:
                w = float(dims)
        else:
            continue

        if src in idx and tgt in idx:
            S[idx[src], idx[tgt]] = np.clip(w, 0.0, 1.0)
            existing_edges.add((src, tgt))

    return S, existing_edges
