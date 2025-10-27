"""
Structural Matrix Computation
-----------------------------

Build the structural matrix S from edges.json, representing the factual
topology of the ego graph.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def build_structural_matrix(
    ego_dir: Path,
    nodes: List[str],
    idx: Dict[str, int]
) -> np.ndarray:
    """
    Build directed structural matrix S from edges.json.

    Args:
        ego_dir: Path to ego graph directory
        nodes: Ordered list of node IDs
        idx: Mapping from node ID to matrix index

    Returns:
        S: (n, n) structural weight matrix where S[i,j] is the edge weight from i to j
    """
    n = len(nodes)
    S = np.zeros((n, n))

    edges_path = ego_dir / "edges.json"
    with open(edges_path) as f:
        edges_list = json.load(f)

    for e in edges_list:
        src, tgt = e["source"], e["target"]
        if src in idx and tgt in idx:
            w = float(e.get("actual", 0.3) or 0.0)
            S[idx[src], idx[tgt]] = np.clip(w, 0.0, 1.0)

    return S
