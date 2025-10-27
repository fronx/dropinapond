"""
Compute semantic coherence regions and scores from an existing analysis JSON.

Usage:
    uv run src/compute_coherence.py <graph_name>

Reads data/analyses/<graph_name>_latest.json, computes communities on
semantic-only weights (prefer F_MB if present, otherwise F), scores regions
and nodes, then writes metrics.coherence back into the JSON.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from semantic_flow.coherence import compute_coherence as compute_semantic_coherence


def _load_analysis(root: Path, name: str) -> Tuple[Path, dict]:
    path = root / "data" / "analyses" / f"{name}_latest.json"
    with open(path) as f:
        data = json.load(f)
    return path, data


def _reconstruct_mats(analysis: dict) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    fields = analysis.get("metrics", {}).get("fields", {})
    edges = fields.get("edge_fields", {})
    edges_mb = fields.get("edge_fields_blanket", {})

    nodes = sorted(edges.keys())
    idx = {nid: i for i, nid in enumerate(nodes)}
    n = len(nodes)

    F = np.zeros((n, n))
    D = np.zeros((n, n))
    F_MB = np.zeros((n, n))

    for i, u in enumerate(nodes):
        row = edges.get(u, {})
        for v, vals in row.items():
            if v == u:
                continue
            j = idx.get(v)
            if j is None:
                continue
            F[i, j] = float(vals.get("predictability_raw", 0.0))
            D[i, j] = float(vals.get("distance_raw", 0.0))

    # symmetrize conservatively by averaging
    F = 0.5 * (F + F.T)
    D = 0.5 * (D + D.T)

    if edges_mb:
        for i, u in enumerate(nodes):
            row = edges_mb.get(u, {})
            for v, vals in row.items():
                if v == u:
                    continue
                j = idx.get(v)
                if j is None:
                    continue
                F_MB[i, j] = float(vals.get("predictability_blanket", 0.0))
        F_MB = 0.5 * (F_MB + F_MB.T)

    return nodes, F, D, F_MB


def compute_coherence_from_analysis(analysis: dict) -> dict:
    """Rebuild mats and compute coherence using the shared module.

    Uses existing metrics.clusters as the region basis.
    """
    nodes, F, D, F_MB = _reconstruct_mats(analysis)
    clusters = analysis.get("metrics", {}).get("clusters")
    return compute_semantic_coherence(nodes, clusters or [nodes], F, D, F_MB)


def main(name: str) -> None:
    root = Path(__file__).parent.parent
    path, analysis = _load_analysis(root, name)
    coherence = compute_coherence_from_analysis(analysis)
    analysis.setdefault("metrics", {}).setdefault("coherence", {})
    analysis["metrics"]["coherence"] = coherence
    with open(path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"[ok] Updated coherence in {path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run src/compute_coherence.py <graph_name>")
        sys.exit(1)
    main(sys.argv[1])
