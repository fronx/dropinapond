"""
Serialization
------------

Assemble analysis results into JSON-serializable format for export.
"""

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np


def matrix_to_dict(mat: np.ndarray, nodes: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Convert a matrix to nested dict format for JSON export.

    Only includes non-zero, non-diagonal entries.

    Args:
        mat: (n, n) matrix
        nodes: Ordered list of node IDs

    Returns:
        Nested dict mapping source -> target -> value
    """
    out: Dict[str, Dict[str, float]] = {}
    for i, u in enumerate(nodes):
        row = {}
        for j, v in enumerate(nodes):
            if i != j and mat[i, j] > 0:
                row[v] = float(mat[i, j])
        out[u] = row
    return out


def build_layers(
    S: np.ndarray,
    A: np.ndarray,
    W: np.ndarray,
    nodes: List[str]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Build layers section with structural, semantic, and effective edges.

    Args:
        S: Structural matrix
        A: Semantic affinity matrix
        W: Blended effective weight matrix
        nodes: Ordered list of node IDs

    Returns:
        Dict with keys: structural_edges, semantic_affinity, effective_edges
    """
    return {
        "structural_edges": matrix_to_dict(S, nodes),
        "semantic_affinity": matrix_to_dict(A, nodes),
        "effective_edges": matrix_to_dict(W, nodes),
    }


def build_edge_fields(
    F: np.ndarray,
    D: np.ndarray,
    nodes: List[str]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Build edge_fields section with raw predictability and distance.

    Args:
        F: Predictability field matrix
        D: Distance field matrix
        nodes: Ordered list of node IDs

    Returns:
        Nested dict with predictability_raw and distance_raw per edge
    """
    edge_fields = {}
    for i, src in enumerate(nodes):
        row = {}
        for j, tgt in enumerate(nodes):
            if i == j:
                continue
            row[tgt] = {
                "predictability_raw": float(F[i, j]),
                "distance_raw": float(D[i, j]),
            }
        edge_fields[src] = row
    return edge_fields


def build_edge_fields_blanket(
    F_MB: np.ndarray,
    E_MB: np.ndarray,
    nodes: List[str]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Build edge_fields_blanket section with Markov blanket metrics.

    Args:
        F_MB: Markov blanket predictability matrix
        E_MB: Exploration potential matrix
        nodes: Ordered list of node IDs

    Returns:
        Nested dict with predictability_blanket and exploration_potential per edge
    """
    edge_fields_MB = {}
    for i, src in enumerate(nodes):
        row = {}
        for j, tgt in enumerate(nodes):
            if i == j:
                continue
            row[tgt] = {
                "predictability_blanket": float(F_MB[i, j]),
                "exploration_potential": float(E_MB[i, j]),
            }
        edge_fields_MB[src] = row
    return edge_fields_MB


def build_analysis_output(
    graph_name: str,
    params: dict,
    S: np.ndarray,
    A: np.ndarray,
    W: np.ndarray,
    F: np.ndarray,
    D: np.ndarray,
    F_MB: np.ndarray,
    E_MB: np.ndarray,
    clusters: List[List[str]],
    suggestions: List[Dict],
    nodes: List[str],
    coherence: Optional[Dict] = None,
    phrase_similarities: Optional[Dict] = None,
    phrase_contributions: Optional[Dict] = None,
    standout_phrases: Optional[Dict] = None,
) -> dict:
    """
    Assemble complete analysis output for JSON export.

    Args:
        graph_name: Name of the ego graph
        params: Analysis parameters dict
        S: Structural matrix
        A: Semantic affinity matrix
        W: Blended effective weight matrix
        F: Predictability field matrix
        D: Distance field matrix
        F_MB: Markov blanket predictability matrix
        E_MB: Exploration potential matrix
        clusters: List of node clusters
        suggestions: List of suggestion dicts
        nodes: Ordered list of node IDs
        coherence: Optional coherence metrics dict
        phrase_similarities: Optional phrase-level similarity data
        phrase_contributions: Optional phrase contribution breakdown data
        standout_phrases: Optional network-aware standout phrase data

    Returns:
        Complete analysis dict ready for JSON serialization
    """
    metrics = {
        "version": "semantic-flow-1.0",
        "ego_graph_file": graph_name,
        "parameters": params,
        "metrics": {
            "clusters": clusters,
            "layers": build_layers(S, A, W, nodes),
            "fields": {
                "edge_fields": build_edge_fields(F, D, nodes),
                "edge_fields_blanket": build_edge_fields_blanket(F_MB, E_MB, nodes)
            }
        },
        "recommendations": {"semantic_suggestions": suggestions},
    }
    if coherence is not None:
        metrics["metrics"]["coherence"] = coherence
    if phrase_similarities is not None:
        # Extract just the focal node's similarities (first node in nodes list)
        focal_id = nodes[0] if nodes else None
        if focal_id and focal_id in phrase_similarities:
            metrics["metrics"]["phrase_similarities"] = phrase_similarities[focal_id]
    if phrase_contributions is not None:
        metrics["metrics"]["phrase_contributions"] = phrase_contributions
    if standout_phrases is not None:
        metrics["metrics"]["standout_phrases"] = standout_phrases
    return metrics


def write_analysis(analysis: dict, out_dir: Path, name: str) -> Path:
    """
    Write analysis to both timestamped and latest files.

    Args:
        analysis: Analysis dict to serialize
        out_dir: Output directory
        name: Graph name for filename

    Returns:
        Path to latest file
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_path = out_dir / f"{name}_{ts}.json"
    latest_path = out_dir / f"{name}_latest.json"

    for path in [ts_path, latest_path]:
        with open(path, "w") as f:
            json.dump(analysis, f, indent=2)

    print(f"[ok] Wrote {ts_path.name}")
    print(f"[ok] Updated {latest_path.name}")
    return latest_path
