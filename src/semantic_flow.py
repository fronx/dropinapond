"""
Semantic-Structural Flow Analysis
---------------------------------

Blend factual topology (edges.json) with semantic affinity derived from phrase
embeddings in ChromaDB, then simulate diffusion on the blended graph.

Outputs a JSON under data/analyses/<name>_latest.json that your existing UI
(EgoGraphView) can already visualize.

Usage:
    uv run src/semantic_flow.py <graph_name> [--alpha 0.6] [--cos-min 0.2]
"""

from __future__ import annotations
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime

# Local project imports
from storage import load_ego_graph
from embeddings import get_embedding_service

# Semantic flow modules
from semantic_flow import (
    build_structural_matrix,
    load_phrase_data,
    compute_semantic_affinity_matrix,
    blend_matrices,
    compute_fields,
    detect_clusters,
    generate_suggestions,
    build_analysis_output,
)


# ---------- Parameters ----------

@dataclass
class Params:
    name: str = "fronx"
    alpha: float = 0.4        # blend: 1.0 = structural only
    cos_min: float = 0.25     # ignore phrase-pair cosine below this
    suggest_k: int = 3        # top-N non-edges suggested per node
    suggest_pool: int = 15    # how many nearest-by-mean to check
    export_dir: Optional[Path] = None


# ---------- Small utilities ----------

def _timestamped_and_latest_paths(base_dir: Path, name: str) -> Tuple[Path, Path]:
    base_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"{name}_{ts}.json", base_dir / f"{name}_latest.json"


# ---------- Core analysis ----------

def analyze(params: Params) -> Path:
    root = Path(__file__).parent.parent
    ego_dir = root / "data" / "ego_graphs" / params.name
    out_dir = params.export_dir or (root / "data" / "analyses")

    embedding_service = get_embedding_service()
    ego = load_ego_graph(ego_dir, embedding_service)

    nodes = [ego.focal] + [n for n in ego.nodes if n != ego.focal]
    idx = {nid: i for i, nid in enumerate(nodes)}

    edges_path = ego_dir / "edges.json"
    with open(edges_path) as f:
        edges_list = json.load(f)
    existing_edges = {(e["source"], e["target"]) for e in edges_list}

    S = build_structural_matrix(ego_dir, nodes, idx)

    phrase_E, phrase_w, mean_vec = load_phrase_data(
        embedding_service, params.name, nodes
    )

    A = compute_semantic_affinity_matrix(
        S, nodes, phrase_E, phrase_w, cos_min=params.cos_min
    )

    W = blend_matrices(S, A, params.alpha)

    F, D, F_MB, E_MB = compute_fields(A, mean_vec, nodes)

    clusters = detect_clusters(W, nodes)

    suggestions = generate_suggestions(
        nodes, mean_vec, phrase_E, phrase_w, existing_edges,
        cos_min=params.cos_min,
        suggest_k=params.suggest_k,
        suggest_pool=params.suggest_pool
    )

    analysis = build_analysis_output(
        graph_name=params.name,
        params=vars(params),
        S=S,
        A=A,
        W=W,
        F=F,
        D=D,
        F_MB=F_MB,
        E_MB=E_MB,
        clusters=clusters,
        suggestions=suggestions,
        nodes=nodes
    )

    ts_path, latest_path = _timestamped_and_latest_paths(out_dir, params.name)
    with open(ts_path, "w") as f:
        json.dump(analysis, f, indent=2)
    with open(latest_path, "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"[ok] Wrote {ts_path.name}")
    print(f"[ok] Updated {latest_path.name}")
    return latest_path


# ---------- CLI ----------

def _parse_args(argv: List[str]) -> Params:
    p = Params()
    if len(argv) >= 2:
        p.name = argv[1]
    for i, a in enumerate(argv[2:], start=2):
        if a == "--alpha" and i + 1 < len(argv):
            p.alpha = float(argv[i + 1])
        if a == "--cos-min" and i + 1 < len(argv):
            p.cos_min = float(argv[i + 1])
    return p


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run src/semantic_flow.py <graph_name> [--alpha 0.6] [--cos-min 0.2]")
        sys.exit(1)
    params = _parse_args(sys.argv)
    analyze(params)
