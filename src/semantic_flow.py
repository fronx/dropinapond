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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from storage import load_ego_graph
from embeddings import get_embedding_service
from semantic_flow import (
    build_structural_matrix,
    load_phrase_data,
    compute_semantic_affinity_matrix,
    blend_matrices,
    compute_fields,
    detect_clusters,
    generate_suggestions,
    compute_all_phrase_similarities,
    build_analysis_output,
    write_analysis,
)
from semantic_flow.coherence import compute_coherence as compute_semantic_coherence


@dataclass
class Params:
    name: str = "fronx"
    alpha: float = 0.4        # blend: 1.0 = structural only
    cos_min: float = 0.25     # ignore phrase-pair cosine below this
    suggest_k: int = 3        # top-N non-edges suggested per node
    suggest_pool: int = 15    # how many nearest-by-mean to check
    export_dir: Optional[Path] = None

def analyze(params: Params) -> Path:
    root = Path(__file__).parent.parent
    ego_dir = root / "data" / "ego_graphs" / params.name
    out_dir = params.export_dir or (root / "data" / "analyses")

    embedding_service = get_embedding_service()
    ego = load_ego_graph(ego_dir, embedding_service)

    nodes = [ego.focal] + [n for n in ego.nodes if n != ego.focal]
    idx = {nid: i for i, nid in enumerate(nodes)}

    S, existing_edges = build_structural_matrix(ego_dir, nodes, idx)
    phrase_E, phrase_w, mean_vec = load_phrase_data(embedding_service, params.name, nodes)
    A = compute_semantic_affinity_matrix(S, nodes, phrase_E, phrase_w, cos_min=params.cos_min)
    W = blend_matrices(S, A, params.alpha)

    # - F: Raw mutual predictability (symmetric sqrt of A * A.T)
    # - D: Semantic distance (1 - cosine similarity of mean embeddings)
    # - F_MB: Markov-blanket predictability (coupling given context)
    # - E_MB: Exploration potential (F_MB * (1 - D))
    F, D, F_MB, E_MB = compute_fields(A, mean_vec, nodes)

    clusters = detect_clusters(W, nodes)
    suggestions = generate_suggestions(
        nodes, mean_vec, phrase_E, phrase_w, existing_edges,
        cos_min=params.cos_min, suggest_k=params.suggest_k, suggest_pool=params.suggest_pool
    )

    # Compute coherence using existing clusters as basis
    coherence = compute_semantic_coherence(nodes, clusters, F, D, F_MB)

    # Compute phrase-level similarities for UI display
    # Use 0.3 threshold to catch all potential matches, return top 100
    # GUI will filter by similarity >= 0.65 for "shared" vs "unique"
    phrase_similarities = compute_all_phrase_similarities(
        embedding_service, params.name, nodes,
        similarity_threshold=0.3, top_k=100
    )

    analysis = build_analysis_output(
        params.name, vars(params), S, A, W, F, D, F_MB, E_MB, clusters, suggestions, nodes, coherence, phrase_similarities
    )

    return write_analysis(analysis, out_dir, params.name)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run src/semantic_flow.py <graph_name> [--alpha 0.4] [--cos-min 0.25]")
        sys.exit(1)

    params = Params(name=sys.argv[1])
    for i, arg in enumerate(sys.argv[2:], start=2):
        if arg == "--alpha" and i + 1 < len(sys.argv):
            params.alpha = float(sys.argv[i + 1])
        elif arg == "--cos-min" and i + 1 < len(sys.argv):
            params.cos_min = float(sys.argv[i + 1])

    analyze(params)
