"""
Semantic-Structural Flow Analysis
---------------------------------

Blend factual topology (edges.json) with semantic affinity derived from phrase
embeddings in ChromaDB, then simulate diffusion on the blended graph.

Outputs a JSON under data/analyses/analysis_latest.json that your existing UI
(EgoGraphView) can already visualize.

Usage:
    uv run src/semantic_flow.py [--alpha 0.6] [--cos-min 0.2]
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
    compute_phrase_contribution_breakdown,
    compute_standout_phrases,
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
    alpha: float = 0.4        # blend: 1.0 = structural only
    cos_min: float = 0.25     # ignore phrase-pair cosine below this
    suggest_k: int = 3        # top-N non-edges suggested per node
    suggest_pool: int = 15    # how many nearest-by-mean to check
    export_dir: Optional[Path] = None

def analyze(params: Params) -> Path:
    root = Path(__file__).parent.parent
    ego_dir = root / "data" / "ego_graph"
    out_dir = params.export_dir or (root / "data" / "analyses")

    embedding_service = get_embedding_service()
    ego = load_ego_graph(ego_dir, embedding_service)

    nodes = [ego.focal] + [n for n in ego.nodes if n != ego.focal]
    idx = {nid: i for i, nid in enumerate(nodes)}

    S, existing_edges = build_structural_matrix(ego_dir, nodes, idx)
    phrase_E, phrase_w, mean_vec = load_phrase_data(embedding_service, "ego_graph", nodes)
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
        embedding_service, "ego_graph", nodes,
        similarity_threshold=0.3, top_k=100
    )

    # Compute phrase contributions for each neighbor
    # This shows which focal phrases contribute most to predictability
    focal_id = nodes[0]
    focal_idx = idx[focal_id]
    phrase_contributions = {}
    for neighbor_id in nodes[1:]:
        contributions = compute_phrase_contribution_breakdown(
            embedding_service, "ego_graph", focal_id, neighbor_id,
            cos_min=params.cos_min, top_k=10
        )
        if contributions:
            phrase_contributions[neighbor_id] = contributions

    # Compute standout phrases for all neighbors
    # This shows which phrases make a person uniquely stand out in the network
    # by comparing their phrase affinity with you against the network baseline
    standout_phrases = {}
    neighbor_ids = [n for n in nodes if n != focal_id]

    # Compute standout for all neighbors (function already filters to positive scores)
    for neighbor_id in neighbor_ids:
        standout = compute_standout_phrases(
            embedding_service, "ego_graph", focal_id, neighbor_id,
            neighbor_ids, cos_min=params.cos_min, top_k=5
        )
        if standout:
            standout_phrases[neighbor_id] = standout

    analysis = build_analysis_output(
        None, vars(params), S, A, W, F, D, F_MB, E_MB, clusters, suggestions, nodes, coherence, phrase_similarities, phrase_contributions, standout_phrases
    )

    return write_analysis(analysis, out_dir, None)


if __name__ == "__main__":
    params = Params()
    for i, arg in enumerate(sys.argv[1:], start=1):
        if arg == "--alpha" and i + 1 < len(sys.argv):
            params.alpha = float(sys.argv[i + 1])
        elif arg == "--cos-min" and i + 1 < len(sys.argv):
            params.cos_min = float(sys.argv[i + 1])

    analyze(params)
