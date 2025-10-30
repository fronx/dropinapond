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
from typing import Optional, List, Dict
import numpy as np

from storage import load_ego_graph, EgoData
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
class AnalysisParams:
    """Parameters for semantic flow analysis."""
    alpha: float = 0.4        # blend: 1.0 = structural only
    cos_min: float = 0.25     # ignore phrase-pair cosine below this
    suggest_k: int = 3        # top-N non-edges suggested per node
    suggest_pool: int = 15    # how many nearest-by-mean to check


@dataclass
class AnalysisResult:
    """Results of semantic flow analysis."""
    nodes: List[str]
    params: Dict
    S: np.ndarray  # Structural matrix
    A: np.ndarray  # Semantic affinity matrix
    W: np.ndarray  # Blended effective weight matrix
    F: np.ndarray  # Predictability field matrix
    D: np.ndarray  # Distance field matrix
    F_MB: np.ndarray  # Markov blanket predictability matrix
    E_MB: np.ndarray  # Exploration potential matrix
    clusters: List[List[str]]
    suggestions: List[Dict]
    coherence: Optional[Dict] = None
    phrase_similarities: Optional[Dict] = None
    phrase_contributions: Optional[Dict] = None
    standout_phrases: Optional[Dict] = None

def analyze(ego_data: EgoData, params: AnalysisParams, embedding_service) -> AnalysisResult:
    """
    Run semantic flow analysis on an ego graph.

    Args:
        ego_data: Ego graph data (from load_ego_graph or load_ego_graph_from_neo4j)
        params: Analysis parameters
        embedding_service: EmbeddingService instance for phrase operations

    Returns:
        AnalysisResult with all computed metrics
    """
    # Build node ordering (focal first, then others)
    nodes = [ego_data.focal] + [n for n in ego_data.nodes if n != ego_data.focal]
    idx = {nid: i for i, nid in enumerate(nodes)}

    # Build structural matrix from edges in EgoData
    S, existing_edges = build_structural_matrix(ego_data.edges, nodes, idx)

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

    return AnalysisResult(
        nodes=nodes,
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
        coherence=coherence,
        phrase_similarities=phrase_similarities,
        phrase_contributions=phrase_contributions,
        standout_phrases=standout_phrases
    )


def save_analysis_to_json(analysis_result: AnalysisResult, output_dir: Path) -> Path:
    """
    Save analysis results to JSON files (single-graph model).

    Creates:
        - analysis_YYYYMMDD_HHMMSS.json (timestamped snapshot)
        - analysis_latest.json (most recent)

    Args:
        analysis_result: Analysis results to save
        output_dir: Output directory for JSON files

    Returns:
        Path to latest file (analysis_latest.json)
    """
    # Build dict for JSON serialization using existing helper
    analysis_dict = build_analysis_output(
        None,  # graph_name (single-graph model)
        analysis_result.params,
        analysis_result.S,
        analysis_result.A,
        analysis_result.W,
        analysis_result.F,
        analysis_result.D,
        analysis_result.F_MB,
        analysis_result.E_MB,
        analysis_result.clusters,
        analysis_result.suggestions,
        analysis_result.nodes,
        analysis_result.coherence,
        analysis_result.phrase_similarities,
        analysis_result.phrase_contributions,
        analysis_result.standout_phrases
    )

    return write_analysis(analysis_dict, output_dir, None)


if __name__ == "__main__":
    # Parse command-line arguments
    params = AnalysisParams()
    for i, arg in enumerate(sys.argv[1:], start=1):
        if arg == "--alpha" and i + 1 < len(sys.argv):
            params.alpha = float(sys.argv[i + 1])
        elif arg == "--cos-min" and i + 1 < len(sys.argv):
            params.cos_min = float(sys.argv[i + 1])

    # Setup paths
    root = Path(__file__).parent.parent
    ego_dir = root / "data" / "ego_graph"
    out_dir = root / "data" / "analyses"

    # Load ego graph from files
    embedding_service = get_embedding_service()
    ego_data = load_ego_graph(ego_dir, embedding_service)

    # Run analysis
    analysis_result = analyze(ego_data, params, embedding_service)

    # Save to JSON files
    save_analysis_to_json(analysis_result, out_dir)
