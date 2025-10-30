"""
Semantic-Structural Flow Analysis - Modular Components
------------------------------------------------------

Blend factual topology (edges.json) with semantic affinity derived from phrase
embeddings in ChromaDB, then compute various fields and metrics.

This package provides the modular components. For the main orchestrator with
analyze(), AnalysisParams, etc., import from the parent-level semantic_flow module.
"""

from .structural import build_structural_matrix
from .semantic import load_phrase_data, compute_semantic_affinity_matrix, compute_phrase_contribution_breakdown, compute_standout_phrases
from .blending import blend_matrices, compute_fields
from .clustering import detect_clusters
from .suggestions import generate_suggestions
from .phrase_similarities import compute_all_phrase_similarities
from .serialize import build_analysis_output, write_analysis

__all__ = [
    "build_structural_matrix",
    "load_phrase_data",
    "compute_semantic_affinity_matrix",
    "compute_phrase_contribution_breakdown",
    "compute_standout_phrases",
    "blend_matrices",
    "compute_fields",
    "detect_clusters",
    "generate_suggestions",
    "compute_all_phrase_similarities",
    "build_analysis_output",
    "write_analysis",
]
