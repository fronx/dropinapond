"""
Semantic-Structural Flow Analysis - Modular Components
------------------------------------------------------

Blend factual topology (edges.json) with semantic affinity derived from phrase
embeddings in ChromaDB, then compute various fields and metrics.
"""

from .structural import build_structural_matrix
from .semantic import load_phrase_data, compute_semantic_affinity_matrix
from .blending import blend_matrices, compute_fields
from .clustering import detect_clusters
from .suggestions import generate_suggestions
from .serialize import build_analysis_output, write_analysis

__all__ = [
    "build_structural_matrix",
    "load_phrase_data",
    "compute_semantic_affinity_matrix",
    "blend_matrices",
    "compute_fields",
    "detect_clusters",
    "generate_suggestions",
    "build_analysis_output",
    "write_analysis",
]
