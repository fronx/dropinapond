"""
Debug pairwise semantic affinity between two nodes.

Usage:
    uv run src/debug_affinity_pair.py <graph_name> <node_a> <node_b> [--cos-min 0.2]
"""

from __future__ import annotations
import sys
import numpy as np
from pathlib import Path

# import functions from your analysis code
from embeddings import get_embedding_service
from semantic_flow import _phrase_matrix, _semantic_affinity, _weighted_mean_embedding

def main(graph_name: str, node_a: str, node_b: str, cos_min: float = 0.2):
    emb = get_embedding_service()

    Ea, wa = _phrase_matrix(emb, graph_name, node_a)
    Eb, wb = _phrase_matrix(emb, graph_name, node_b)

    if Ea.size == 0 or Eb.size == 0:
        print(f"[warn] Missing embeddings for {node_a} or {node_b}")
        return

    # --- phrase-level inspection
    S = Ea @ Eb.T
    W = np.outer(wa, wb)
    mask = S >= cos_min

    print(f"\nPairwise cosine matrix (rows={node_a} phrases, cols={node_b} phrases):")
    np.set_printoptions(precision=3, suppress=True)
    print(S)

    print(f"\nAbove threshold ({cos_min}): {mask.sum()} / {mask.size}")
    if mask.sum() > 0:
        print("\nTop contributing pairs (cosine, weights):")
        indices = np.argwhere(mask)
        scores = [(S[i, j], wa[i], wb[j]) for i, j in indices]
        for val, wa_i, wb_j in sorted(scores, key=lambda x: -x[0])[:10]:
            print(f"  cos={val:.3f}, wa={wa_i:.2f}, wb={wb_j:.2f}")

    # --- aggregate affinity via your normal formula
    aff = _semantic_affinity(Ea, wa, Eb, wb, cos_min=cos_min)
    print(f"\nAggregate _semantic_affinity = {aff:.4f}")

    # --- mean-embedding cosine (less strict)
    ma = _weighted_mean_embedding(Ea, wa)
    mb = _weighted_mean_embedding(Eb, wb)
    cos_mean = np.dot(ma, mb)
    print(f"Mean-embedding cosine = {cos_mean:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: uv run src/debug_affinity_pair.py <graph_name> <node_a> <node_b> [--cos-min 0.2]")
        sys.exit(1)

    graph = sys.argv[1]
    a = sys.argv[2]
    b = sys.argv[3]

    # default
    cos_min = 0.2
    # look for flag form
    if "--cos-min" in sys.argv:
        try:
            idx = sys.argv.index("--cos-min")
            cos_min = float(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("[warn] --cos-min flag provided but no numeric value found; using default 0.2")

    main(graph, a, b, cos_min)