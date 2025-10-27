"""
Cluster Detection
----------------

Detect communities in the blended ego graph using greedy modularity maximization.
"""

from __future__ import annotations
from typing import List

import numpy as np
import networkx as nx


def detect_clusters(W: np.ndarray, nodes: List[str]) -> List[List[str]]:
    """
    Detect clusters in blended graph using greedy modularity.

    Args:
        W: (n, n) blended effective weight matrix
        nodes: Ordered list of node IDs

    Returns:
        List of clusters, where each cluster is a sorted list of node IDs
    """
    n = len(nodes)

    # Symmetrize for undirected clustering
    Wu = W + W.T

    # Build networkx graph
    G = nx.Graph()
    for u in nodes:
        G.add_node(u)
    for i in range(n):
        for j in range(i + 1, n):
            w = float(Wu[i, j])
            if w > 0:
                G.add_edge(nodes[i], nodes[j], weight=w)

    # Detect communities
    if G.number_of_edges() == 0:
        clusters = [sorted(nodes)]
    else:
        comms = nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
        clusters = [sorted(list(c)) for c in comms]

    return clusters
