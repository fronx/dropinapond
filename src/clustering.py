# clustering.py
"""
Clustering algorithms for ego graph neighbors.

Separates clustering logic from the main ego_ops module.
"""
from typing import List, Dict, Optional
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from scipy.stats import entropy as shannon_entropy


def ego_clusters(G: nx.Graph, focal: str, method: str = "greedy") -> List[set]:
    """
    Cluster neighbors of F inside the ego graph (exclude F itself).
    Returns a list of sets (node IDs) for clusters among N(F).

    Args:
        G: NetworkX graph containing the ego network
        focal: ID of the focal node
        method: Clustering method to use ("greedy" or "kmeans")

    Returns:
        List of sets, where each set contains node IDs belonging to that cluster
    """
    H = nx.Graph()
    nbrs = [n for n in G.neighbors(focal)]
    H.add_nodes_from(nbrs)
    for u in nbrs:
        for v in nbrs:
            if u < v and G.has_edge(u, v):
                H.add_edge(u, v, **G.get_edge_data(u, v))
    if H.number_of_edges() == 0 or H.number_of_nodes() <= 2:
        return [set(nbrs)] if nbrs else []
    if method == "greedy":
        comms = nx.algorithms.community.greedy_modularity_communities(H, weight="weight")
        return [set(c) for c in comms]
    else:
        # fallback: 2-means on embeddings of neighbors using graph degree as a tie-breaker
        # (only used if you switch method)
        raise NotImplementedError


def jaccard_overlap(G: nx.Graph, focal: str, j: str) -> float:
    """
    overlap_{Fj} = |N(F) ∩ N(j)| / |N(F) ∪ N(j)| on the ego set.

    Measures structural overlap between focal node's neighborhood
    and neighbor j's neighborhood.
    """
    NF = set(G.neighbors(focal))
    Nj = set(G.neighbors(j))
    inter = len(NF & Nj)
    uni = len(NF | Nj)
    return 0.0 if uni == 0 else inter / uni


def tie_weight_entropy(G: nx.Graph, focal: str, clusters: List[set]) -> float:
    """
    Entropy of attention/tie weights from F to clusters.

    Measures how evenly distributed your attention is across clusters.
    Higher entropy = more balanced attention across multiple clusters.
    Lower entropy = attention concentrated in fewer clusters.

    Args:
        G: NetworkX graph
        focal: ID of the focal node
        clusters: List of cluster sets

    Returns:
        Shannon entropy in bits
    """
    weights = []
    for C in clusters:
        wC = 0.0
        for j in C:
            if G.has_edge(focal, j):
                wC += G[focal][j].get("weight", 1.0)
        weights.append(wC)
    w = np.array(weights, dtype=float)
    if w.sum() <= 1e-12:
        return 0.0
    p = w / w.sum()
    return float(shannon_entropy(p, base=2))
