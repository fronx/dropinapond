# clustering.py
"""
Clustering algorithms for ego graph neighbors.

Separates clustering logic from the main ego_ops module.
Includes both discrete clustering and continuous kernel-based neighborhoods.
"""
from typing import List, Dict, Optional, TYPE_CHECKING
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from scipy.stats import entropy as shannon_entropy

if TYPE_CHECKING:
    from storage import EgoData


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


# ---------------------------
# Kernel-based neighborhoods (continuous, overlapping)
# ---------------------------

def gaussian_kernel(distance: float, bandwidth: float) -> float:
    """
    Gaussian (RBF) kernel: exp(-distance^2 / (2 * bandwidth^2))

    Args:
        distance: Euclidean distance in embedding space
        bandwidth: Kernel bandwidth (controls smoothness)

    Returns:
        Similarity value in [0, 1]
    """
    return float(np.exp(-(distance ** 2) / (2 * bandwidth ** 2)))


def compute_kernel_neighborhoods(
    ego_data: 'EgoData',
    bandwidth: Optional[float] = None,
    kernel: str = 'gaussian'
) -> Dict[str, Dict[str, float]]:
    """
    Compute continuous, overlapping neighborhoods using kernel similarity.

    Instead of hard cluster assignments, each neighbor gets a continuous
    weight indicating how much they belong to each other neighbor's
    "neighborhood" in semantic space.

    Args:
        ego_data: EgoData object with embeddings
        bandwidth: Kernel bandwidth. If None, uses median distance heuristic
        kernel: Kernel type ('gaussian' only for now)

    Returns:
        Dict mapping each neighbor to their kernel weights to all other neighbors
        Format: {neighbor_id: {other_id: similarity_weight, ...}}
    """
    from storage import EgoData  # Import here to avoid circular dependency

    focal = ego_data.focal
    neighbors = [n for n in ego_data.nodes if n != focal]

    if len(neighbors) == 0:
        return {}

    # Get embeddings for all neighbors
    embeddings = {n: ego_data.embeddings[n] for n in neighbors}

    # Compute pairwise distances
    distances = {}
    distance_list = []
    for i, n1 in enumerate(neighbors):
        for n2 in neighbors[i:]:
            if n1 == n2:
                distances[(n1, n2)] = 0.0
            else:
                dist = float(np.linalg.norm(embeddings[n1] - embeddings[n2]))
                distances[(n1, n2)] = dist
                distances[(n2, n1)] = dist
                distance_list.append(dist)

    # Auto-select bandwidth using median distance if not specified
    if bandwidth is None:
        if len(distance_list) > 0:
            bandwidth = float(np.median(distance_list))
        else:
            bandwidth = 1.0

    # Compute kernel similarities
    kernel_weights = {}
    for n1 in neighbors:
        weights = {}
        for n2 in neighbors:
            if n1 == n2:
                weights[n2] = 1.0  # Perfect self-similarity
            else:
                dist = distances[(n1, n2)]
                if kernel == 'gaussian':
                    weights[n2] = gaussian_kernel(dist, bandwidth)
                else:
                    raise ValueError(f"Unknown kernel type: {kernel}")
        kernel_weights[n1] = weights

    return kernel_weights


def kernel_neighborhood_entropy(
    ego_data: 'EgoData',
    kernel_weights: Dict[str, Dict[str, float]],
    tie_weights: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Compute entropy of each neighbor's kernel-based neighborhood.

    High entropy = evenly connected to many neighbors (bridge/hub)
    Low entropy = strongly connected to a few neighbors (cluster core)

    Args:
        ego_data: EgoData object
        kernel_weights: Output from compute_kernel_neighborhoods
        tie_weights: Optional edge weights from focal to neighbors

    Returns:
        Dict mapping neighbor_id to their neighborhood entropy
    """
    entropies = {}

    for neighbor, weights_dict in kernel_weights.items():
        # Get weights to other neighbors (excluding self)
        other_weights = [w for n, w in weights_dict.items() if n != neighbor]

        if len(other_weights) == 0:
            entropies[neighbor] = 0.0
            continue

        # Optionally weight by tie strength from focal
        if tie_weights and neighbor in tie_weights:
            # Scale by how much attention focal gives to this neighbor
            other_weights = [w * tie_weights[neighbor] for w in other_weights]

        w = np.array(other_weights, dtype=float)
        if w.sum() <= 1e-12:
            entropies[neighbor] = 0.0
        else:
            p = w / w.sum()
            entropies[neighbor] = float(shannon_entropy(p, base=2))

    return entropies


def identify_bridge_nodes(
    kernel_weights: Dict[str, Dict[str, float]],
    threshold: float = 0.3
) -> List[str]:
    """
    Identify bridge nodes that connect multiple semantic regions.

    Bridge nodes have high kernel similarity to neighbors that are
    themselves dissimilar to each other.

    Args:
        kernel_weights: Output from compute_kernel_neighborhoods
        threshold: Minimum kernel weight to consider a connection

    Returns:
        List of node IDs that appear to be bridges, sorted by bridge score
    """
    bridge_scores = {}

    for node, weights in kernel_weights.items():
        # Get nodes strongly connected to this one
        strong_connections = [n for n, w in weights.items() if w >= threshold and n != node]

        if len(strong_connections) < 2:
            bridge_scores[node] = 0.0
            continue

        # Measure how dissimilar the strong connections are to each other
        dissimilarity_sum = 0.0
        count = 0
        for i, n1 in enumerate(strong_connections):
            for n2 in strong_connections[i+1:]:
                # How similar are n1 and n2 to each other?
                similarity = kernel_weights.get(n1, {}).get(n2, 0.0)
                dissimilarity_sum += (1.0 - similarity)
                count += 1

        # Bridge score = average dissimilarity of connected neighbors
        bridge_scores[node] = dissimilarity_sum / count if count > 0 else 0.0

    # Return sorted by bridge score
    bridges = sorted(bridge_scores.items(), key=lambda x: x[1], reverse=True)
    return [node for node, score in bridges if score > 0.0]
