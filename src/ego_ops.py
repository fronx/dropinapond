# ego_ops.py
"""
Navigation metrics and analysis for ego graphs.

For data loading, see storage.py.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Iterable, Tuple, Optional
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import networkx as nx
from scipy.linalg import expm
import matplotlib.pyplot as plt

# Import EgoData and loading function from storage
from storage import EgoData, load_ego_graph
# Import clustering functions
from clustering import (
    ego_clusters, jaccard_overlap, tie_weight_entropy,
    compute_kernel_neighborhoods, kernel_neighborhood_entropy, identify_bridge_nodes
)

# ---------------------------
# Utilities
# ---------------------------

def _r2_vector(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R^2 for a single vector target (treat dimensions as datapoints).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    num = np.sum((y_true - y_pred) ** 2)
    denom = np.sum((y_true - y_true.mean()) ** 2)
    if denom <= 1e-12:
        return 1.0 if num <= 1e-12 else 0.0
    return 1.0 - (num / denom)

def _cos(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + eps)

# ---------------------------
# 1) Ego picture: overlaps, clustering, attention entropy
# ---------------------------
# (Clustering functions now in clustering.py)

# ---------------------------
# 2) Public legibility (neighbors -> you)
# ---------------------------

def public_legibility_r2(ego: EgoData, neighbors: Iterable[str], lam: float = 1e-3) -> float:
    r"""
    Fit: \hat z_F = sum_j alpha_j z_j with ridge on coefficients alpha (L2).
    Closed form: alpha = (Z Z^T + lam I)^{-1} Z z_F, where Z is (n x d).
    """
    F = ego.focal
    zF = ego.embeddings[F]
    nbrs = list(neighbors)
    if len(nbrs) == 0:
        return 0.0
    Z = np.stack([ego.embeddings[j] for j in nbrs], axis=0)  # (n, d)
    ZZt = Z @ Z.T  # (n, n)
    A = ZZt + lam * np.eye(len(nbrs))
    b = Z @ zF  # (n,)
    alpha = np.linalg.solve(A, b)  # (n,)
    z_hat = alpha @ Z  # (d,)
    return _r2_vector(zF, z_hat)

def public_legibility_r2_per_neighbor(ego: EgoData, lam: float = 1e-3) -> Dict[str, float]:
    """
    R^2 when using a single neighbor j to reconstruct z_F (ridge).
    alpha_j closed form reduces to (z_j·z_F)/(||z_j||^2 + lam).
    """
    F = ego.focal
    zF = ego.embeddings[F]
    out = {}
    for j in ego.nodes:
        if j == F:
            continue
        zj = ego.embeddings[j]
        denom = np.dot(zj, zj) + lam
        alpha_j = float(np.dot(zj, zF) / denom)
        z_hat = alpha_j * zj
        out[j] = _r2_vector(zF, z_hat)
    return out

def compute_phrase_similarities(
    embedding_service,
    graph_name: str,
    focal_id: str,
    neighbor_id: str,
    top_k: int = 10
) -> List[Dict]:
    """
    Compute semantic similarity between focal node's phrases and neighbor's phrases.
    Returns top-k most similar phrase pairs based on cosine similarity of embeddings.

    NOTE: This is for contextual understanding, NOT what's used in readability R².
    Readability uses mean embeddings, not individual phrase pairs.

    Args:
        embedding_service: EmbeddingService instance
        graph_name: Name of the ego graph
        focal_id: ID of the focal node
        neighbor_id: ID of the neighbor node
        top_k: Number of top pairs to return

    Returns:
        List of dicts with keys: focal_phrase, neighbor_phrase, similarity, focal_weight, neighbor_weight
    """
    # Get all phrase data for focal node
    focal_data = embedding_service.get_all_node_phrases(graph_name, focal_id)

    # Get all phrase data for neighbor
    neighbor_data = embedding_service.get_all_node_phrases(graph_name, neighbor_id)

    if not focal_data or not neighbor_data:
        return []

    similarities = []

    # Compute all pairwise similarities
    for focal_phrase_id, focal_phrase_data in focal_data.items():
        focal_emb = focal_phrase_data['embedding']
        focal_text = focal_phrase_data['text']
        focal_weight = focal_phrase_data['metadata'].get('weight', 1.0)

        for neighbor_phrase_id, neighbor_phrase_data in neighbor_data.items():
            neighbor_emb = neighbor_phrase_data['embedding']
            neighbor_text = neighbor_phrase_data['text']
            neighbor_weight = neighbor_phrase_data['metadata'].get('weight', 1.0)

            sim = _cos(focal_emb, neighbor_emb)
            if sim > 0.3:  # Only include reasonably similar pairs
                similarities.append({
                    'focal_phrase': focal_text,
                    'neighbor_phrase': neighbor_text,
                    'similarity': float(sim),
                    'focal_weight': focal_weight,
                    'neighbor_weight': neighbor_weight
                })

    # Sort by similarity * combined weights
    similarities.sort(key=lambda x: x['similarity'] * x['focal_weight'] * x['neighbor_weight'], reverse=True)

    return similarities[:top_k]

def compute_readability_explanation(
    ego: EgoData,
    embedding_service,
    graph_name: str,
    neighbor_id: str,
    lam: float = 1e-3
) -> Dict:
    """
    Compute detailed explanation of readability (R²) for a specific neighbor.
    Shows what ACTUALLY goes into the ridge regression computation.

    Returns:
        Dict with:
        - r2: The R² score
        - alpha: The ridge coefficient
        - focal_mean_contributors: Phrases contributing most to focal's mean embedding
        - neighbor_mean_contributors: Phrases contributing most to neighbor's mean embedding
        - reconstruction_quality: How well neighbor reconstructs each focal phrase
    """
    F = ego.focal
    zF = ego.embeddings[F]
    zj = ego.embeddings[neighbor_id]

    # Compute readability
    denom = np.dot(zj, zj) + lam
    alpha_j = float(np.dot(zj, zF) / denom)
    z_hat = alpha_j * zj
    r2 = _r2_vector(zF, z_hat)

    # Get phrase data
    focal_data = embedding_service.get_all_node_phrases(graph_name, F)
    neighbor_data = embedding_service.get_all_node_phrases(graph_name, neighbor_id)

    # Compute contribution of each focal phrase to the mean
    focal_phrases_data = []
    focal_weights_sum = sum(p['metadata'].get('weight', 1.0) for p in focal_data.values())

    for phrase_id, phrase_data in focal_data.items():
        weight = phrase_data['metadata'].get('weight', 1.0)
        normalized_weight = weight / focal_weights_sum
        # Contribution = how much this phrase's embedding contributes to mean (weighted)
        contribution = normalized_weight
        focal_phrases_data.append({
            'phrase': phrase_data['text'],
            'weight': weight,
            'contribution_to_mean': contribution
        })

    # Sort by contribution
    focal_phrases_data.sort(key=lambda x: x['contribution_to_mean'], reverse=True)

    # Same for neighbor
    neighbor_phrases_data = []
    neighbor_weights_sum = sum(p['metadata'].get('weight', 1.0) for p in neighbor_data.values())

    for phrase_id, phrase_data in neighbor_data.items():
        weight = phrase_data['metadata'].get('weight', 1.0)
        normalized_weight = weight / neighbor_weights_sum
        neighbor_phrases_data.append({
            'phrase': phrase_data['text'],
            'weight': weight,
            'contribution_to_mean': normalized_weight
        })

    neighbor_phrases_data.sort(key=lambda x: x['contribution_to_mean'], reverse=True)

    # Compute how well the reconstruction works for each focal phrase
    reconstruction_errors = []
    for phrase_id, phrase_data in focal_data.items():
        phrase_emb = phrase_data['embedding']
        # Project the reconstruction onto this phrase
        reconstruction_fit = _cos(z_hat, phrase_emb)
        actual_fit = _cos(zF, phrase_emb)
        reconstruction_errors.append({
            'phrase': phrase_data['text'],
            'actual_alignment': float(actual_fit),
            'reconstructed_alignment': float(reconstruction_fit),
            'error': float(abs(actual_fit - reconstruction_fit))
        })

    # Sort by error (worst reconstructed first)
    reconstruction_errors.sort(key=lambda x: x['error'], reverse=True)

    return {
        'r2': float(r2),
        'alpha': alpha_j,
        'focal_mean_contributors': focal_phrases_data[:10],
        'neighbor_mean_contributors': neighbor_phrases_data[:10],
        'reconstruction_quality': reconstruction_errors[:10]
    }

# ---------------------------
# 3) Subjective attunement (you -> neighbors)
# ---------------------------

def subjective_attunement_r2(ego: EgoData, cluster: Iterable[str], lam: float = 1e-3) -> float:
    """
    For each neighbor j in cluster, fit a scalar beta_j s.t. beta_j z_F ~ z_j (ridge on beta).
    beta_j = (z_F·z_j)/(||z_F||^2 + lam).
    Aggregate into a single R^2 over all neighbors and dimensions.
    """
    F = ego.focal
    zF = ego.embeddings[F]
    denom = np.dot(zF, zF) + lam
    C = list(cluster)
    if len(C) == 0:
        return 0.0
    # Ground truth matrix (n, d) and predictions (n, d)
    Ztrue = np.stack([ego.embeddings[j] for j in C], axis=0)
    betas = [float(np.dot(zF, ego.embeddings[j]) / denom) for j in C]
    Zpred = np.stack([b * zF for b in betas], axis=0)
    # Vectorized R^2 over n*d entries
    y_true = Ztrue.flatten()
    y_pred = Zpred.flatten()
    num = np.sum((y_true - y_pred) ** 2)
    denom_all = np.sum((y_true - y_true.mean()) ** 2)
    if denom_all <= 1e-12:
        return 1.0 if num <= 1e-12 else 0.0
    return float(1.0 - num / denom_all)

# ---------------------------
# 4) Heat-residual (novelty vs pocket at small t)
# ---------------------------

def heat_residual_norm(
    ego: EgoData,
    pocket: Iterable[str],
    t: float = 0.1,
    use_normalized: bool = True
) -> float:
    """
    s_F^{(k)}(t) = || z_F - (e^{-t L} Z)_F || where L is the (normalized) Laplacian
    on the subgraph induced by {F} ∪ pocket.
    """
    F = ego.focal
    nodes = [F] + [n for n in pocket if n != F]
    idx = {n: i for i, n in enumerate(nodes)}
    # Build adjacency
    G = ego.graph().subgraph(nodes).copy()
    n = len(nodes)
    A = np.zeros((n, n))
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1.0)
        i, j = idx[u], idx[v]
        A[i, j] = A[j, i] = w
    dvec = A.sum(axis=1)
    if use_normalized:
        with np.errstate(divide='ignore'):
            invsqrt = np.where(dvec > 0, 1.0 / np.sqrt(dvec), 0.0)
        L = np.eye(n) - (invsqrt[:, None] * A * invsqrt[None, :])
    else:
        L = np.diag(dvec) - A
    K = expm(-t * L)  # heat kernel
    Z = ego.Z(nodes)  # (n, d)
    HZ = K @ Z
    zF = ego.embeddings[F]
    return float(np.linalg.norm(zF - HZ[idx[F]]))

# ---------------------------
# 5) Translation vectors and query shaping
# ---------------------------

def pocket_centroid(ego: EgoData, pocket: Iterable[str]) -> np.ndarray:
    X = np.stack([ego.embeddings[n] for n in pocket], axis=0)
    return X.mean(axis=0)

def translate_query(q: np.ndarray, tvec: np.ndarray, alpha: float) -> np.ndarray:
    """
    q_k = normalize(q + alpha * t_k)
    """
    return _normalize(q + alpha * tvec)

# ---------------------------
# 6) Orientation score
# ---------------------------

@dataclass
class OrientationWeights:
    lam1: float = 1.0  # low-overlap benefit
    lam2: float = 1.0  # public legibility
    lam3: float = 1.0  # subjective attunement to cluster
    lam4: float = 1.0  # topical match after translation
    lam5: float = 0.5  # penalty for instability

def orientation_scores(
    ego: EgoData,
    clusters: List[set],
    target_pocket_by_node: Dict[str, int],
    q_base: np.ndarray,
    home_pocket_idx: int,
    alpha: float = 0.2,
    instabilities: Optional[Dict[str, float]] = None,
    weights: OrientationWeights = OrientationWeights(),
    ridge_lam: float = 1e-3
) -> Dict[str, float]:
    """
    Compute per-neighbor Score(j) using only ego info.
    target_pocket_by_node: mapping node -> cluster index it belongs to.

    NOTE: This function now delegates to orientation_score_breakdowns() to ensure
    the scores and their explanations are always computed using the same logic.
    """
    breakdowns = orientation_score_breakdowns(
        ego=ego,
        clusters=clusters,
        target_pocket_by_node=target_pocket_by_node,
        q_base=q_base,
        home_pocket_idx=home_pocket_idx,
        alpha=alpha,
        instabilities=instabilities,
        weights=weights,
        ridge_lam=ridge_lam
    )
    # Extract just the total scores
    return {node_id: breakdown["total_score"] for node_id, breakdown in breakdowns.items()}

def orientation_score_breakdowns(
    ego: EgoData,
    clusters: List[set],
    target_pocket_by_node: Dict[str, int],
    q_base: np.ndarray,
    home_pocket_idx: int,
    alpha: float = 0.2,
    instabilities: Optional[Dict[str, float]] = None,
    weights: OrientationWeights = OrientationWeights(),
    ridge_lam: float = 1e-3
) -> Dict[str, Dict]:
    """
    Compute detailed breakdowns of orientation scores showing each component's contribution.
    Returns a dict mapping each neighbor to a breakdown dict with:
      - total_score: final orientation score
      - weights: the lambda weights used
      - components: dict with each of the 5 components showing:
          - raw_value: the metric value before weighting
          - weighted_contribution: the value after applying lambda weight
          - metadata: additional context about how it was computed
    """
    G = ego.graph()
    F = ego.focal

    # Precompute all intermediate values (same as orientation_scores)
    overlaps = {j: jaccard_overlap(G, F, j) for j in G.neighbors(F)}
    one_minus_overlap = {j: 1.0 - ov for j, ov in overlaps.items()}

    r2_in_per_neighbor = public_legibility_r2_per_neighbor(ego, lam=ridge_lam)

    r2_out_by_pocket = {
        k: subjective_attunement_r2(ego, clusters[k], lam=ridge_lam)
        for k in range(len(clusters))
    }

    mu_home = pocket_centroid(ego, clusters[home_pocket_idx])
    t_by_pocket = {k: pocket_centroid(ego, clusters[k]) - mu_home for k in range(len(clusters))}
    qk_by_pocket = {k: translate_query(q_base, t_by_pocket[k], alpha=alpha) for k in t_by_pocket}

    inst = instabilities or {}

    # Build cluster names for display
    cluster_names = {}
    for k, cluster in enumerate(clusters):
        if ego.names:
            member_names = [str(ego.names.get(node_id, node_id)) for node_id in sorted(cluster)]
        else:
            member_names = [str(node_id) for node_id in sorted(cluster)]
        cluster_names[k] = ", ".join(member_names)

    # Compute breakdowns for each neighbor
    breakdowns: Dict[str, Dict] = {}
    for j in G.neighbors(F):
        k = target_pocket_by_node.get(j, None)

        # Component 1: Network exploration (1 - overlap)
        exploration_raw = one_minus_overlap[j]
        exploration_contrib = weights.lam1 * exploration_raw

        # Component 2: Readability (R²_in - they get you)
        readability_raw = r2_in_per_neighbor.get(j, 0.0)
        readability_contrib = weights.lam2 * readability_raw

        # Component 3: Subjective attunement (R²_out - you get their cluster)
        if k is not None:
            attunement_raw = r2_out_by_pocket.get(k, 0.0)
            attunement_contrib = weights.lam3 * attunement_raw
            target_cluster_name = cluster_names.get(k, f"Cluster {k}")
        else:
            attunement_raw = 0.0
            attunement_contrib = 0.0
            target_cluster_name = None

        # Component 4: Topical relevance (cosine after translation)
        if k is not None:
            cos_term = _cos(qk_by_pocket[k], ego.embeddings[j])
            relevance_raw = cos_term
            relevance_contrib = weights.lam4 * cos_term
            translation_magnitude = float(np.linalg.norm(t_by_pocket[k]))
        else:
            relevance_raw = 0.0
            relevance_contrib = 0.0
            translation_magnitude = 0.0

        # Component 5: Instability penalty
        instability_raw = inst.get(j, 0.0)
        instability_contrib = -weights.lam5 * instability_raw

        # Total score
        total = (exploration_contrib + readability_contrib + attunement_contrib +
                relevance_contrib + instability_contrib)

        breakdowns[j] = {
            "total_score": float(total),
            "weights": {
                "lambda1_exploration": weights.lam1,
                "lambda2_readability": weights.lam2,
                "lambda3_attunement": weights.lam3,
                "lambda4_relevance": weights.lam4,
                "lambda5_instability": weights.lam5,
            },
            "components": {
                "exploration": {
                    "raw_value": float(exploration_raw),
                    "weighted_contribution": float(exploration_contrib),
                    "metadata": {
                        "overlap": float(overlaps[j]),
                        "one_minus_overlap": float(exploration_raw),
                        "description": "Network exploration benefit (low overlap = high novelty)"
                    }
                },
                "readability": {
                    "raw_value": float(readability_raw),
                    "weighted_contribution": float(readability_contrib),
                    "metadata": {
                        "r2_in": float(readability_raw),
                        "description": "How well they can predict your interests (public legibility)"
                    }
                },
                "attunement": {
                    "raw_value": float(attunement_raw),
                    "weighted_contribution": float(attunement_contrib),
                    "metadata": {
                        "r2_out": float(attunement_raw),
                        "target_cluster_name": target_cluster_name,
                        "target_cluster_index": k,
                        "description": "How well you understand their cluster's interests"
                    }
                },
                "relevance": {
                    "raw_value": float(relevance_raw),
                    "weighted_contribution": float(relevance_contrib),
                    "metadata": {
                        "cosine_after_translation": float(relevance_raw),
                        "home_cluster_name": cluster_names.get(home_pocket_idx, f"Cluster {home_pocket_idx}"),
                        "home_cluster_index": home_pocket_idx,
                        "target_cluster_name": target_cluster_name,
                        "translation_vector_magnitude": float(translation_magnitude),
                        "description": "Semantic alignment after translating from your cluster to theirs"
                    }
                },
                "instability": {
                    "raw_value": float(instability_raw),
                    "weighted_contribution": float(instability_contrib),
                    "metadata": {
                        "instability_score": float(instability_raw),
                        "description": "Penalty for relationship instability (negative = stable)"
                    }
                }
            }
        }

    return breakdowns

def cluster_residual_direction(ego: EgoData, cluster, lam: float = 1e-3):
    F = ego.focal
    zF = ego.embeddings[F]
    denom = np.dot(zF, zF) + lam
    Rs = []
    for j in cluster:
        zj = ego.embeddings[j]
        beta = float(np.dot(zF, zj) / denom)
        Rs.append(zj - beta * zF)
    R = np.stack(Rs, axis=0)  # (n,d)
    rbar = R.mean(axis=0)
    nrm = np.linalg.norm(rbar)
    t_hat = rbar / (nrm + 1e-12)
    # average residual energy (for gating/novelty)
    avg_residual_norm = float(np.mean(np.sum(R**2, axis=1))**0.5)
    return t_hat, avg_residual_norm

def subjective_attunement_r2_rank2(ego: EgoData, cluster, lam: float = 1e-3):
    # Attunement using span{z_F, t_k} with ridge on the 2 coefficients per neighbor
    F = ego.focal
    zF = ego.embeddings[F]
    t_hat, _ = cluster_residual_direction(ego, cluster, lam=lam)
    # Orthonormalize basis with simple Gram-Schmidt
    u1 = zF / (np.linalg.norm(zF) + 1e-12)
    t_ = t_hat - np.dot(u1, t_hat) * u1
    u2 = t_ / (np.linalg.norm(t_) + 1e-12)
    B = np.stack([u1, u2], axis=1)  # (d,2)
    Ztrue = np.stack([ego.embeddings[j] for j in cluster], axis=0)  # (n,d)
    BtB = B.T @ B  # ~ I, but keep general
    A = BtB + lam * np.eye(2)
    Bt = B.T
    # Fit 2 coeffs per neighbor independently
    coeffs = (np.linalg.solve(A, Bt @ Ztrue.T)).T  # (n,2)
    Zpred = (B @ coeffs.T).T  # (n,d)
    y_true = Ztrue.flatten()
    y_pred = Zpred.flatten()
    num = np.sum((y_true - y_pred) ** 2)
    denom_all = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - (num / denom_all) if denom_all > 1e-12 else (1.0 if num <= 1e-12 else 0.0)

def gated_attunement_score(ego, cluster, tau=0.3, lam=1e-3):
    r1 = subjective_attunement_r2(ego, cluster, lam=lam)
    if r1 >= tau:
        return r1
    r2 = subjective_attunement_r2_rank2(ego, cluster, lam=lam)
    # Only trust the richer model if it materially improves and exceeds gate
    if r2 - r1 >= 0.1 and r2 >= tau:
        return r2
    # Treat as novelty: return 0 so orientation leans on overlap/heat/cosine-after-translation
    return 0.0

def diffusion_distance_to_pocket(ego, pocket, t=0.25, eps=1e-12):
    G = ego.graph()
    nodes = list(ego.nodes)
    idx = {n:i for i,n in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n,n))
    for u,v,d in G.edges(data=True):
        i,j = idx[u], idx[v]
        w = d.get("weight",1.0)
        A[i,j]=A[j,i]=w
    dvec = A.sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        Dinv = np.where(dvec>0, 1.0/dvec, 0.0)
    # random-walk Laplacian
    Lrw = np.eye(n) - (Dinv[:,None]*A)
    K = expm(-t*Lrw)                 # heat kernel
    pi = dvec/ (dvec.sum()+eps)      # stationary weights
    rF = K[idx[ego.focal]]
    rP = np.mean([K[idx[j]] for j in pocket], axis=0)
    diff = rF - rP
    D2 = np.sum((diff**2)/(pi+eps))
    return float(np.sqrt(D2))

# ---------------------------
# Analysis JSON export
# ---------------------------

def save_analysis_json(
    ego_graph_path: Path,
    overlaps: Dict[str, float],
    clusters: List[set],
    attention_entropy: float,
    r2_in_all: float,
    r2_in_per_cluster: Dict[int, float],
    r2_out: Dict[int, float],
    s_t: Dict[int, float],
    r2_in_per_neighbor: Dict[str, float],
    scores: Dict[str, float],
    phrase_similarities: Optional[Dict[str, List[Dict]]] = None,
    orientation_score_breakdowns: Optional[Dict[str, Dict]] = None,
    kernel_neighborhoods: Optional[Dict[str, Dict]] = None,
    output_dir: Optional[Path] = None,
    ego_data: Optional['EgoData'] = None
) -> Path:
    """
    Save analysis results to JSON file.

    Args:
        ego_graph_path: Path to the original ego graph JSON file
        overlaps: Dict of node_id -> overlap score
        clusters: List of cluster sets
        attention_entropy: Entropy of attention distribution
        r2_in_all: Overall public legibility R^2
        r2_in_per_cluster: Per-cluster public legibility R^2
        r2_out: Per-cluster subjective attunement R^2
        s_t: Per-cluster heat-residual novelty
        r2_in_per_neighbor: Per-neighbor readability R^2
        scores: Orientation scores per neighbor
        phrase_similarities: Per-neighbor phrase-level semantic overlaps
        output_dir: Directory to save analysis (defaults to data/analyses/)

    Returns:
        Path to saved JSON file
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "analyses"
    output_dir.mkdir(parents=True, exist_ok=True)

    ego_graph_path = Path(ego_graph_path)

    # Load ego graph version (handle both modular directory and monolithic file)
    if ego_graph_path.is_dir():
        # Modular format - read from metadata.json
        with open(ego_graph_path / "metadata.json") as f:
            metadata = json.load(f)
        ego_graph_version = metadata.get("version", "unknown")
        ego_graph_name = ego_graph_path.name
    else:
        # Monolithic format - read from the JSON file
        with open(ego_graph_path) as f:
            ego_graph_data = json.load(f)
        ego_graph_version = ego_graph_data.get("version", "unknown")
        ego_graph_name = ego_graph_path.stem

    # Convert clusters from sets to sorted lists for JSON serialization
    clusters_serializable = [sorted(list(c)) for c in clusters]

    # Helper to get node name (fallback to ID if no name)
    def get_node_name(node_id):
        if ego_data and ego_data.names and node_id in ego_data.names:
            return ego_data.names[node_id]
        return node_id

    # Convert cluster-keyed dicts to use tuple-of-names as keys (more readable)
    def cluster_dict_to_serializable(cluster_dict):
        return {
            ", ".join([get_node_name(node_id) for node_id in sorted(list(clusters[k]))]): v
            for k, v in cluster_dict.items()
        }

    # Sort recommendations by score (descending)
    recommendations = [
        {"node_id": node_id, "score": score}
        for node_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]

    analysis = {
        "version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "ego_graph_file": str(ego_graph_path.name),
        "ego_graph_version": ego_graph_version,
        "metrics": {
            "overlaps": overlaps,
            "clusters": clusters_serializable,
            "attention_entropy": attention_entropy,
            "public_legibility_overall": r2_in_all,
            "public_legibility_per_cluster": cluster_dict_to_serializable(r2_in_per_cluster),
            "subjective_attunement_per_cluster": cluster_dict_to_serializable(r2_out),
            "heat_residual_novelty": cluster_dict_to_serializable(s_t),
            "per_neighbor_readability": r2_in_per_neighbor,
            "orientation_scores": scores,
            "orientation_score_breakdowns": orientation_score_breakdowns or {},
            "phrase_similarities": phrase_similarities or {},
            "kernel_neighborhoods": kernel_neighborhoods or {}
        },
        "recommendations": recommendations
    }

    # Generate filename with timestamp (ego_graph_name already set above)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{ego_graph_name}_{timestamp_str}.json"

    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    # Also save as _latest.json for easy web access
    latest_path = output_dir / f"{ego_graph_name}_latest.json"
    with open(latest_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    return output_path

if __name__ == "__main__":
    # Get filename from command line
    if len(sys.argv) < 2:
        print("Usage: python src/ego_ops.py <name>")
        print("Example: python src/ego_ops.py fronx")
        sys.exit(1)

    name = sys.argv[1]
    ego_graphs_dir = Path(__file__).parent.parent / "data" / "ego_graphs"

    # Check for modular directory format
    modular_path = ego_graphs_dir / name

    # Import and initialize embedding service
    from embeddings import get_embedding_service
    embedding_service = get_embedding_service()

    # Load from modular directory
    if not modular_path.exists() or not modular_path.is_dir():
        print(f"Error: Could not find ego graph directory at '{modular_path}'")
        print(f"Expected modular format: data/ego_graphs/{name}/")
        sys.exit(1)

    ego = load_ego_graph(modular_path, embedding_service)
    fixture_path = modular_path

    print(f"Loaded ego graph from: {fixture_path}\n")

    G = ego.graph()
    F = ego.focal
    neighbors = list(G.neighbors(F))

    # Helper to format node as "Name (id)" or just "id" if no name
    def format_node(node_id):
        if ego.names and node_id in ego.names:
            return f"{ego.names[node_id]} ({node_id})"
        return node_id

    # 1) Ego picture
    overlaps = {j: jaccard_overlap(G, F, j) for j in neighbors}
    clusters = ego_clusters(G, F)
    # Entropy of attention across clusters
    H = tie_weight_entropy(G, F, clusters)

    # Map node -> cluster index
    pocket_by_node = {}
    for k, C in enumerate(clusters):
        for n in C:
            pocket_by_node[n] = k

    # 2) Public legibility
    r2_in_all = public_legibility_r2(ego, neighbors, lam=1e-3)
    r2_in_per_cluster = {
        k: public_legibility_r2(ego, clusters[k], lam=1e-3)
        for k in range(len(clusters))
    }
    r2_in_per_neighbor = public_legibility_r2_per_neighbor(ego, lam=1e-3)

    # 3) Subjective attunement
    r2_out = {k: subjective_attunement_r2(ego, clusters[k], lam=1e-3) for k in range(len(clusters))}

    # 4) Heat-residual novelty vs each pocket
    s_t = {k: heat_residual_norm(ego, clusters[k], t=0.12, use_normalized=True) for k in range(len(clusters))}

    # 5) Translation vectors and query shaping
    # Assume home pocket = one containing most of F's tie-weight (choose by max cosine weight sum)
    tie_weight_by_pocket = {
        k: sum(G[F][j].get("weight", 1.0) for j in clusters[k] if G.has_edge(F, j))
        for k in range(len(clusters))
    }
    home_idx = max(tie_weight_by_pocket, key=lambda k: tie_weight_by_pocket[k])
    q_base = ego.embeddings[F]  # start with your own content vector as "message"
    alpha = 0.2  # gentle translation toward a pocket

    # 6) Orientation scores (mild exploration)
    # Compute both scores and their detailed breakdowns
    score_breakdowns = orientation_score_breakdowns(
        ego=ego,
        clusters=clusters,
        target_pocket_by_node=pocket_by_node,
        q_base=q_base,
        home_pocket_idx=home_idx,
        alpha=alpha,
        instabilities={},  # if you have any flakiness prior, inject here
        weights=OrientationWeights(lam1=1.0, lam2=0.8, lam3=0.7, lam4=1.2, lam5=0.5),
        ridge_lam=1e-3
    )
    # Extract just the scores for backward compatibility
    scores = {node_id: breakdown["total_score"] for node_id, breakdown in score_breakdowns.items()}

    # Pretty-print a compact summary
    print("Neighbors:", [format_node(j) for j in neighbors])
    print("\nOverlaps (1 - overlap favors exploration):")
    for j in neighbors:
        print(f"  {format_node(j)}: {overlaps[j]:.2f}  -> (1-overlap)={1-overlaps[j]:.2f}")
    print("\nClusters:", [[format_node(n) for n in sorted(list(C))] for C in clusters])
    print(f"Attention entropy H_F (bits): {H:.3f}")
    print(f"\nPublic legibility R^2_in (all neighbors): {r2_in_all:.3f}")
    print("Per-cluster R^2_in:", {tuple([format_node(n) for n in sorted(list(clusters[k]))]): round(v,3) for k,v in r2_in_per_cluster.items()})
    print("\nSubjective attunement R^2_out per cluster:", {tuple([format_node(n) for n in sorted(list(clusters[k]))]): round(v,3) for k,v in r2_out.items()})
    print("\nHeat-residual novelty s_F(t) per pocket:", {tuple([format_node(n) for n in sorted(list(clusters[k]))]): round(v,3) for k,v in s_t.items()})
    print("\nPer-neighbor readability R^2_in(j):", {format_node(j): round(r2_in_per_neighbor[j],3) for j in neighbors})
    print("\nOrientation scores (higher is better):", {format_node(j): round(scores[j],3) for j in neighbors})
    best = max(scores, key=scores.__getitem__)
    print(f"\nBest next approach: {format_node(best)}")

    # 7) Kernel-based neighborhoods (continuous alternative to discrete clusters)
    print("\n" + "="*70)
    print("KERNEL-BASED NEIGHBORHOODS (continuous semantic regions)")
    print("="*70)
    kernel_weights = compute_kernel_neighborhoods(ego, bandwidth=None)
    kernel_entropies = kernel_neighborhood_entropy(ego, kernel_weights)
    bridge_nodes = identify_bridge_nodes(kernel_weights, threshold=0.3)

    print(f"\nKernel bandwidth (auto-selected): {np.median([w for weights in kernel_weights.values() for w in weights.values() if 0 < w < 1]):.3f}" if kernel_weights else "N/A")

    print(f"\nNeighborhood Entropy (how evenly distributed connections are):")
    print("  High entropy = bridge/hub connecting many regions")
    print("  Low entropy = cluster core, tightly connected to few neighbors")
    sorted_entropies = sorted(kernel_entropies.items(), key=lambda x: x[1], reverse=True)
    for i, (neighbor, entropy) in enumerate(sorted_entropies[:10], 1):
        # Get top 3 strongest kernel connections for this neighbor
        top_connections = sorted(
            [(n, w) for n, w in kernel_weights[neighbor].items() if n != neighbor],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        conn_str = ", ".join([f"{format_node(n).split(' (')[0]} ({w:.2f})" for n, w in top_connections])
        print(f"  {i:2d}. {format_node(neighbor):30s} entropy={entropy:.3f}  top: {conn_str}")

    if bridge_nodes:
        print(f"\nBridge Nodes (connecting semantically distant neighbors):")
        print("  These people connect regions that are otherwise dissimilar")
        for i, bridge in enumerate(bridge_nodes[:8], 1):
            # Get the dissimilar connections this bridge connects
            strong_conns = [(n, kernel_weights[bridge][n]) for n in kernel_weights[bridge]
                          if kernel_weights[bridge][n] >= 0.3 and n != bridge]
            if len(strong_conns) >= 2:
                # Find most dissimilar pair
                max_dissim = 0.0
                dissim_pair = None
                for j, (n1, w1) in enumerate(strong_conns):
                    for n2, w2 in strong_conns[j+1:]:
                        dissim = 1.0 - kernel_weights.get(n1, {}).get(n2, 0.0)
                        if dissim > max_dissim:
                            max_dissim = dissim
                            dissim_pair = (n1, n2)
                if dissim_pair:
                    print(f"  {i}. {format_node(bridge):30s} bridges {format_node(dissim_pair[0]).split(' (')[0]} ↔ {format_node(dissim_pair[1]).split(' (')[0]} (dissim={max_dissim:.2f})")
    print("="*70)

    # 8) Compute phrase-level semantic similarities for each neighbor
    print("\nComputing phrase similarities...")
    phrase_similarities = {}
    for j in neighbors:
        similarities = compute_phrase_similarities(
            embedding_service=embedding_service,
            graph_name=name,
            focal_id=F,
            neighbor_id=j,
            top_k=10
        )
        phrase_similarities[j] = similarities
        if similarities:
            print(f"  {format_node(j)}: {len(similarities)} semantic overlaps (top: {similarities[0]['similarity']:.2f})")

    # Prepare kernel neighborhoods data for JSON export
    kernel_data = {
        "weights": kernel_weights,
        "entropies": kernel_entropies,
        "bridge_nodes": bridge_nodes
    }

    # Save analysis to JSON
    analysis_path = save_analysis_json(
        ego_graph_path=fixture_path,
        overlaps=overlaps,
        clusters=clusters,
        attention_entropy=H,
        r2_in_all=r2_in_all,
        r2_in_per_cluster=r2_in_per_cluster,
        r2_out=r2_out,
        s_t=s_t,
        r2_in_per_neighbor=r2_in_per_neighbor,
        scores=scores,
        phrase_similarities=phrase_similarities,
        orientation_score_breakdowns=score_breakdowns,
        kernel_neighborhoods=kernel_data,
        ego_data=ego
    )
    print(f"\nAnalysis saved to: {analysis_path}")

    # Visualization
    print("\nGenerating visualization...")

    # Define colors for clusters
    cluster_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']

    # Assign colors to nodes based on cluster
    node_colors = {}
    for k, cluster in enumerate(clusters):
        color = cluster_colors[k % len(cluster_colors)]
        for node in cluster:
            node_colors[node] = color

    # Focal node gets special color
    node_colors[F] = '#E74C3C'

    # Create figure
    plt.figure(figsize=(14, 10))

    # Use spring layout for positioning
    pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)

    # Draw edges with varying thickness based on actual weight
    for u, v in G.edges():
        # Use actual weight if it exists, otherwise default to thin line
        if u == F or v == F:
            actual = G[u][v].get('actual', 0.3)
            # Scale thickness VERY dramatically: 0.2 -> 2, 0.9 -> 20
            width = 2.0 + actual * 20.0
            print(f"Edge {format_node(u)} -> {format_node(v)}: actual={actual:.2f}, width={width:.1f}")
        else:
            # Non-focal edges are thinner
            width = 0.5
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                               width=width, alpha=0.6, edge_color='gray')

    # Draw nodes
    for node in G.nodes():
        color = node_colors.get(node, 'lightgray')
        size = 3000 if node == F else 1500
        nx.draw_networkx_nodes(G, pos, nodelist=[node],
                               node_color=color, node_size=size,
                               edgecolors='black', linewidths=2)

    # Draw node labels
    labels = {node: format_node(node).split(' (')[0] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')

    # Draw edge labels with channels and orientation scores
    edge_labels = {}
    for u, v in G.edges():
        parts = []

        # Add channels if edge involves focal node
        if u == F or v == F:
            channels = G[u][v].get('channels', [])
            if channels:
                channel_str = ', '.join(channels)
                parts.append(f"📡 {channel_str}")

            # Add actual strength
            actual = G[u][v].get('actual')
            if actual is not None:
                parts.append(f"💪 {actual:.1f}")

            # Add orientation score for neighbor
            neighbor = v if u == F else u
            if neighbor in scores:
                parts.append(f"🎯 {scores[neighbor]:.2f}")

        if parts:
            edge_labels[(u, v)] = '\n'.join(parts)

    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7,
                                  bbox=dict(boxstyle='round,pad=0.3',
                                           facecolor='white', alpha=0.8))

    # Add legend for clusters
    from matplotlib.patches import Patch
    legend_elements = []
    legend_elements.append(Patch(facecolor='#E74C3C', edgecolor='black', label=f'Focal: {format_node(F)}'))
    for k, cluster in enumerate(clusters):
        color = cluster_colors[k % len(cluster_colors)]
        cluster_names = [format_node(n).split(' (')[0] for n in sorted(cluster)]
        label = f"Cluster {k+1}: {', '.join(cluster_names)}"
        legend_elements.append(Patch(facecolor=color, edgecolor='black', label=label))

    plt.legend(handles=legend_elements, loc='upper left', fontsize=9)
    plt.title(f"Ego Graph: {format_node(F)}\nBest next interaction: {format_node(best)} (score: {scores[best]:.2f})\n\n📡 = channels | 💪 = connection strength | 🎯 = orientation score",
              fontsize=12, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()