# clustering.py
"""
Clustering algorithms for ego graph neighbors.

Separates clustering logic from the main ego_ops module.
Includes both discrete clustering and continuous kernel-based neighborhoods.
"""
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
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

def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize each row of X to unit norm."""
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(nrm, eps)


def _pairwise_cosine_dist(X: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine distance matrix using chordal distance on unit sphere.

    Args:
        X: (n, d) array of embeddings

    Returns:
        D: (n, n) distance matrix in [0, 2]
    """
    Xn = _l2_normalize_rows(X)
    S = Xn @ Xn.T  # cosine similarity in [-1, 1]
    D = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * S))  # chordal distance in [0, 2]
    return D


def _self_tuned_sigmas(D: np.ndarray, k_sig: int = 5) -> np.ndarray:
    """
    Compute self-tuned bandwidth for each node as distance to k-th nearest neighbor.

    Args:
        D: (n, n) pairwise distance matrix
        k_sig: k for k-th nearest neighbor (excluding self)

    Returns:
        sig: (n,) array of per-node bandwidths
    """
    n = D.shape[0]
    # Mask diagonal with large value so it's not selected
    sortd = np.sort(D + np.eye(n) * 1e9, axis=1)
    k = np.clip(k_sig, 1, max(1, n - 1))
    sig = sortd[:, k - 1]  # 0-based index
    # Safety floor
    sig = np.maximum(sig, 1e-6)
    return sig


def _build_anisotropic_kernel(
    D: np.ndarray,
    sig: np.ndarray,
    tau: float = 1.0,
    zero_diag: bool = True
) -> np.ndarray:
    """
    Build anisotropic kernel with temperature control.

    Uses mutual kernel (Zelnik-Manor & Perona): K_ij = exp(-D_ij^2 / (tau^2 * σ_i * σ_j))

    Args:
        D: (n, n) pairwise distance matrix
        sig: (n,) per-node bandwidths
        tau: global temperature multiplier (lower = sharper)
        zero_diag: if True, set diagonal to 0 (no self-loops)

    Returns:
        K: (n, n) kernel matrix
    """
    Sij = (sig[:, None] * sig[None, :]) * (tau ** 2)
    K = np.exp(-(D ** 2) / np.maximum(Sij, 1e-12))
    if zero_diag:
        np.fill_diagonal(K, 0.0)
    return K


def _keep_topk_per_row(K: np.ndarray, k: Optional[int]) -> np.ndarray:
    """
    Sparsify kernel by keeping only top-k weights per row.

    Args:
        K: (n, n) kernel matrix
        k: number of top weights to keep (None = keep all)

    Returns:
        K_sparse: (n, n) sparsified kernel
    """
    if k is None:
        return K
    n = K.shape[0]
    out = np.zeros_like(K)
    for i in range(n):
        row = K[i]
        # Indices of top-k (excluding ties to be deterministic)
        idx = np.argpartition(row, -k)[-k:]
        out[i, idx] = row[idx]
    return out


def _row_stochastic(K: np.ndarray) -> np.ndarray:
    """Convert kernel matrix to row-stochastic (Markov) matrix."""
    rs = K.sum(axis=1, keepdims=True)
    rs[rs <= 1e-12] = 1.0
    return K / rs


def kernel_from_embeddings(
    emb_by_node: Dict[str, np.ndarray],
    *,
    metric: str = "cosine",
    k_sig: int = 5,
    tau: float = 0.5,
    keep_topk: Optional[int] = None,
    zero_diag: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Build anisotropic, temperature-controlled kernel K (n x n) from neighbor embeddings.

    This implements the Zelnik-Manor & Perona self-tuning spectral clustering kernel
    with a global temperature parameter for additional control.

    Args:
        emb_by_node: Dict mapping node IDs to embedding vectors
        metric: 'cosine' (recommended for semantic spaces) or 'euclidean'
        k_sig: k for self-tuned σ_i (distance to k-th nearest neighbor)
        tau: global temperature multiplier (lower = sharper contrast)
        keep_topk: sparsify each row to its top-k weights before row-norm (optional)
        zero_diag: if True, zero out diagonal (no self-loops)

    Returns:
        K: (n, n) kernel matrix (NOT yet row-stochastic)
        nodes: list of node IDs in matrix order
    """
    nodes = sorted(emb_by_node.keys())
    X = np.stack([emb_by_node[n] for n in nodes], axis=0)

    if metric == "cosine":
        D = _pairwise_cosine_dist(X)
    elif metric == "euclidean":
        D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    sig = _self_tuned_sigmas(D, k_sig=k_sig)
    K = _build_anisotropic_kernel(D, sig, tau=tau, zero_diag=zero_diag)
    if keep_topk is not None:
        K = _keep_topk_per_row(K, keep_topk)
    return K, nodes


def _renyi2_eff_n(p_row: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute Rényi-2 effective number of neighbors for a row-stochastic row.

    Returns:
        Effective number: 1 / sum(p_i^2)
    """
    q = p_row / (p_row.sum() + eps)
    return float(1.0 / np.sum(q * q + eps))


def calibrate_tau_for_target_effn(
    emb_by_node: Dict[str, np.ndarray],
    target_effn: float = 7.0,
    *,
    metric: str = "cosine",
    k_sig: int = 5,
    zero_diag: bool = True,
    keep_topk: Optional[int] = None,
    tau_lo: float = 0.15,
    tau_hi: float = 1.2,
    iters: int = 18
) -> float:
    """
    Binary search on tau to achieve target mean Rényi-2 effective neighbors.

    Args:
        emb_by_node: Dict mapping node IDs to embeddings
        target_effn: desired mean effective number of neighbors
        metric: distance metric ('cosine' or 'euclidean')
        k_sig: k for self-tuned bandwidths
        zero_diag: zero diagonal in kernel
        keep_topk: optional sparsification
        tau_lo, tau_hi: search bounds
        iters: number of binary search iterations

    Returns:
        Calibrated tau value
    """
    def mean_effn(tau):
        K, _ = kernel_from_embeddings(
            emb_by_node, metric=metric, k_sig=k_sig, tau=tau,
            keep_topk=keep_topk, zero_diag=zero_diag
        )
        P = _row_stochastic(K)
        vals = [_renyi2_eff_n(P[i]) for i in range(P.shape[0])]
        return float(np.mean(vals))

    lo, hi = tau_lo, tau_hi
    mlo, mhi = mean_effn(lo), mean_effn(hi)

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        mmid = mean_effn(mid)
        # Move toward target
        if mmid > target_effn:
            # Too flat -> sharpen: decrease tau
            hi, mhi = mid, mmid
        else:
            # Too spiky -> soften: increase tau
            lo, mlo = mid, mmid

    return 0.5 * (lo + hi)


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
    bandwidth: Optional[float] = None,  # kept for backward-compat (unused in new path)
    kernel: str = 'gaussian',  # kept for backward-compat (unused in new path)
    *,
    metric: str = "cosine",
    k_sig: int = 5,
    tau: float = 0.5,
    keep_topk: Optional[int] = None,
    zero_diag: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Compute continuous, overlapping neighborhoods using kernel similarity.

    Now uses anisotropic kernel with self-tuned bandwidths and temperature control
    for better contrast in semantic spaces.

    Args:
        ego_data: EgoData object with embeddings
        bandwidth: (DEPRECATED - kept for backward compat, not used)
        kernel: (DEPRECATED - kept for backward compat, not used)
        metric: 'cosine' (recommended for semantic spaces) or 'euclidean'
        k_sig: k for self-tuned σ_i (distance to k-th nearest neighbor)
        tau: global temperature multiplier (lower = sharper contrast)
        keep_topk: sparsify each row to its top-k weights before row-norm (optional)
        zero_diag: if True, zero out diagonal (no self-loops)

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

    # Build kernel using new anisotropic approach
    K, nodes = kernel_from_embeddings(
        embeddings, metric=metric, k_sig=k_sig, tau=tau,
        keep_topk=keep_topk, zero_diag=zero_diag
    )

    # Convert matrix back to dict-of-dicts for backward compatibility
    kernel_weights: Dict[str, Dict[str, float]] = {}
    for i, ni in enumerate(nodes):
        row = {}
        for j, nj in enumerate(nodes):
            row[nj] = float(K[i, j]) if ni != nj else (0.0 if zero_diag else 1.0)
        kernel_weights[ni] = row

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


# ---------------------------
# Diffusion & Markov blanket primitives (kernel-based)
# ---------------------------

from scipy.linalg import expm as scipy_expm


def _row_stochastic_from_kernel(kernel_weights: Dict[str, Dict[str, float]]) -> Tuple[np.ndarray, List[str]]:
    """
    Build row-stochastic Markov operator P from kernel weights dict.
    Returns P (n x n) and node order.
    """
    nodes = sorted(kernel_weights.keys())
    idx = {n: i for i, n in enumerate(nodes)}
    K = np.zeros((len(nodes), len(nodes)), dtype=float)
    for i, ni in enumerate(nodes):
        wi = kernel_weights[ni]
        for nj, w in wi.items():
            if nj in idx:
                K[i, idx[nj]] = float(w)
    # zero-out self if you prefer a strict walk; leave as-is to keep self-mass
    # K[np.arange(len(nodes)), np.arange(len(nodes))] = 0.0
    row_sums = K.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 1e-12] = 1.0
    P = K / row_sums
    return P, nodes


def diffusion_profile(P: np.ndarray, t: float, i: int, mode: str = "rw") -> np.ndarray:
    """
    r_i^(t): diffusion row for node i after 'time' t.
    mode='rw': use e^{-t L_rw} with L_rw = I - P  (random-walk Laplacian)
    mode='powers': use integer steps: P^k with k=round(t)
    """
    n = P.shape[0]
    if mode == "powers":
        k = max(1, int(round(t)))
        r = np.eye(1, n, i) @ np.linalg.matrix_power(P, k)
        return np.asarray(r).ravel()
    # heat on random-walk Laplacian
    Lrw = np.eye(n) - P
    Kt = scipy_expm(-t * Lrw)   # (n,n)
    r = Kt[i]             # row i
    # normalize numerically just in case
    s = r.sum()
    return r / (s if s > 1e-12 else 1.0)


def diffusion_entropy(r: np.ndarray, alpha: float = 2.0, eps: float = 1e-12) -> float:
    """
    Rényi-α entropy; α=2 emphasizes concentration and spreads better than Shannon
    for small n. Returns H_2 by default.
    """
    p = np.clip(r, 0.0, 1.0)
    p = p / (p.sum() + eps)
    if abs(alpha - 2.0) < 1e-9:
        return float(-np.log((p**2).sum() + eps))
    # general α if you want it
    if alpha == 1.0:
        # Shannon (nats)
        nz = p[p > 0]
        return float(-np.sum(nz * np.log(nz)))
    s = (p**alpha).sum() + eps
    return float(np.log(1.0/s) / (alpha - 1.0))


def diffusion_distance(r1: np.ndarray, r2: np.ndarray, stationary: Optional[np.ndarray] = None, eps: float = 1e-12) -> float:
    """
    L2 distance in diffusion space, optionally reweighted by stationary distribution.
    """
    if stationary is None:
        diff = r1 - r2
        return float(np.linalg.norm(diff))
    w = 1.0 / (np.clip(stationary, eps, None))
    return float(np.sqrt(((r1 - r2)**2 * w).sum()))


def return_probability(P: np.ndarray, t: float, i: int, mode: str = "rw") -> float:
    """
    ρ_t(i) = probability of returning to i after diffusion time t.
    """
    n = P.shape[0]
    if mode == "powers":
        k = max(1, int(round(t)))
        return float(np.linalg.matrix_power(P, k)[i, i])
    Lrw = np.eye(n) - P
    Kt = scipy_expm(-t * Lrw)
    return float(Kt[i, i])


def select_markov_blanket_greedy(P: np.ndarray,
                                 t: float,
                                 i: int,
                                 candidates: Optional[np.ndarray] = None,
                                 k_max: int = 5,
                                 tol: float = 1e-3) -> Tuple[List[int], float]:
    """
    Greedy blanket B_i ⊆ candidates that reconstruct r_i^(t) as a convex combination of {r_j^(t)}_{j in B_i}.
    Minimizes squared error with nonnegative weights that sum to 1 (simple projected LS).
    Returns (indices of blanket, final MSE).
    """
    n = P.shape[0]
    if candidates is None:
        # all nodes except i
        candidates = np.array([j for j in range(n) if j != i], dtype=int)
    r_i = diffusion_profile(P, t, i)          # (n,)
    chosen: List[int] = []
    Rsel = np.zeros((n, 0), dtype=float)      # columns will be r_j
    mse_prev = np.inf

    for _ in range(k_max):
        # try adding each remaining candidate; pick the one that gives best MSE
        best_j = None
        best_mse = np.inf
        best_coef = None

        for j in candidates:
            Rtry = np.column_stack([Rsel, diffusion_profile(P, t, j)])
            # solve min ||Rtry w - r_i||^2  s.t. w>=0, sum w = 1
            # cheap NNLS with simplex projection via iterative least squares + projection
            w = _simplex_projected_ls(Rtry, r_i)
            resid = r_i - Rtry @ w
            mse = float((resid @ resid) / len(resid))
            if mse < best_mse - 1e-12:
                best_mse, best_j, best_coef = mse, j, w

        if best_j is None:
            break
        # accept
        chosen.append(int(best_j))
        candidates = candidates[candidates != best_j]
        Rsel = np.column_stack([Rsel, diffusion_profile(P, t, best_j)])
        # stopping if improvement stalls
        if mse_prev - best_mse <= tol:
            break
        mse_prev = best_mse

    # final MSE
    if Rsel.shape[1] == 0:
        return [], float(((r_i**2).sum()) / len(r_i))
    w = _simplex_projected_ls(Rsel, r_i)
    resid = r_i - Rsel @ w
    mse = float((resid @ resid) / len(resid))
    return chosen, mse


def _simplex_projected_ls(R: np.ndarray, y: np.ndarray, iters: int = 100, lr: float = 0.5) -> np.ndarray:
    """
    Minimize ||R w - y||^2 over w in the simplex (w>=0, sum w = 1).
    Small projected gradient loop is sufficient for n<=20.
    """
    m = R.shape[1]
    if m == 0:
        return np.zeros(0, dtype=float)
    w = np.full(m, 1.0/m, dtype=float)
    for _ in range(iters):
        grad = 2.0 * R.T @ (R @ w - y)
        w -= lr * grad
        w = _project_to_simplex(w)
    return w


def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    """
    Euclidean projection onto simplex {w: w>=0, sum w = 1}.
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0]
    if len(rho) == 0:
        # fallback: uniform if degenerate
        return np.full(n, 1.0/n, dtype=float)
    rho = rho[-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    return w / s if s > 1e-12 else np.full(n, 1.0/n, dtype=float)


def blanket_metrics_for_node(P: np.ndarray,
                             i: int,
                             t_small: float = 0.5,
                             t_big: float = 2.0,
                             k_max: int = 5,
                             mode: str = "rw") -> Dict[str, float]:
    """
    Compute a compact set of interpretable metrics for node i.

    Args:
        P: Row-stochastic Markov operator (n x n)
        i: Node index
        t_small: Short-range diffusion time
        t_big: Long-range diffusion time
        k_max: Maximum blanket size
        mode: 'rw' for random-walk Laplacian (matrix exponential)
              'powers' for discrete steps (matrix powers)

    Returns:
        Dict with 7 metrics: blanket_size, blanket_mse, H2_tsmall, H2_tbig,
        delta_H2, return_prob_k2, bridgeyness
    """
    n = P.shape[0]
    # stationary for reweighting (random-walk): degree proportional
    deg = P.sum(axis=1)
    pi = deg / (deg.sum() + 1e-12)

    r1 = diffusion_profile(P, t_small, i, mode=mode)
    r2 = diffusion_profile(P, t_big, i, mode=mode)
    H2_1 = diffusion_entropy(r1, alpha=2.0)
    H2_2 = diffusion_entropy(r2, alpha=2.0)
    dH2 = H2_2 - H2_1
    rho2 = return_probability(P, t=2.0, i=i, mode="powers")  # integer steps for interpretability

    blanket, mse = select_markov_blanket_greedy(P, t=t_small, i=i, k_max=k_max)
    # bridgeyness: how far apart blanket members are, in diffusion space at t_small
    if len(blanket) >= 2:
        profiles = [diffusion_profile(P, t_small, j, mode=mode) for j in blanket]
        # mean pairwise diffusion distance reweighted by pi
        dists = []
        for a in range(len(blanket)):
            for b in range(a+1, len(blanket)):
                dists.append(diffusion_distance(profiles[a], profiles[b], stationary=pi))
        bridgey = float(np.mean(dists))
    else:
        bridgey = 0.0

    return {
        "blanket_size": float(len(blanket)),
        "blanket_mse": float(mse),         # lower is better coverage of i by its blanket
        "H2_tsmall": float(H2_1),
        "H2_tbig": float(H2_2),
        "delta_H2": float(dH2),            # how much new reach i gets by diffusing further
        "return_prob_k2": float(rho2),     # stickiness
        "bridgeyness": float(bridgey)      # blanket diversity in diffusion space
    }
