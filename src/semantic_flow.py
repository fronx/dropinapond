"""
Semantic-Structural Flow Analysis
---------------------------------

Blend factual topology (edges.json) with semantic affinity derived from phrase
embeddings in ChromaDB, then simulate diffusion on the blended graph.

Outputs a JSON under data/analyses/<name>_latest.json that your existing UI
(EgoGraphView) can already visualize.

Usage:
    uv run src/semantic_flow.py <graph_name> [--alpha 0.6] [--cos-min 0.2]
"""

from __future__ import annotations
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import networkx as nx

# Local project imports
from storage import load_ego_graph
from embeddings import get_embedding_service


# ---------- Parameters ----------

@dataclass
class Params:
    name: str = "fronx"
    alpha: float = 0.3        # blend: 1.0 = structural only
    cos_min: float = 0.5      # ignore phrase-pair cosine below this
    suggest_k: int = 3        # top-N non-edges suggested per node
    suggest_pool: int = 15    # how many nearest-by-mean to check
    export_dir: Optional[Path] = None


# ---------- Small utilities ----------

def _normalize_rows(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    rs = M.sum(axis=1, keepdims=True)
    rs = np.where(rs > eps, rs, 1.0)
    return M / rs


def _phrase_matrix(embedding_service, graph_name: str, node_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (E, w) where E:(m,d) unit-norm embeddings, w:(m,) phrase weights."""
    data = embedding_service.get_all_node_phrases(graph_name, node_id) or {}
    vecs, ws = [], []
    for _, p in data.items():
        v = np.asarray(p["embedding"], dtype=float)
        n = np.linalg.norm(v)
        if n > 0:
            vecs.append(v / n)
            ws.append(float(p.get("metadata", {}).get("weight", 1.0)))
    if not vecs:
        return np.zeros((0, 1)), np.zeros(0)
    E = np.stack(vecs, axis=0)
    w = np.asarray(ws)
    return E, w


def _weighted_mean_embedding(E: np.ndarray, w: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if E.size == 0:
        return np.zeros(1)
    ww = w / (w.sum() + eps)
    m = (ww[:, None] * E).sum(axis=0)
    n = np.linalg.norm(m)
    return m / (n + eps)


def _semantic_affinity(Ei: np.ndarray, wi: np.ndarray,
                       Ej: np.ndarray, wj: np.ndarray,
                       cos_min: float = 0.2, eps: float = 1e-12) -> float:
    """Phrase-level cosine affinity, weighted by phrase weights."""
    if Ei.size == 0 or Ej.size == 0:
        return 0.0
    S = Ei @ Ej.T
    mask = (S >= cos_min)
    if not np.any(mask):
        return 0.0
    W = np.outer(wi, wj)
    num = float((S * W * mask).sum())
    den = float((W * mask).sum() + eps)
    val = max(0.0, min(1.0, num / den))
    return val


def _timestamped_and_latest_paths(base_dir: Path, name: str) -> Tuple[Path, Path]:
    base_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"{name}_{ts}.json", base_dir / f"{name}_latest.json"


# ---------- Core analysis ----------

def analyze(params: Params) -> Path:
    root = Path(__file__).parent.parent
    ego_dir = root / "data" / "ego_graphs" / params.name
    out_dir = params.export_dir or (root / "data" / "analyses")

    embedding_service = get_embedding_service()
    ego = load_ego_graph(ego_dir, embedding_service)

    nodes = [ego.focal] + [n for n in ego.nodes if n != ego.focal]
    id_to_name = {nid: ego.names.get(nid, nid) for nid in nodes}

    idx = {nid: i for i, nid in enumerate(nodes)}
    n = len(nodes)

    # --- Structural matrix S (directed) ---
    S = np.zeros((n, n))
    edges_path = ego_dir / "edges.json"
    with open(edges_path) as f:
        edges_list = json.load(f)

    for e in edges_list:
        src, tgt = e["source"], e["target"]
        if src in idx and tgt in idx:
            w = float(e.get("actual", 0.3) or 0.0)
            S[idx[src], idx[tgt]] = np.clip(w, 0.0, 1.0)

    # --- Semantic phrase data ---
    phrase_E, phrase_w, mean_vec = {}, {}, {}
    for nid in nodes:
        E, w = _phrase_matrix(embedding_service, params.name, nid)
        phrase_E[nid] = E
        phrase_w[nid] = w
        mean_vec[nid] = _weighted_mean_embedding(E, w)

    # --- Semantic affinity on existing edges ---
    A = np.zeros((n, n))
    for i, src in enumerate(nodes):
        Ei, wi = phrase_E[src], phrase_w[src]
        for j, tgt in enumerate(nodes):
            if i == j or S[i, j] <= 0.0:
                continue
            Ej, wj = phrase_E[tgt], phrase_w[tgt]
            A[i, j] = _semantic_affinity(Ei, wi, Ej, wj, cos_min=params.cos_min)
            print("DEBUG", src, tgt, A[i, j])

    # --- Blended effective weights ---
    alpha = params.alpha
    W = alpha * S + (1 - alpha) * A

    # --- Mutual predictability and semantic distance fields (raw) ---
    # mean_vec already computed above
    M = np.stack([mean_vec[nid] for nid in nodes], axis=0)
    M_norm = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)

    # Predictability: symmetric mutual affinity
    F = np.sqrt(A * A.T)

    # Distance: 1 - cosine similarity between node mean embeddings
    D = 1 - np.clip(M_norm @ M_norm.T, -1.0, 1.0)

    # Normalize both to [0,1] for comparability
    F /= F.max() + 1e-12
    D /= D.max() + 1e-12

    # --- Markov-blanket coupling (mutual predictability given context) ---
    # Normalize rows of A to get local conditional probabilities, then
    # take elementwise product with its transpose to get symmetric coupling.
    row_sums = np.sum(A, axis=1, keepdims=True) + 1e-12
    A_norm = A / row_sums
    F_MB = A_norm * A_norm.T
    # Normalize for comparability
    F_MB /= F_MB.max() + 1e-12

    # Optional derived measure combining compatibility and contrast
    # Uses already-normalized D in [0,1]
    E_MB = F_MB * (1 - D)

    # --- Clusters (undirected for simplicity) ---
    Wu = W + W.T
    G = nx.Graph()
    for i, u in enumerate(nodes):
        G.add_node(u)
    for i in range(n):
        for j in range(i + 1, n):
            w = float(Wu[i, j])
            if w > 0:
                G.add_edge(nodes[i], nodes[j], weight=w)
    if G.number_of_edges() == 0:
        clusters = [sorted(nodes)]
    else:
        comms = nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
        clusters = [sorted(list(c)) for c in comms]

    # --- Suggestions: high-affinity non-edges ---
    M = np.stack([mean_vec[nid] for nid in nodes], axis=0)
    M_norm = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    mean_sim = np.clip(M_norm @ M_norm.T, -1.0, 1.0)
    existing = {(e["source"], e["target"]) for e in edges_list}
    suggestions = []
    for i, src in enumerate(nodes):
        cands = [(j, mean_sim[i, j]) for j in range(n) if j != i and (src, nodes[j]) not in existing]
        cands.sort(key=lambda t: t[1], reverse=True)
        cands = cands[:params.suggest_pool]
        Ei, wi = phrase_E[src], phrase_w[src]
        refined = []
        for j, _ in cands:
            tgt = nodes[j]
            Ej, wj = phrase_E[tgt], phrase_w[tgt]
            aff = _semantic_affinity(Ei, wi, Ej, wj, cos_min=params.cos_min)
            refined.append((tgt, aff))
        refined.sort(key=lambda t: t[1], reverse=True)
        for tgt, aff in refined[:params.suggest_k]:
            if aff > 0:
                suggestions.append({"source": src, "target": tgt, "affinity": float(aff)})

    # --- Helper: matrix → nested dict ---
    def mat_to_dict(mat: np.ndarray) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for i, u in enumerate(nodes):
            row = {}
            for j, v in enumerate(nodes):
                if i != j and mat[i, j] > 0:
                    row[v] = float(mat[i, j])
            out[u] = row
        return out

    layers = {
        "structural_edges": mat_to_dict(S),
        "semantic_affinity": mat_to_dict(A),
        "effective_edges": mat_to_dict(W),
    }

    analysis = {
        "version": "semantic-flow-1.0",
        "ego_graph_file": params.name,
        "parameters": vars(params),
        "metrics": {
            "clusters": clusters,
            "layers": layers,
        },
        "recommendations": {"semantic_suggestions": suggestions},
    }

    edge_fields = {}
    for i, src in enumerate(nodes):
        row = {}
        for j, tgt in enumerate(nodes):
            if i == j:
                continue
            row[tgt] = {
                "predictability_raw": float(F[i, j]),
                "distance_raw": float(D[i, j]),
            }
        edge_fields[src] = row

    analysis["metrics"]["fields"] = {"edge_fields": edge_fields}

    # Export Markov-blanket coupling fields
    edge_fields_MB = {}
    for i, src in enumerate(nodes):
        row = {}
        for j, tgt in enumerate(nodes):
            if i == j:
                continue
            row[tgt] = {
                "predictability_blanket": float(F_MB[i, j]),
                "exploration_potential": float(E_MB[i, j]),
            }
        edge_fields_MB[src] = row

    analysis["metrics"]["fields"]["edge_fields_blanket"] = edge_fields_MB

    ts_path, latest_path = _timestamped_and_latest_paths(out_dir, params.name)
    with open(ts_path, "w") as f:
        json.dump(analysis, f, indent=2)
    with open(latest_path, "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"[ok] Wrote {ts_path.name}")
    print(f"[ok] Updated {latest_path.name}")
    return latest_path


# ---------- CLI ----------

def _parse_args(argv: List[str]) -> Params:
    p = Params()
    if len(argv) >= 2:
        p.name = argv[1]
    for i, a in enumerate(argv[2:], start=2):
        if a == "--alpha" and i + 1 < len(argv):
            p.alpha = float(argv[i + 1])
        if a == "--cos-min" and i + 1 < len(argv):
            p.cos_min = float(argv[i + 1])
    return p


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run src/semantic_flow.py <graph_name> [--alpha 0.6] [--cos-min 0.2]")
        sys.exit(1)
    params = _parse_args(sys.argv)
    analyze(params)
