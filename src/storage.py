# storage.py
"""
Data loading and storage utilities for ego graphs.

Uses modular directory-based format for scalability and editability.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Iterable, Tuple, Optional
import json
import numpy as np

from dataclasses import dataclass


@dataclass
class EgoData:
    """
    Local ego graph around focal node F.

    nodes: list of node IDs including F and its neighbors (strings or ints).
    focal: the ID of the focal node (e.g., "F").
    embeddings: dict node_id -> np.ndarray of shape (d,)
    edges: iterable of (u, v, dims) where dims is a dict with edge dimensions:
           - 'potential': derived from embedding similarity (default)
           - 'actual': real interaction strength (optional)
           - 'past', 'present', 'future': temporal dimensions (optional)
           Legacy format (u, v, w) or (u, v) also supported.
    names: dict node_id -> human-readable name (optional)
    """
    nodes: List[str]
    focal: str
    embeddings: Dict[str, np.ndarray]
    edges: Iterable[Tuple[str, str, Optional[float | Dict[str, float]]]]
    names: Optional[Dict[str, str]] = None

    def graph(self, edge_dim: str = 'potential'):
        """
        Build a NetworkX graph using the specified edge dimension.

        edge_dim: which dimension to use for edge weights ('potential', 'actual', etc.)
        """
        import networkx as nx
        G = nx.Graph()
        for u in self.nodes:
            G.add_node(u)
        for e in self.edges:
            if len(e) == 2:
                u, v = e
                w = 1.0
                attrs = {}
            else:
                u, v, dims = e
                if dims is None:
                    w = 1.0
                    attrs = {}
                elif isinstance(dims, dict):
                    w = dims.get(edge_dim, dims.get('potential', 1.0))
                    # Preserve ALL edge attributes
                    attrs = dims.copy()
                else:
                    w = float(dims)
                    attrs = {}
            if u in G and v in G:
                G.add_edge(u, v, weight=float(w), **attrs)
        return G

    def Z(self, subset: Optional[Iterable[str]] = None) -> np.ndarray:
        if subset is None:
            subset = self.nodes
        return np.stack([self.embeddings[n] for n in subset], axis=0)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def load_ego_graph(
    directory: str | Path,
    embedding_service
) -> EgoData:
    """
    Load ego graph from v0.2 modular directory structure.

    Expected structure:
    directory/
      metadata.json
      self.json
      connections/
        person1.json
        person2.json
        ...
      edges.json
      contact_points.json  # not used in EgoData, but part of schema

    Args:
        directory: Path to modular ego graph directory
        embedding_service: EmbeddingService instance (required)

    Returns:
        EgoData instance
    """
    directory = Path(directory)

    if embedding_service is None:
        raise ValueError(
            "v0.2 modular format requires an EmbeddingService instance. "
            "Import from src.embeddings and pass get_embedding_service()"
        )

    # Load metadata
    with open(directory / "metadata.json") as f:
        metadata = json.load(f)

    if metadata.get("version") != "0.2":
        raise ValueError(f"Only v0.2 format supported. Found version: {metadata.get('version')}")

    if metadata.get("format") != "modular":
        raise ValueError(f"Expected modular format. Found: {metadata.get('format')}")

    # Load self
    with open(directory / "self.json") as f:
        self_node = json.load(f)

    focal_id = self_node["id"]

    # Load connections
    connections_dir = directory / "connections"
    connections_data = []
    if connections_dir.exists():
        for conn_file in sorted(connections_dir.glob("*.json")):
            with open(conn_file) as f:
                connections_data.append(json.load(f))

    # Load edges
    with open(directory / "edges.json") as f:
        edges_data = json.load(f)

    # Use directory name as graph name
    graph_name = directory.name

    # Process nodes and compute mean embeddings from phrases
    nodes = []
    embeddings = {}
    names = {}
    nodes_data = [self_node] + connections_data

    for node in nodes_data:
        node_id = node["id"]
        nodes.append(node_id)
        names[node_id] = node.get("name", node_id)

        phrases = node.get("phrases", [])
        if not phrases:
            # No phrases - use zero vector (edge case)
            embeddings[node_id] = np.zeros(384)  # all-MiniLM-L6-v2 dimension
            continue

        # Extract phrase texts and weights
        # Use content-based IDs (hash of text) so changes to phrase text are detected
        import hashlib
        phrase_texts = [p["text"] for p in phrases]
        phrase_ids = [
            f"{node_id}:{hashlib.sha256(text.encode()).hexdigest()[:16]}"
            for text in phrase_texts
        ]
        weights = {
            phrase_ids[i]: p.get("weight", 1.0)
            for i, p in enumerate(phrases)
        }

        # Check if phrases already exist
        try:
            existing_embeddings = embedding_service.get_embeddings(graph_name, phrase_ids)
            if len(existing_embeddings) != len(phrase_ids):
                # Some missing, add them
                embedding_service.add_phrases(
                    graph_name=graph_name,
                    node_id=node_id,
                    phrases=phrase_texts,
                    phrase_ids=phrase_ids
                )
        except:
            # Add all phrases
            embedding_service.add_phrases(
                graph_name=graph_name,
                node_id=node_id,
                phrases=phrase_texts,
                phrase_ids=phrase_ids
            )

        # Compute weighted mean embedding
        mean_embedding = embedding_service.compute_mean_embedding(
            graph_name=graph_name,
            node_id=node_id,
            weights=weights
        )
        embeddings[node_id] = mean_embedding

    # Build edges from explicit edge list only
    edge_dict = {}  # (u, v) -> dims dict

    if edges_data:
        for e in edges_data:
            u, v = e["source"], e["target"]
            key = tuple(sorted([u, v]))

            # Compute potential from embeddings
            cos_sim = max(0.0, _cos(embeddings[u], embeddings[v]))
            dims = {"potential": cos_sim}

            # Add any other dimensions (actual, etc.)
            extra_dims = {k: val for k, val in e.items() if k not in ["source", "target"]}
            dims.update(extra_dims)

            edge_dict[key] = dims

    # Convert to list format
    edges = [(u, v, dims) for (u, v), dims in edge_dict.items()]

    return EgoData(nodes=nodes, focal=focal_id, embeddings=embeddings, edges=edges, names=names)
