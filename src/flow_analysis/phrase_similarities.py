"""
Phrase Similarity Computation
-----------------------------

Compute pairwise phrase-level semantic overlaps between focal node and neighbors.
These provide interpretable explanations of what creates semantic affinity.
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_phrase_similarities_for_pair(
    embedding_service,
    graph_name: str,
    focal_id: str,
    neighbor_id: str,
    similarity_threshold: float = 0.3,
    top_k: int = 10
) -> List[Dict]:
    """
    Compute semantic similarity between focal node's phrases and neighbor's phrases.

    Uses ChromaDB's vector search to efficiently find top-k most similar phrase pairs,
    avoiding O(N²) brute force comparison.

    NOTE: This is for contextual understanding in the UI, NOT what's used in
    predictability metrics. Those use mean embeddings, not individual phrase pairs.

    Args:
        embedding_service: EmbeddingService instance with ChromaDB access
        graph_name: Name of the ego graph collection
        focal_id: ID of the focal node
        neighbor_id: ID of the neighbor node
        similarity_threshold: Minimum cosine similarity to include (default: 0.3)
        top_k: Number of top pairs to return (default: 10)

    Returns:
        List of dicts with keys:
            - focal_phrase: Text from focal node
            - neighbor_phrase: Text from neighbor node
            - similarity: Cosine similarity score
            - focal_weight: Weight of focal phrase
            - neighbor_weight: Weight of neighbor phrase
    """
    # Get collection (uses get_or_create_collection to handle missing collections gracefully)
    collection = embedding_service.get_or_create_collection(graph_name)

    # Get all phrase data for focal node
    focal_data = embedding_service.get_all_node_phrases(graph_name, focal_id)

    if not focal_data:
        return []

    all_similarities = []

    # For each focal phrase, use ChromaDB's vector search to find similar neighbor phrases
    for focal_phrase_id, focal_phrase_data in focal_data.items():
        focal_emb = focal_phrase_data['embedding']
        focal_text = focal_phrase_data['text']
        focal_weight = focal_phrase_data['metadata'].get('weight', 1.0)

        # Query ChromaDB for top-k most similar phrases from this neighbor
        # This uses HNSW index, much faster than brute force O(N²)
        results = collection.query(
            query_embeddings=[focal_emb.tolist()],
            n_results=min(top_k * 2, 20),  # Get extras to filter by node_id
            where={"node_id": neighbor_id},  # Only phrases from this neighbor
            include=["metadatas", "distances", "documents"]
        )

        if not results['ids'] or not results['ids'][0]:
            continue

        # Convert ChromaDB distance to cosine similarity
        # ChromaDB uses squared euclidean by default, so we need to convert
        for distance, metadata, doc in zip(
            results['distances'][0],
            results['metadatas'][0],
            results['documents'][0]
        ):
            # Convert squared euclidean distance to cosine similarity
            # For normalized vectors: cosine = 1 - (euclidean²/2)
            similarity = 1.0 - (distance / 2.0)

            if similarity > similarity_threshold:
                all_similarities.append({
                    'focal_phrase': focal_text,
                    'neighbor_phrase': doc,
                    'similarity': float(similarity),
                    'focal_weight': focal_weight,
                    'neighbor_weight': metadata.get('weight', 1.0)
                })

    # Deduplicate by neighbor phrase - keep only best focal match for each neighbor phrase
    # This avoids showing the same neighbor phrase multiple times in the UI
    best_matches = {}
    for match in all_similarities:
        neighbor_phrase = match['neighbor_phrase']
        score = match['similarity'] * match['focal_weight'] * match['neighbor_weight']

        if neighbor_phrase not in best_matches or score > best_matches[neighbor_phrase]['score']:
            best_matches[neighbor_phrase] = {
                'match': match,
                'score': score
            }

    # Extract matches and sort by score
    deduplicated = [item['match'] for item in best_matches.values()]
    deduplicated.sort(
        key=lambda x: x['similarity'] * x['focal_weight'] * x['neighbor_weight'],
        reverse=True
    )

    return deduplicated[:top_k]


def compute_all_phrase_similarities(
    embedding_service,
    graph_name: str,
    nodes: List[str],
    similarity_threshold: float = 0.3,
    top_k: int = 10
) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Compute phrase similarities for all pairs involving the focal node.

    Args:
        embedding_service: EmbeddingService instance
        graph_name: Name of the ego graph
        nodes: List of node IDs (focal node should be first)
        similarity_threshold: Minimum cosine similarity to include
        top_k: Number of top pairs to return per neighbor

    Returns:
        Nested dict: focal_id -> neighbor_id -> list of similarity dicts
    """
    if not nodes:
        return {}

    focal_id = nodes[0]  # First node is always focal in our convention
    result = {focal_id: {}}

    for neighbor_id in nodes[1:]:
        similarities = compute_phrase_similarities_for_pair(
            embedding_service=embedding_service,
            graph_name=graph_name,
            focal_id=focal_id,
            neighbor_id=neighbor_id,
            similarity_threshold=similarity_threshold,
            top_k=top_k
        )

        if similarities:
            result[focal_id][neighbor_id] = similarities

    return result
