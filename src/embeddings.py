"""
Embedding service using ChromaDB with sentence-transformers.

This module provides the interface between phrase text and vector embeddings,
using ChromaDB for persistent storage and automatic embedding computation.

Architecture:
- ChromaDB handles embedding computation (using all-MiniLM-L6-v2 by default)
- Embeddings are stored persistently on disk
- JSON ego graphs store only text, embeddings fetched on demand
- Each ego graph has its own ChromaDB collection
"""

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Manages embeddings for ego graphs using ChromaDB.

    Each ego graph gets its own collection in ChromaDB. Phrase embeddings are
    computed once and cached. The service provides methods to add phrases,
    retrieve embeddings, and manage collections.
    """

    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the embedding service.

        Args:
            persist_directory: Directory where ChromaDB will persist data
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Use default sentence-transformers embedding function
        # This uses all-MiniLM-L6-v2 (384 dimensions)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        logger.info(f"Initialized EmbeddingService with persist_directory={persist_directory}")

    def get_or_create_collection(self, graph_name: str):
        """
        Get or create a ChromaDB collection for an ego graph.

        Args:
            graph_name: Name of the ego graph (e.g., "fronx")

        Returns:
            ChromaDB collection object
        """
        collection_name = f"ego_graph_{graph_name}"

        collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"graph_name": graph_name}
        )

        logger.debug(f"Using collection: {collection_name} (count={collection.count()})")
        return collection

    def add_phrases(
        self,
        graph_name: str,
        node_id: str,
        phrases: List[str],
        phrase_ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict]] = None
    ) -> List[str]:
        """
        Add phrases to the collection, computing embeddings automatically.

        Args:
            graph_name: Name of the ego graph
            node_id: ID of the node these phrases belong to
            phrases: List of phrase texts
            phrase_ids: Optional list of IDs (generated if not provided)
            metadata: Optional list of metadata dicts for each phrase

        Returns:
            List of phrase IDs that were added
        """
        if not phrases:
            return []

        collection = self.get_or_create_collection(graph_name)

        # Generate IDs if not provided
        if phrase_ids is None:
            phrase_ids = [f"{node_id}:phrase_{i}" for i in range(len(phrases))]

        # Add node_id to metadata
        if metadata is None:
            metadata = [{"node_id": node_id} for _ in phrases]
        else:
            for meta in metadata:
                meta["node_id"] = node_id

        # ChromaDB will automatically compute embeddings
        collection.add(
            documents=phrases,
            ids=phrase_ids,
            metadatas=metadata
        )

        logger.info(f"Added {len(phrases)} phrases for node {node_id} in {graph_name}")
        return phrase_ids

    def get_embeddings(
        self,
        graph_name: str,
        phrase_ids: List[str]
    ) -> np.ndarray:
        """
        Retrieve embeddings for specific phrase IDs.

        Args:
            graph_name: Name of the ego graph
            phrase_ids: List of phrase IDs to retrieve

        Returns:
            numpy array of shape (n_phrases, embedding_dim)
        """
        if not phrase_ids:
            return np.array([])

        collection = self.get_or_create_collection(graph_name)

        result = collection.get(
            ids=phrase_ids,
            include=["embeddings"]
        )

        embeddings = np.array(result["embeddings"])
        logger.debug(f"Retrieved {len(embeddings)} embeddings from {graph_name}")

        return embeddings

    def get_node_embeddings(
        self,
        graph_name: str,
        node_id: str
    ) -> Dict[str, np.ndarray]:
        """
        Retrieve all phrase embeddings for a specific node.

        Args:
            graph_name: Name of the ego graph
            node_id: ID of the node

        Returns:
            Dict mapping phrase IDs to embeddings
        """
        collection = self.get_or_create_collection(graph_name)

        result = collection.get(
            where={"node_id": node_id},
            include=["embeddings"]
        )

        embeddings_dict = {
            phrase_id: np.array(embedding)
            for phrase_id, embedding in zip(result["ids"], result["embeddings"])
        }

        logger.debug(f"Retrieved {len(embeddings_dict)} phrase embeddings for node {node_id}")
        return embeddings_dict

    def get_all_node_phrases(
        self,
        graph_name: str,
        node_id: str
    ) -> Dict[str, Dict]:
        """
        Retrieve all phrases and their data for a node.

        Args:
            graph_name: Name of the ego graph
            node_id: ID of the node

        Returns:
            Dict mapping phrase IDs to dicts with keys: text, embedding, metadata
        """
        collection = self.get_or_create_collection(graph_name)

        result = collection.get(
            where={"node_id": node_id},
            include=["embeddings", "documents", "metadatas"]
        )

        phrases_dict = {}
        for i, phrase_id in enumerate(result["ids"]):
            phrases_dict[phrase_id] = {
                "text": result["documents"][i],
                "embedding": np.array(result["embeddings"][i]),
                "metadata": result["metadatas"][i]
            }

        return phrases_dict

    def compute_mean_embedding(
        self,
        graph_name: str,
        node_id: str,
        weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Compute weighted mean embedding for a node from its phrases.

        Args:
            graph_name: Name of the ego graph
            node_id: ID of the node
            weights: Optional dict mapping phrase IDs to weights

        Returns:
            Mean embedding vector
        """
        embeddings_dict = self.get_node_embeddings(graph_name, node_id)

        if not embeddings_dict:
            raise ValueError(f"No embeddings found for node {node_id}")

        embeddings = np.array(list(embeddings_dict.values()))

        if weights is None:
            # Uniform weights
            return embeddings.mean(axis=0)
        else:
            # Weighted average
            phrase_ids = list(embeddings_dict.keys())
            w = np.array([weights.get(pid, 1.0) for pid in phrase_ids])
            w = w / w.sum()  # Normalize
            return (embeddings.T @ w)

    def delete_collection(self, graph_name: str):
        """
        Delete a collection (useful for testing or cleanup).

        Args:
            graph_name: Name of the ego graph
        """
        collection_name = f"ego_graph_{graph_name}"
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.warning(f"Could not delete collection {collection_name}: {e}")

    def list_collections(self) -> List[str]:
        """
        List all ego graph collections.

        Returns:
            List of graph names
        """
        collections = self.client.list_collections()
        graph_names = []
        for collection in collections:
            if collection.name.startswith("ego_graph_"):
                graph_name = collection.name.replace("ego_graph_", "")
                graph_names.append(graph_name)
        return graph_names


# Singleton instance for easy access
_service: Optional[EmbeddingService] = None


def get_embedding_service(persist_directory: str = "./chroma_db") -> EmbeddingService:
    """
    Get or create the singleton embedding service instance.

    Args:
        persist_directory: Directory where ChromaDB persists data

    Returns:
        EmbeddingService instance
    """
    global _service
    if _service is None:
        _service = EmbeddingService(persist_directory=persist_directory)
    return _service
