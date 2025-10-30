#!/usr/bin/env python3
"""Import ego graphs from JSON files to Neo4j Aura.

Usage:
    python scripts/import_to_neo4j.py <graph_name> [--clear-existing]

Examples:
    # Import the 'fronx' graph
    python scripts/import_to_neo4j.py fronx

    # Import and replace existing graph
    python scripts/import_to_neo4j.py fronx --clear-existing

Environment variables required:
    NEO4J_ID (or NEO4J_URI): Neo4j instance ID or connection URI
    NEO4J_USERNAME: Neo4j username
    NEO4J_PASSWORD: Neo4j password
"""

import argparse
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage import load_ego_graph
from src.neo4j_storage import save_ego_graph_to_neo4j
from src.embeddings import get_embedding_service


def load_node_details(ego_graph_dir: Path) -> dict:
    """Load detailed node information from JSON files.

    Args:
        ego_graph_dir: Path to ego graph directory

    Returns:
        Dict mapping node_id to full node details (phrases, capabilities, etc.)
    """
    node_details = {}

    # Load self node
    self_path = ego_graph_dir / 'self.json'
    if self_path.exists():
        with open(self_path) as f:
            self_data = json.load(f)
            node_details[self_data['id']] = self_data

    # Load connection nodes
    connections_dir = ego_graph_dir / 'connections'
    if connections_dir.exists():
        for conn_file in connections_dir.glob('*.json'):
            with open(conn_file) as f:
                conn_data = json.load(f)
                node_details[conn_data['id']] = conn_data

    return node_details


def load_contact_points(ego_graph_dir: Path) -> dict:
    """Load contact points from JSON file.

    Args:
        ego_graph_dir: Path to ego graph directory

    Returns:
        Dict with past/present/potential contact events
    """
    contact_points_path = ego_graph_dir / 'contact_points.json'
    if contact_points_path.exists():
        with open(contact_points_path) as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(
        description='Import ego graph from JSON files to Neo4j Aura'
    )
    parser.add_argument(
        'graph_name',
        help='Name of the ego graph to import (subdirectory in data/ego_graphs/)'
    )
    parser.add_argument(
        '--clear-existing',
        action='store_true',
        help='Clear existing graph with same name before importing'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data/ego_graphs'),
        help='Directory containing ego graph data (default: data/ego_graphs)'
    )
    parser.add_argument(
        '--embedding-model',
        choices=['openai', 'local'],
        default='openai',
        help='Embedding model to use: openai (recommended) or local (sentence-transformers)'
    )
    parser.add_argument(
        '--openai-model',
        default='text-embedding-3-small',
        help='OpenAI model name (default: text-embedding-3-small)'
    )

    args = parser.parse_args()

    # Verify graph directory exists
    ego_graph_dir = args.data_dir / args.graph_name
    if not ego_graph_dir.exists():
        print(f"Error: Graph directory not found: {ego_graph_dir}")
        sys.exit(1)

    print(f"Importing ego graph '{args.graph_name}' to Neo4j...")

    # Load ego graph using existing loader (gets EgoData structure)
    print("  Loading graph from JSON files...")
    embedding_service = get_embedding_service()
    ego_data = load_ego_graph(str(ego_graph_dir), embedding_service)

    print(f"  Loaded {len(ego_data.nodes)} nodes, {len(list(ego_data.edges))} edges")

    # Load detailed node information
    print("  Loading node details...")
    node_details = load_node_details(ego_graph_dir)

    # Load contact points
    print("  Loading contact points...")
    contact_points = load_contact_points(ego_graph_dir)

    # Save to Neo4j
    print(f"  Saving to Neo4j (using {args.embedding_model} embeddings)...")
    try:
        save_ego_graph_to_neo4j(
            graph_name=args.graph_name,
            ego_data=ego_data,
            node_details=node_details,
            contact_points=contact_points,
            embedding_model=args.embedding_model,
            openai_model=args.openai_model,
            clear_existing=args.clear_existing
        )
        print(f"\nSuccessfully imported '{args.graph_name}' to Neo4j!")
        print(f"  Focal node: {ego_data.focal} ({ego_data.names[ego_data.focal]})")
        print(f"  Total nodes: {len(ego_data.nodes)}")
        print(f"  Total edges: {len(list(ego_data.edges))}")
        print(f"  Embedding model: {args.embedding_model}")

    except Exception as e:
        print(f"\nError importing to Neo4j: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
