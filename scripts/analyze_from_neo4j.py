#!/usr/bin/env python3
"""Run semantic flow analysis using Neo4j as the data source.

This demonstrates how to use the Neo4j backend as a drop-in replacement
for the file-based storage, with no changes to the analysis pipeline.

Usage:
    python scripts/analyze_from_neo4j.py <graph_name>

Example:
    python scripts/analyze_from_neo4j.py fronx

Environment variables required:
    NEO4J_ID (or NEO4J_URI): Neo4j instance ID or connection URI
    NEO4J_USERNAME: Neo4j username
    NEO4J_PASSWORD: Neo4j password
"""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.neo4j_storage import load_ego_graph_from_neo4j
from src.embeddings import get_embedding_service
from src.semantic_flow import run_analysis


def main():
    parser = argparse.ArgumentParser(
        description='Run semantic flow analysis using Neo4j backend'
    )
    parser.add_argument(
        'graph_name',
        help='Name of the ego graph in Neo4j'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.4,
        help='Blending parameter (0=structural only, 1=semantic only, default: 0.4)'
    )
    parser.add_argument(
        '--cos-min',
        type=float,
        default=0.25,
        help='Minimum cosine similarity threshold (default: 0.25)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/analyses'),
        help='Directory for analysis output (default: data/analyses)'
    )

    args = parser.parse_args()

    print(f"Loading ego graph '{args.graph_name}' from Neo4j...")

    try:
        # Load from Neo4j
        embedding_service = get_embedding_service()
        ego_data = load_ego_graph_from_neo4j(
            graph_name=args.graph_name,
            embedding_service=embedding_service
        )

        print(f"Loaded {len(ego_data.nodes)} nodes, {len(list(ego_data.edges))} edges")
        print(f"Focal node: {ego_data.focal} ({ego_data.names[ego_data.focal]})")

        # Run analysis (same as with file-based storage)
        print("\nRunning semantic flow analysis...")
        run_analysis(
            ego_data=ego_data,
            graph_name=args.graph_name,
            alpha=args.alpha,
            cos_min=args.cos_min,
            output_dir=args.output_dir
        )

        print(f"\nAnalysis complete! Results saved to {args.output_dir}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
