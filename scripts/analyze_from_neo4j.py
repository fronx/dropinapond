#!/usr/bin/env python3
"""Run semantic flow analysis using Neo4j as the data source (single-graph model).

NOTE: Currently loads from file-based storage. Phase 2 will integrate Neo4j loading.

Usage:
    python scripts/analyze_from_neo4j.py

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
from src.semantic_flow import analyze, Params


def main():
    parser = argparse.ArgumentParser(
        description='Run semantic flow analysis using Neo4j backend (single-graph model)'
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

    print(f"Running semantic flow analysis from file-based storage...")
    print("NOTE: Phase 2 will load from Neo4j instead")

    try:
        # Note: The analyze() function internally loads from file-based storage
        # Phase 2 will refactor analyze() to accept ego_data from Neo4j

        params = Params(
            alpha=args.alpha,
            cos_min=args.cos_min,
            export_dir=args.output_dir
        )

        output_path = analyze(params)
        print(f"\nAnalysis complete! Results saved to {output_path}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
