#!/usr/bin/env python3
"""Start the FastAPI server, resuming Neo4j if needed."""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

def check_neo4j_needed():
    """Check if Neo4j credentials are configured."""
    return all([
        os.getenv('NEO4J_ID'),
        os.getenv('NEO4J_USERNAME'),
        os.getenv('NEO4J_PASSWORD')
    ])

def check_neo4j_connectivity():
    """Test if Neo4j instance is reachable."""
    neo4j_id = os.getenv('NEO4J_ID')
    username = os.getenv('NEO4J_USERNAME')
    password = os.getenv('NEO4J_PASSWORD')

    uri = f"neo4j+s://{neo4j_id}.databases.neo4j.io"

    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False

def resume_neo4j():
    """Resume the Neo4j instance."""
    print("Neo4j instance not reachable. Attempting to resume...")
    resume_script = Path(__file__).parent / "resume_neo4j.py"
    result = subprocess.run([sys.executable, str(resume_script)])
    return result.returncode == 0

def start_server(port=3002):
    """Start the FastAPI server."""
    print(f"\nStarting FastAPI server on port {port}...")
    subprocess.run([
        "uv", "run", "uvicorn", "server.main:app",
        "--reload", "--port", str(port)
    ])

def main():
    port = int(os.getenv('API_PORT', 3002))

    # Check if Neo4j is configured
    if check_neo4j_needed():
        print("Neo4j credentials detected. Checking connectivity...")

        # Try to connect
        if not check_neo4j_connectivity():
            print("✗ Neo4j not reachable")

            # Try to resume
            if not resume_neo4j():
                print("\nError: Failed to resume Neo4j instance.")
                print("Options:")
                print("  1. Check Neo4j Aura console: https://console.neo4j.io/")
                print("  2. Use file-based storage by removing Neo4j credentials from .env")
                sys.exit(1)

            # Check connectivity again after resume
            print("\nVerifying Neo4j connectivity...")
            if not check_neo4j_connectivity():
                print("✗ Still can't connect. Please check Neo4j console.")
                sys.exit(1)

        print("✓ Neo4j is ready")
    else:
        print("Using file-based storage (no Neo4j credentials)")

    # Start the server
    start_server(port)

if __name__ == "__main__":
    main()
