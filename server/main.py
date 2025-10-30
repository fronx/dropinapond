from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import json
import os
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage import load_ego_graph, EgoData
from src.neo4j_storage import load_ego_graph_from_neo4j
from src.embeddings import get_embedding_service
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# CORS configuration for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_graph() -> EgoData:
    """Load ego graph from configured data source (Neo4j or files)."""
    # Auto-detect data source based on environment variables
    if os.getenv('NEO4J_ID') and os.getenv('NEO4J_USERNAME') and os.getenv('NEO4J_PASSWORD'):
        print("[INFO] Loading from Neo4j (env vars detected)")
        return load_ego_graph_from_neo4j()
    else:
        print("[INFO] Loading from files (no Neo4j env vars)")
        ego_dir = Path(__file__).parent.parent / "data" / "ego_graph"
        embedding_service = get_embedding_service()
        return load_ego_graph(ego_dir, embedding_service)

@app.get("/api/graph")
async def get_graph():
    """Return ego graph structure (from Neo4j or files, auto-detected)."""
    ego_data = load_graph()

    # Convert to JSON-serializable format
    # Extract weight value: handle both dict (from Neo4j) and float (from files)
    def extract_weight(edge):
        if len(edge) <= 2:
            return 1.0
        weight_data = edge[2]
        if isinstance(weight_data, dict):
            return weight_data.get('actual', 1.0)
        return weight_data

    return {
        "nodes": ego_data.nodes,
        "focal": ego_data.focal,
        "edges": [{"source": e[0], "target": e[1], "weight": extract_weight(e)}
                  for e in ego_data.edges],
        "names": ego_data.names,
        # embeddings excluded (too large, GUI doesn't need raw embeddings)
    }

@app.get("/api/analysis")
async def get_analysis():
    """Return analysis results from JSON file (metrics, clusters, suggestions)."""
    # Analysis stays in JSON files - simpler and more natural for matrix data
    analyses_dir = Path(__file__).parent.parent / "data" / "analyses"
    latest_file = analyses_dir / "analysis_latest.json"

    if not latest_file.exists():
        return {"error": "No analysis found. Run analysis first."}

    with open(latest_file) as f:
        return json.load(f)

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
