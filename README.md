# Drop in a Pond

A **semantic network navigation system** for understanding and strategically navigating your social/professional network using geometric reasoning over embedding spaces.

**Distributed and privacy-preserving**: each person runs local computations without revealing their full network to others.

## Quick Start

### Option 1: Visual Interface (with API Backend)

The GUI connects to a FastAPI backend that auto-detects your data source:

```bash
# Terminal 1: Start the backend server
uv sync
uv run uvicorn server.main:app --reload --port 3001

# Terminal 2: Start the GUI
cd gui
npm install
npm run dev
```

Then open `http://localhost:5173/` to view your ego graph.

**Single decision point - backend environment variables:**
- With Neo4j credentials (`.env`: `NEO4J_ID`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`): Backend loads from Neo4j
- Without Neo4j credentials: Backend loads from file-based storage in `data/ego_graph/`

The GUI always connects to the backend API and doesn't need to know which data source is active.

See [gui/README.md](gui/README.md) for GUI details.

### Option 2: Conversational Interface

Build your ego graph through natural conversation with Claude:

```bash
# Install dependencies
uv sync

# In a Claude Code session, run:
/ego-session
```

Claude will guide you through building your network graph conversationally. See [Conversational Interface](docs/CONVERSATIONAL_INTERFACE.md) for details.

### Option 3: Analyze Existing Graph

```bash
# Install dependencies
uv sync

# Run analysis from files
uv run python src/semantic_flow.py

# Or run analysis from Neo4j (if credentials are set)
uv run python scripts/analyze_from_neo4j.py
```

Both commands analyze your ego graph and output semantic flow analysis to `data/analyses/`. The data source is determined by:
- `analyze_from_neo4j.py`: Always loads from Neo4j (requires Neo4j credentials in `.env`)
- `semantic_flow.py`: Always loads from files in `data/ego_graph/`

Both produce identical analysis output files.

## Documentation

- **[Documentation Index](docs/INDEX.md)** - Start here for full overview
- **[Conversational Interface](docs/CONVERSATIONAL_INTERFACE.md)** - Build your graph through natural dialogue
- **[Semantic Flow Guide](docs/SEMANTIC_FLOW_GUIDE.md)** - Understanding analysis output
- **[Vision & User Experience](docs/VISION.md)** - What is this and what's it like to use?
- **[Architecture](docs/ARCHITECTURE.md)** - How does it work internally?

## What It Does

The system helps you answer questions like:

- What semantic regions exist in my network landscape?
- Who could bridge me to new communities?
- How should I frame my message to reach different regions?
- Where is the novelty - who offers perspectives I don't yet understand?

It operates on **phrase-level semantic embeddings** stored in ChromaDB, blending structural relationships with semantic affinity to reveal navigation opportunities.

## Analysis Output

The system performs semantic-structural flow analysis and generates:

- **F** (predictability): Mutual phrase-level affinity between people
- **D** (distance): Semantic distance based on mean embeddings
- **F_MB** (Markov-blanket predictability): Context-aware coupling
- **E_MB** (exploration potential): Combines predictability with distance
- **Coherence metrics**: Semantic-structural alignment scores
- **Cluster detection**: Community structure in blended graph
- **Connection suggestions**: Recommended new connections based on semantic proximity

See [Semantic Flow Guide](docs/SEMANTIC_FLOW_GUIDE.md) for detailed explanations.

## Ego Graph Format

Ego graphs use a modular directory structure with embeddings stored separately in ChromaDB:

```
data/ego_graph/
├── metadata.json           # Version and graph-level info
├── self.json              # Ego node's semantic field
├── connections/           # Individual files for each person
│   ├── person1.json
│   ├── person2.json
│   └── ...
├── edges.json            # All relationship edges
└── contact_points.json   # Past/present/potential interactions
```

**self.json** and **connections/*.json** contain:
```json
{
  "id": "person_id",
  "name": "Person Name",
  "phrases": [
    {"text": "topic area", "weight": 0.8, "last_updated": "2025-10-24"}
  ],
  "capabilities": ["skill1", "skill2"],
  "availability": [
    {"date": "2025-10-24", "score": 0.8, "content": "Available"}
  ],
  "notes": [
    {"date": "2025-10-24", "content": "Met at conference"}
  ]
}
```

**edges.json** contains:
```json
[
  {
    "source": "your_id",
    "target": "person_id",
    "actual": 0.8
  }
]
```

**Key points**:
- Phrases contain only text and weight, not embeddings
- Embeddings are computed and cached in ChromaDB (`./chroma_db/`)
- Mean embeddings computed on-demand from weighted phrase embeddings
- Use the `/ego-session` command to build your graph conversationally

## Dependencies

- Python ≥3.9
- numpy ≥1.24
- networkx ≥3.0
- scipy ≥1.10
- scikit-learn ≥1.2

Managed with [uv](https://github.com/astral-sh/uv) for fast, reliable package management.
