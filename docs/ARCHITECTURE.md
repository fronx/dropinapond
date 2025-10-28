# Architecture

## System Overview

Drop in a Pond consists of three main components:

1. **Data Storage**: Modular JSON files + ChromaDB for embeddings
2. **Analysis Engine**: Python program that computes semantic flow metrics
3. **Visualization**: React-based web GUI that renders analysis results

```
┌─────────────────┐
│  Ego Graph      │
│  (JSON files)   │
└────────┬────────┘
         │
         ├──→ Python Analysis (semantic_flow.py)
         │    - Loads graph data
         │    - Computes embeddings via ChromaDB
         │    - Runs semantic flow analysis
         │    - Outputs analysis JSON
         │
         ├──→ ChromaDB
         │    - Stores phrase embeddings
         │    - Similarity search
         │
         ↓
┌─────────────────┐
│  Analysis JSON  │
│  (timestamped)  │
└────────┬────────┘
         │
         └──→ React GUI
              - Force-directed visualization
              - Interactive exploration
              - Metric display
```

## Data Storage

### Ego Graph Format

Each person's network is stored in a modular directory:

```
data/ego_graphs/<name>/
├── metadata.json           # Graph metadata
├── self.json              # Focal node's phrases
├── connections/           # One file per person
│   ├── person1.json
│   └── person2.json
├── edges.json            # Relationship strengths
└── contact_points.json   # Interaction history
```

**Key design decision**: Phrases contain only text and weights, NOT embeddings. This keeps files human-readable and version-control-friendly.

### ChromaDB Integration

Embeddings are computed on-demand and cached in ChromaDB:

- **Collection per graph**: Each ego graph gets its own ChromaDB collection
- **Phrase-level**: Each phrase gets a 384-dimensional embedding (sentence-transformers `all-MiniLM-L6-v2`)
- **Content-addressed**: Phrase IDs are hashes of text, enabling change detection
- **Automatic sync**: When JSON changes, embeddings are recomputed

**Why ChromaDB?**
- Fast similarity search for finding semantic matches
- Persistent caching (don't recompute unchanged embeddings)
- Clean separation of human-readable data from machine representations

## Analysis Engine

The Python analysis program (`src/semantic_flow.py`) performs:

1. **Load graph data** from modular JSON files (`src/storage.py`)
2. **Compute embeddings** for all phrases via ChromaDB (`src/embeddings.py`)
3. **Build matrices**:
   - Structural: Relationship strengths from `edges.json`
   - Semantic: Phrase-level affinity between connected people
   - Blended: Weighted combination of structure + semantics
4. **Compute fields**: Predictability, distance, coupling, exploration potential
5. **Detect clusters**: Community structure in the blended graph
6. **Generate suggestions**: High-affinity non-edges for new connections
7. **Output analysis**: Timestamped JSON file with all results

**Key modules**:
- `src/semantic_flow/structural.py`: Build adjacency matrix
- `src/semantic_flow/semantic.py`: Compute phrase-level affinity
- `src/semantic_flow/blending.py`: Blend structure and semantics
- `src/semantic_flow/clustering.py`: Detect communities
- `src/semantic_flow/coherence.py`: Measure semantic-structural alignment
- `src/semantic_flow/suggestions.py`: Recommend new connections

See [SEMANTIC_FLOW_GUIDE.md](SEMANTIC_FLOW_GUIDE.md) for detailed explanation of all metrics.

## Analysis Output

The analysis engine writes a JSON file to `data/analyses/<name>_latest.json`:

```json
{
  "graph_name": "fronx",
  "timestamp": "2025-10-27T23:26:13",
  "parameters": {
    "alpha": 0.4,
    "cos_min": 0.25
  },
  "metrics": {
    "layers": {
      "structural_edges": [[...]],
      "semantic_affinity": [[...]],
      "effective_edges": [[...]]
    },
    "fields": {
      "edge_fields": {...},
      "edge_fields_blanket": {...}
    },
    "coherence": {
      "regions": [...],
      "nodes": {...}
    }
  },
  "recommendations": {
    "semantic_suggestions": [...]
  }
}
```

This JSON is the **interface contract** between the analysis engine and the GUI.

## Visualization

The React-based GUI (`gui/`) reads analysis JSON and provides:

- **Force-directed graph layout**: Nodes positioned by `effective_edges`
- **Cluster coloring**: Hue shows community, saturation shows fit
- **Interactive exploration**: Click nodes to see details
- **Metric display**: All analysis results in sidebar

**Key files**:
- `gui/src/lib/egoGraphLoader.js`: Loads graph + analysis data
- `gui/src/lib/d3Layout.js`: Computes force-directed positions
- `gui/src/components/EgoGraphView.jsx`: Main orchestrator
- `gui/src/components/PersonNode.jsx`: Node rendering with encoding
- `gui/src/components/PersonDetailSidebar.jsx`: Metric display panel

## Design Principles

### Phrase-Level Semantic Fields

Instead of representing each person as a single vector, we store **multiple phrases with weights**. This preserves semantic nuance:

- "machine learning" and "embodied cognition" are both AI-related but occupy different semantic regions
- Weighted phrases capture importance and recency
- Mean embeddings computed on-demand for distance calculations
- Phrase-level affinity reveals specific overlaps, not just aggregate similarity

### Local Computation (Markov Blanket)

All analysis is local to the ego graph - you only need data about:
- Yourself (ground truth)
- Your immediate neighbors (your predictions)
- Their connections to each other

No global network view required. This enables:
- Privacy-preserving computation
- Scalable analysis (O(n²) in neighborhood size, not global network)
- Distributed deployment (future)

### Separation of Concerns

- **Data files**: Human-readable, version-control-friendly JSON
- **Embeddings**: Machine representation in ChromaDB
- **Analysis**: Computed on-demand, cached in timestamped files
- **Visualization**: Stateless UI that reads analysis JSON

This makes the system:
- **Inspectable**: You can read/edit the data files directly
- **Reproducible**: Same input → same output
- **Extensible**: Add new metrics without changing data format

## Technology Stack

- **Python 3.9+**: Analysis engine
- **ChromaDB**: Embedding storage and similarity search
- **sentence-transformers**: `all-MiniLM-L6-v2` model (384 dims)
- **networkx**: Graph algorithms and clustering
- **scikit-learn**: Statistical computations
- **React + Vite**: Web GUI
- **ReactFlow**: Force-directed graph visualization
- **D3.js**: Force simulation for layout

## Future Considerations

These are directions the architecture could evolve, not committed roadmap items:

- **Temporal dynamics**: Phrase weights decay over time unless refreshed
- **Federated protocol**: Lightweight updates between ego graphs
- **Incremental analysis**: Recompute only changed portions
- **Kernel-based neighborhoods**: Smooth transitions instead of discrete clusters

See [SEMANTIC_FLOW_GUIDE.md](SEMANTIC_FLOW_GUIDE.md) for metric details and interpretation.

See [CONVERSATIONAL_INTERFACE.md](CONVERSATIONAL_INTERFACE.md) for building ego graphs through dialogue.
