# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Drop in a Pond is a semantic network navigation system that uses geometric reasoning over embedding spaces to help users understand and strategically navigate their social/professional networks. It's distributed and privacy-preserving - each person maintains their own "epistemic ego graph" with predictions of neighbors' semantic fields.

## Working Agreements

### What Claude Should Do

- **Understand context before implementing**: Ask what problem we're solving and what success looks like. (e.g., "Are we optimizing for read or write performance?" or "Is the cramped feeling when zoomed in or out?")
- **Clarify subjective terms**: Words like "designy," "polished," "better," "efficient" mean different things. Ask what specifically matters. (e.g., "By efficient - faster execution, less memory, or cleaner code?")
- **Surface genuine ambiguity**: When there's a real choice with different trade-offs, present options briefly. Don't offer options when there isn't actual ambiguity - that's just noise. (e.g., "Memory cache is faster but ephemeral; SQLite is persistent but slower. Which matters more here?")
- **Flag uncertainty early**: If something might not work or seems wrong, say so before implementing. (e.g., "Not sure ReactFlow will respect mid-render position changes - test first?" or "100k row migration might need batching?")
- **Check when to commit**: Ask whether to commit now or keep iterating. Don't assume.
- **A clarifying question beats a wrong guess**: 30 seconds of discussion saves multiple iteration cycles.

### What Claude Can Expect from Me

- **Clear feedback**: I'll tell you directly when something is too big, too small, or not quite right.
- **Context when I have it**: If I know the viewing context (e.g., "we'll be zoomed out") or constraints, I'll share them.
- **Patience with questions**: Asking clarifying questions is always welcome and won't be seen as incompetence.
- **Honesty about preferences**: If I have a preference about commit style, design direction, or code organization, I'll tell you.
- **Engagement**: I'm actively reviewing the work and will catch things you miss. We're collaborating, not delegating.

## Development Principles

- **Less code is better.** Reduce, reuse. First refactor, then extend.
- **Make it work first**, then make it clean
- **Separate concerns**: functional code from effects, loops from loop bodies
- **Don't guess.** Use web search to look up APIs, how to use libraries etc. When implementing third-party library APIs, ALWAYS verify the correct method signatures by checking official documentation, reading type definitions, or creating test scripts to explore the API before writing code.
- **Step back and identify the core problem being asked about, rather than immediately jumping to modify existing code patterns**
- **Before proposing solutions, use the Task tool to analyze data flow. Ask it to trace how data moves through the system and report back with: 1) Data structures and their fields, 2) Function call chain with parameters, 3) Where data is missing or needs to be passed**
- **Consult documentation first**: Review `docs/architecture/` before starting. Use it to understand system context, but verify against actual code and update documentation when you find discrepancies.
- **Document architectural decisions**: When making system design choices, create ADRs in `docs/architecture/adr/` following the established template.
- **TypeScript prop interfaces**: When adding new properties to component interfaces, make them **required by default**. Only mark properties as optional if there's a specific runtime reason (e.g., intermediate migration states, genuinely optional features). If a property is always provided by callers, it should be required in the type signature.


## Common Commands

### Development

```bash
# Install dependencies (uses uv for package management)
uv sync

# Run semantic flow analysis (single-graph model)
uv run python src/semantic_flow.py

# Optional parameters
uv run python src/semantic_flow.py --alpha 0.6 --cos-min 0.2
```

**Output**: Creates timestamped analysis files in `data/analyses/`:
- `analysis_YYYYMMDD_HHMMSS.json`: Timestamped snapshot
- `analysis_latest.json`: Most recent analysis (single-graph model)

**Analysis includes**:
- **F**: Raw mutual predictability (symmetric phrase-level affinity)
- **D**: Semantic distance (1 - cosine similarity of mean embeddings)
- **F_MB**: Markov-blanket predictability (coupling given context)
- **E_MB**: Exploration potential (F_MB × (1 - D))
- **Coherence scores**: Semantic-structural alignment metrics
- **Cluster detection**: Community structure in blended graph
- **Connection suggestions**: Recommended new connections based on semantic proximity

See [docs/SEMANTIC_FLOW_GUIDE.md](docs/SEMANTIC_FLOW_GUIDE.md) for detailed explanation.

### Ego Graph Building Sessions

To help a user build or update their ego graph through conversation, use:

```
/ego-session
```

This loads a conversational guide that helps you:
- Extract semantic information naturally from dialogue
- Structure updates to ego graph JSON files
- Compute embeddings for new phrases
- Provide navigation insights at appropriate moments

See [.claude/commands/ego-session.md](.claude/commands/ego-session.md) for full details.

### FastAPI Backend Server

The backend provides a unified API that auto-detects your data source (Neo4j or files):

```bash
# Start the backend server
uv run uvicorn server.main:app --reload --port 3001
```

**Endpoints**:
- `GET /health`: Health check
- `GET /api/graph`: Returns ego graph structure (nodes, edges, focal, names)
- `GET /api/analysis`: Returns latest analysis results

**Single decision point - environment variables in `.env`**:
- With Neo4j credentials (`NEO4J_ID`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`): Loads from Neo4j
- Without Neo4j credentials: Loads from file-based storage in `data/ego_graph/`

Server logs indicate which data source is active: `[INFO] Loading from Neo4j (env vars detected)` or `[INFO] Loading from files (no Neo4j env vars)`.

### GUI Development

The project includes a React-based visualization interface:

```bash
# Start development server (requires backend running)
cd gui
npm install
npm run dev
# Opens at http://localhost:5173/
```

**GUI Architecture**:
- Built with React, Vite, xyflow/react for graph visualization
- Uses D3.js force-directed layout for natural node positioning
- Single-graph model: loads THE ego graph (no name input)
- **Always queries FastAPI backend** - no direct file access
- Backend handles data source selection transparently
- Displays analysis results from backend API
- Main components: `EgoGraphView` (orchestrator), `PersonNode` (node display), `PersonDetailSidebar` (details panel)

### Neo4j Backend (Alternative Storage)

Neo4j Aura provides graph-native storage with better scalability and vector search capabilities:

```bash
# Import ego graph to Neo4j with OpenAI embeddings (recommended)
python scripts/import_to_neo4j.py fronx

# Run analysis using Neo4j backend
python scripts/analyze_from_neo4j.py fronx
```

**Key features**:
- **OpenAI embeddings**: 10% better quality (MTEB 62.3 vs 56.3) for ~$0.0004/graph
- **Vector indexes**: Built-in semantic similarity search
- **Scalability**: Better performance for 100+ person networks
- **Cypher queries**: Rich graph traversals for future features

See [docs/NEO4J_BACKEND.md](docs/NEO4J_BACKEND.md) for setup and usage details.

### Testing

No test suite exists yet. When creating tests, use pytest:

```bash
uv run pytest tests/
```

## Architecture

### Core Concepts

**Epistemic ego graphs**: Each person maintains a JSON file with:
- Their own semantic field (ground truth) as phrase-level embeddings
- Predictive models of immediate neighbors' semantic fields
- Edge weights representing potential (semantic alignment) and actual (real interactions)

**Continuous semantic fields**: The system uses phrase-level embeddings as the primary representation (not single vectors per person).

**Temporal dynamics**: The data structure supports phrase weights and timestamps for future exponential decay (τ ≈ 40 days). Re-mentioning a phrase can bump its weight back up, creating a "living graph" that naturally forgets dormant topics.

### Six Navigation Metrics

The system computes six metrics to help navigate your network:

1. **Semantic landscape picture**: Cluster detection and semantic overlap analysis
2. **Public legibility (R²_in)**: How well neighbors can reconstruct your semantic field (ridge regression)
3. **Subjective attunement (R²_out)**: How well you understand neighbors' fields (with legibility threshold)
4. **Heat-residual novelty**: Topological distance using diffusion on graph Laplacian
5. **Translation vectors**: Semantic centroid differences between clusters
6. **Orientation scores**: Composite metric for choosing next interactions

### Code Organization

**src/storage.py**: Data loading and storage utilities
- `EgoData` dataclass: Core data structure for ego graphs
- `load_ego_graph()`: Load from modular directory structure

**src/clustering.py**: Clustering algorithms for ego graph neighbors
- `ego_clusters()`: Cluster neighbors using greedy modularity maximization
- `jaccard_overlap()`: Compute structural overlap between neighborhoods
- `tie_weight_entropy()`: Measure attention distribution across clusters

**src/translation_hints.py**: Finds lexical bridges between people's vocabularies using phrase-level semantic alignment

**src/embeddings.py**: ChromaDB integration for phrase embeddings

**src/neo4j_storage.py**: Neo4j Aura backend for graph storage
- `Neo4jConnection`: Context manager for Neo4j connections
- `load_ego_graph_from_neo4j()`: Load ego graph from Neo4j (returns EgoData)
- `save_ego_graph_to_neo4j()`: Save ego graph to Neo4j with OpenAI or local embeddings
- Supports native vector indexes for semantic similarity search

**src/semantic_flow.py**: Command-line tool for semantic-structural flow analysis
- Blends structural (edges) and semantic (embeddings) information
- Computes diffusion-based fields and predictability metrics
- Outputs analysis to `data/analyses/` for GUI visualization

**src/semantic_flow/**: Modular semantic flow analysis components
- `structural.py`: Build adjacency matrix from edges.json
- `semantic.py`: Load phrase embeddings and compute semantic affinity
- `blending.py`: Blend structural/semantic matrices, compute fields
- `clustering.py`: Detect communities in blended graph
- `suggestions.py`: Generate connection recommendations
- `coherence.py`: Compute semantic-structural coherence metrics
- `serialize.py`: Format and write analysis output

**src/validation.py**: JSON schema validation for ego graph data files

**gui/**: React-based visualization interface
- `src/components/`: React components (EgoGraphView, PersonNode, PersonDetailSidebar, etc.)
- `src/lib/`: Utilities (egoGraphLoader, d3Layout, metricLabels, metricInterpretation)
- Built with Vite, uses xyflow/react and D3.js

**data/ego_graphs/**: Ego graph data in modular directory format
- Each ego graph is a directory: `name/`
- Files are split for better editability and git tracking

**data/analyses/**: Analysis output files (generated by semantic_flow.py)
- Timestamped snapshots (`analysis_YYYYMMDD_HHMMSS.json`)
- Latest analysis (`analysis_latest.json`)
- Used by GUI for visualization

### Data Format (v0.2 - Modular with ChromaDB)

Ego graphs use a modular directory structure (embeddings stored separately in ChromaDB):

```
data/ego_graphs/name/
├── metadata.json           # Version and graph-level info
├── self.json              # Ego node's semantic field
├── connections/           # Individual files for each person
│   ├── person1.json
│   ├── person2.json
│   └── ...
├── edges.json            # All relationship edges
└── contact_points.json   # Past/present/potential interactions
```

**metadata.json**:
```json
{
  "version": "0.2",
  "format": "modular",
  "created_at": "2025-10-24",
  "description": "Ego graph for Your Name"
}
```

**self.json**:
```json
{
  "id": "your_id",
  "name": "Your Name",
  "phrases": [
    {"text": "semantic navigation", "weight": 0.9, "last_updated": "2025-10-24"}
  ]
}
```

**connections/person.json**:
```json
{
  "id": "person_id",
  "name": "Person Name",
  "phrases": [
    {"text": "topic area", "weight": 0.8, "last_updated": "2025-10-24"}
  ],
  "capabilities": ["skill1", "skill2"],
  "availability": [
    {"date": "2025-10-24", "score": 0.8, "content": "Generally available"}
  ],
  "notes": [
    {"date": "2025-10-24", "content": "Met at conference, shared interest in X"}
  ]
}
```

**edges.json**:
```json
[
  {
    "source": "your_id",
    "target": "person_id",
    "actual": 0.8,
    "channels": ["video_calls", "in_person"]
  }
]
```

**contact_points.json**:
```json
{
  "past": [
    {"date": "2024-05", "people": ["your_id", "person_id"], "content": "Met at event X"}
  ],
  "present": [
    {"people": ["your_id", "person_id"], "content": "Currently collaborating on Y"}
  ],
  "potential": [
    {"people": ["your_id", "person_id"], "content": "Plan to work on Z together"}
  ]
}
```

**Key points**:
- `phrases` array contains text only (NOT embeddings - those are in ChromaDB)
- Phrase embeddings computed via sentence-transformers (`all-MiniLM-L6-v2`, 384 dims)
- Embeddings cached in `./chroma_db/` directory (gitignored)
- Each graph gets its own ChromaDB collection
- Mean embedding computed on-demand from weighted phrase embeddings
- Phrase `weight` is optional (defaults to 1.0)
- `capabilities`: List of skills/expertise the person can help with
- `availability`: Timestamped availability observations (score 0-1)
- `notes`: Timestamped qualitative observations about the person
- `contact_points`: Relational/historical information about interactions (past/present/potential)
- Temporal decay not yet implemented (planned v0.3)

### Analysis Output Format

Semantic flow analysis produces JSON files in `data/analyses/`:

```json
{
  "graph_name": "fronx",
  "timestamp": "2025-10-27T23:26:13",
  "parameters": {
    "alpha": 0.4,
    "cos_min": 0.25
  },
  "nodes": [...],
  "fields": {
    "F": [[...], ...],      // Mutual predictability matrix
    "D": [[...], ...],      // Semantic distance matrix
    "F_MB": [[...], ...],   // Markov-blanket predictability
    "E_MB": [[...], ...]    // Exploration potential
  },
  "clusters": {
    "node_id": "cluster_name",
    ...
  },
  "coherence": {
    "node_id": {
      "semantic_coherence": 0.85,
      "structural_coherence": 0.72,
      ...
    }
  },
  "suggestions": {
    "node_id": [
      {"target": "other_id", "score": 0.89, "reason": "..."},
      ...
    ]
  }
}
```

The GUI automatically loads `analysis_latest.json` for visualization (single-graph model).

## Current Implementation Status

**✅ Works now (v0.2)**:
- All six navigation metrics with discrete clustering (working with real embeddings)
- Phrase-level semantic fields (ChromaDB integration)
- Sentence-transformer embeddings (`all-MiniLM-L6-v2`, 384 dims)
- JSON schema validation
- Example fixture (fronx) with 15 people, realistic semantic clusters
- Command-line analysis tool
- Keyphrase translation hints
- Conversational interface (`/ego-session` command)
- Semantic flow analysis (blended semantic-structural diffusion)
- Interactive GUI (React-based force-directed visualization)
- Coherence metrics (semantic-structural alignment)
- Connection suggestions (ML-based recommendations)

**📋 Planned (v0.3+)**:
- Kernel-based neighborhoods instead of discrete clusters
- Gradient-based translation instead of centroid differences
- On-demand clustering (compute transiently, never store)
- Temporal dynamics with exponential decay
- Active inference loop (predict → interact → update)
- Inter-node privacy-preserving protocol

See `docs/V02_MIGRATION.md` for details on the v0.2 architecture.

## Key Design Principles

**Privacy-first**: No centralized graph. Each person's ego graph contains their ground truth plus predictions of neighbors. No raw embedding exchange between nodes.

**Markov blanket principle**: All computations are local (1-2 hops from focal node). No need to see global network structure to make good navigation decisions.

**Continuous over discrete**: Design principle to avoid premature hardening of semantic structure. Phrase-level embeddings provide the foundation for future continuous field operations.

**Living memory**: Design principle for temporal dynamics. Timestamps are captured for future exponential decay implementation. The graph is designed to represent a "living present," not an archive.

## Mathematical Dependencies

The system combines:
- **Spectral graph theory**: Laplacians, heat kernels, diffusion distance
- **Statistical learning**: Ridge regression, R² metrics (scikit-learn)
- **Information theory**: Shannon entropy for attention distribution
- **Graph algorithms**: Community detection (greedy modularity clustering)

## Performance Considerations

Current implementation suitable for ego graphs with 10-1000 neighbors. For larger graphs:
- Use sparse matrix operations (scipy.sparse)
- Replace spectral clustering with approximate methods
- Use incremental updates instead of full recomputation

Heat-residual novelty uses `inv(I + t*L)` which is O(n³). For graphs >100 neighbors, optimize with:
- Truncated spectral decomposition (top-k eigenvectors)
- Local heat propagation (iterative diffusion)
- Sparse solvers (conjugate gradient)

## Documentation

Comprehensive documentation in `docs/`:
- `INDEX.md`: Documentation overview
- `VISION.md`: User experience and use cases
- `ARCHITECTURE.md`: Mathematical foundations and metric explanations
- `DISTRIBUTED.md`: Privacy model and federation protocol
- `CONTINUOUS_FIELDS.md`: Why continuous fields vs discrete clusters
- `TEMPORAL_DYNAMICS.md`: Exponential decay and living memory
- `IMPLEMENTATION.md`: Detailed implementation guide and roadmap

Start with `docs/INDEX.md` for full project overview.
