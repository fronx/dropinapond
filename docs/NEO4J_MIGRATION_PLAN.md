# Neo4j Migration Plan

## Overview

This document tracks the migration of Drop in a Pond to support Neo4j Aura as an alternative storage and analysis backend. The goal is to enable the full pipeline: import ego graphs → run analysis → store results in Neo4j → query from React GUI.

## Current Status (Completed)

### Phase 1: Basic Neo4j Storage ✓

**What we built:**
- Core Neo4j storage module ([src/neo4j_storage.py](../src/neo4j_storage.py))
  - `Neo4jConnection`: Environment-based connection management
  - `load_ego_graph_from_neo4j()`: Loads graph data, returns `EgoData`
  - `save_ego_graph_to_neo4j()`: Saves graph with embeddings
  - Automatic vector index creation for semantic similarity

**Graph schema implemented (Phase 1):**
- `Person` nodes with embeddings and `graph_name` property (384 or 1536 dimensions)
- `Phrase` nodes with embeddings and `graph_name` property
- `Event` nodes for contact points with `graph_name` property
- Relationships: `HAS_PHRASE`, `CONNECTED_TO`, `INVOLVES`
- Vector indexes on Person and Phrase embeddings

**Note:** Phase 1 schema included `graph_name` for multi-graph support. This will be removed in Phase 2 based on single-graph decision.

**Embedding options:**
- OpenAI embeddings (recommended): text-embedding-3-small (MTEB 62.3)
- Local sentence-transformers: all-MiniLM-L6-v2 (MTEB 56.3)
- Server-side computation via Neo4j `genai.vector.encodeBatch()` for OpenAI
- Client-side computation for local embeddings

**Scripts created:**
- [scripts/import_to_neo4j.py](../scripts/import_to_neo4j.py): Migrate JSON graphs to Neo4j
- [scripts/analyze_from_neo4j.py](../scripts/analyze_from_neo4j.py): Run analysis (currently limited)

**Testing results:**
- Successfully imported `fronx` graph: 31 nodes, 61 edges
- Embeddings stored correctly in Neo4j
- Vector indexes created automatically

**Configuration:**
- Added `python-dotenv` dependency for `.env` loading
- Environment variables: `NEO4J_ID`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `OPENAI_API_KEY`
- Auto-construction of URI from `NEO4J_ID`

## Current Limitation

The analysis pipeline ([src/semantic_flow.py](../src/semantic_flow.py)) currently:
1. Loads ego graphs from file-based storage (`load_ego_graph(ego_dir, ...)`)
2. Runs analysis computations
3. Writes results to JSON files in `data/analyses/`

This means:
- `analyze_from_neo4j.py` can't actually use Neo4j data yet
- GUI can't query graph data from Neo4j directly (loads from JSON files)

## Architecture Decision: Analysis Storage

**Decision:** Keep analysis results in JSON files, not Neo4j.

**Rationale:**
- **Separation of concerns:** Neo4j stores source data (graph), JSON stores computed data (analysis)
- **Natural data structures:** Matrix data (F, D, F_MB, E_MB) fits JSON better than graph properties
- **Simpler:** No need to design Neo4j analysis schema or sync logic
- **Regenerable:** Analysis is derived and can be recomputed from graph data anytime
- **Clear architecture:**
  ```
  Neo4j (graph: people, phrases, connections, embeddings)
       ↓
  Analysis pipeline (compute metrics)
       ↓
  JSON files (analysis results: matrices, clusters, suggestions)
       ↓
  FastAPI (serves graph from Neo4j + analysis from JSON)
       ↓
  GUI
  ```

## Phase 1B: Simplify to Single-Graph Model ✓

**Status:** Completed 2025-10-30

**Breaking changes implemented:**

Since we've decided on single-graph model, we refactored Phase 1 code:

### File System Changes ✓
1. Restructured `data/ego_graphs/` → `data/ego_graph/` (singular):
   - Moved `data/ego_graphs/fronx/*` → `data/ego_graph/`
   - Removed subdirectory, graph data lives at top level
   - Updated `.gitignore` to reflect new structure

2. Updated [src/storage.py](../src/storage.py):
   - Uses fixed `"ego_graph"` collection name for ChromaDB
   - Added duplicate phrase detection with clear error messages
   - Always loads from `data/ego_graph/` directory

3. Updated [src/semantic_flow.py](../src/semantic_flow.py):
   - Removed `name` parameter from `Params`
   - Simplified command-line interface (no graph name argument)
   - Writes to `latest.json` instead of `{name}_latest.json`

### Neo4j Changes ✓
1. Updated [src/neo4j_storage.py](../src/neo4j_storage.py):
   - Removed `graph_name` parameter from all functions
   - Removed `graph_name` property from all Cypher queries
   - Simplified queries: `MATCH (focal:Person {is_focal: true})`
   - Fixed Cypher syntax for vector property creation
   - Added progress output during import
   - Computed weighted mean embeddings in Python (simpler than Cypher)

2. Updated Neo4j schema:
   - Deleted all nodes from Phase 1 testing
   - Re-imported without `graph_name` properties
   - Vector indexes created: `person_embedding`, `phrase_embedding`

3. Updated [scripts/import_to_neo4j.py](../scripts/import_to_neo4j.py):
   - Removed `graph_name` argument
   - Changed default embedding model to `local` (sentence-transformers)
   - Always imports THE graph (no name needed)

4. Updated [scripts/analyze_from_neo4j.py](../scripts/analyze_from_neo4j.py):
   - Removed `graph_name` argument
   - Added note that Phase 2 will integrate Neo4j loading

### Testing Phase 1B ✓
- ✅ File-based analysis works correctly after restructure
- ✅ Neo4j import successful: 31 nodes, 61 edges with simplified schema
- ✅ All queries work without `graph_name` filter
- ✅ Analysis outputs to `latest.json` and timestamped files
- ✅ Duplicate phrase detection working with clear error messages

**Actual time:** ~2 hours

## Phase 2: Connect Analysis Pipeline to Neo4j (In Progress)

### Goal

Enable the full pipeline:
1. Load graph from Neo4j
2. Run analysis computations
3. Save results to JSON
4. Serve both graph and analysis via FastAPI
5. GUI queries backend API

**Key insight:** Analysis stays in JSON (natural for matrix data), only graph data in Neo4j.

**No schema changes needed:** Neo4j stores graph, JSON stores analysis - clean separation.

### Implementation Plan

#### Step 0: Simplify JSON Analysis Storage ✓

**Status:** Completed 2025-10-30

**Changes made:**
- Updated [src/semantic_flow/serialize.py](../src/semantic_flow/serialize.py):
  - Filenames changed from `latest.json` to `analysis_latest.json`
  - Timestamped files changed from `{ts}.json` to `analysis_{ts}.json`
  - Updated docstrings to reflect new naming convention

- Updated [src/semantic_flow.py](../src/semantic_flow.py):
  - Updated module docstring to reference new `analysis_latest.json` filename

- Updated [gui/src/lib/egoGraphLoader.js](../gui/src/lib/egoGraphLoader.js):
  - Changed `loadEgoGraph()` to use `/data/ego_graph` (single-graph model)
  - Changed `loadLatestAnalysis()` to load from `/data/analyses/analysis_latest.json`
  - Removed graph name from error messages (single-graph model)

**Testing:**
- ✅ Analysis runs successfully and creates `analysis_latest.json`
- ✅ Timestamped files use new naming: `analysis_20251030_105329.json`
- ✅ Files appear in GUI's public directory

**Next:** Clean up old analysis files (optional) and proceed to Step 1.

#### Step 1: Refactor `analyze()` to Accept `EgoData`

**Current signature:**
```python
def analyze(params: Params) -> Path
```

**New signature:**
```python
def analyze(ego_data: EgoData, params: AnalysisParams) -> AnalysisResult
```

**Changes needed:**
- Remove internal `load_ego_graph()` call from [src/semantic_flow.py:55](../src/semantic_flow.py#L55)
- Accept `ego_data` parameter instead
- Return structured `AnalysisResult` dataclass instead of file path
- Keep all analysis computations unchanged

**Files to modify:**
- [src/semantic_flow.py](../src/semantic_flow.py): Refactor `analyze()` function
- Create new dataclass for `AnalysisResult` to hold all computed fields

#### Step 2: Simplify JSON Analysis Storage

**Current behavior:**
- Analysis writes to `data/analyses/fronx_latest.json` and timestamped files
- Graph name embedded in filename

**New behavior (single-graph model):**
- Analysis writes to `data/analyses/latest.json` and timestamped files
- No graph name in filename (only one graph exists)

**Functions to update:**

```python
# In src/semantic_flow.py or new src/analysis_storage.py
def save_analysis_to_json(analysis: AnalysisResult, output_dir: Path) -> Path
    # Writes to output_dir/latest.json and output_dir/YYYYMMDD_HHMMSS.json
    # No graph name needed

def load_analysis_from_json(analyses_dir: Path) -> AnalysisResult
    # Loads from analyses_dir/latest.json
    # No graph name parameter needed
```

**Design considerations:**
- Keep existing JSON format structure (for GUI compatibility)
- Remove `graph_name` field from JSON output
- Simpler filenames: `latest.json` instead of `fronx_latest.json`

#### Step 3: Update Scripts to Use New Architecture

**[scripts/analyze_from_neo4j.py](../scripts/analyze_from_neo4j.py):**
```python
# Load ego graph from Neo4j (single graph, no name needed)
ego_data = load_ego_graph_from_neo4j()

# Run analysis (now accepts ego_data)
analysis_result = analyze(ego_data, params)

# Save to JSON (analysis stays in JSON files)
save_analysis_to_json(analysis_result, output_dir)
```

**[src/semantic_flow.py](../src/semantic_flow.py) main block:**
```python
# Load from files (single graph at data/ego_graphs/)
ego_data = load_ego_graph(ego_graphs_dir, embedding_service)

# Run analysis
analysis_result = analyze(ego_data, params)

# Save to JSON
save_analysis_to_json(analysis_result, output_dir)
```

**Key point:** Both scripts produce identical JSON output. The only difference is where the graph data comes from (files vs Neo4j).

#### Step 4: Create FastAPI Backend

**Create new `server/` directory with FastAPI application:**

**server/main.py:**
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import json
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.neo4j_storage import load_ego_graph_from_neo4j
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

@app.get("/api/graph")
async def get_graph():
    """Return ego graph structure from Neo4j (nodes, edges, names)."""
    ego_data = load_ego_graph_from_neo4j()
    return {
        "nodes": ego_data.nodes,
        "focal": ego_data.focal,
        "edges": list(ego_data.edges),
        "names": ego_data.names,
        # embeddings excluded (too large, GUI doesn't need raw embeddings)
    }

@app.get("/api/analysis")
async def get_analysis():
    """Return analysis results from JSON file (metrics, clusters, suggestions)."""
    # Analysis stays in JSON files - simpler and more natural for matrix data
    analyses_dir = Path(__file__).parent.parent / "data" / "analyses"
    latest_file = analyses_dir / "latest.json"

    if not latest_file.exists():
        return {"error": "No analysis found. Run analysis first."}

    with open(latest_file) as f:
        return json.load(f)

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
```

**server/requirements.txt:**
```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-dotenv>=1.0.0
```

**Running the backend:**
```bash
# Install dependencies
pip install -r server/requirements.txt

# Run development server
uvicorn server.main:app --reload --port 3001

# Or use uv:
uv pip install -r server/requirements.txt
uv run uvicorn server.main:app --reload --port 3001
```

**Deployment:**
- Create `server/.env` with Neo4j credentials
- Use existing root `.env` or symlink for development
- For production, set environment variables on hosting platform

#### Step 5: Update GUI to Query FastAPI Backend

**Current GUI behavior:**
- Loads ego graph from `data/ego_graphs/fronx/` (symlinked to `gui/public/data/ego_graphs/`)
- Loads analysis from `data/analyses/fronx_latest.json`
- See [gui/src/lib/egoGraphLoader.ts](../gui/src/lib/egoGraphLoader.ts)

**New behavior (with FastAPI backend):**

**Data sources:**
- Graph data: FastAPI queries Neo4j (`GET /api/graph`)
- Analysis data: FastAPI reads JSON file (`GET /api/analysis`)
- No Neo4j credentials in frontend - backend handles all authentication

**Frontend changes:**
- [gui/src/lib/egoGraphLoader.ts](../gui/src/lib/egoGraphLoader.ts): Add API mode
- Create API client module for backend communication
- Add environment variable for backend URL (e.g., `VITE_API_URL=http://localhost:3001`)
- Create new hooks: `useApiGraph()`, `useApiAnalysis()`
- Support both modes:
  - **File mode** (development): Load from symlinked JSON files
  - **API mode** (production): Query FastAPI backend

**Implementation approach:**
```typescript
// gui/src/lib/apiClient.ts
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3001';

export async function fetchGraph() {
  const response = await fetch(`${API_URL}/api/graph`);
  return response.json();
}

export async function fetchAnalysis() {
  const response = await fetch(`${API_URL}/api/analysis`);
  return response.json();
}
```

**Simplified by single-graph model:**
- No graph name in API URLs
- Just `/api/graph` and `/api/analysis` (always returns THE graph)
- Simpler frontend code (no name parameter needed)

### Benefits of This Architecture

**Clean separation of concerns:**
- Neo4j stores graph data (people, relationships, semantic fields) - the source of truth
- JSON stores analysis results (metrics, matrices) - derived computations
- Each stores data in its natural format (graph vs matrices)

**Simplicity:**
- No complex Neo4j analysis schema to design or maintain
- No sync logic between Neo4j and analysis results
- Analysis is regenerable from graph data anytime

**Flexibility:**
- Can swap Neo4j for other graph storage if needed
- Can swap JSON for other analysis storage (PostgreSQL, etc.) independently
- Changes to one don't affect the other

**Performance:**
- Neo4j optimized for graph queries (connections, neighbors, semantic similarity)
- JSON/filesystem fast for reading complete analysis snapshots
- No unnecessary database queries for matrix data

**Future features enabled:**
- Collaborative graphs: Multiple users can share Neo4j instance
- Real-time graph updates: GUI can subscribe to graph changes
- Advanced semantic queries: "Find people semantically similar to Alice" (vector search in Neo4j)
- Analysis stays simple: Just files, easy to version/backup/compare

## Testing Strategy

**Phase 1B: Single-graph simplification**
1. Restructure file system (move fronx/* up one level)
2. Remove graph_name from all Python code
3. Update Neo4j storage to remove graph_name
4. Re-import to Neo4j with simplified schema
5. Verify file-based analysis still works

**Phase 2A: Analysis refactoring**
1. Refactor `analyze()` to accept `EgoData` instead of loading internally
2. Create `AnalysisResult` dataclass
3. Update JSON save/load to use `latest.json` (no graph name)
4. Test with file-based storage (should match current output)

**Phase 2B: Neo4j analysis pipeline**
1. Create `analyze_from_neo4j.py` that loads from Neo4j
2. Test full pipeline: Neo4j → analyze → JSON
3. Verify JSON output identical to file-based pipeline

**Phase 2C: FastAPI backend**
1. Create `server/` directory with FastAPI app
2. Implement `/api/graph` endpoint (queries Neo4j)
3. Implement `/api/analysis` endpoint (reads JSON)
4. Test endpoints with curl/Postman
5. Verify CORS configured correctly

**Phase 2D: GUI integration**
1. Add API client module to GUI
2. Create hooks for fetching from backend
3. Test visualization with API-sourced data
4. Verify all existing features work (details panel, clusters, suggestions)

**Validation:**
- Run analysis from both file and Neo4j sources
- Compare JSON outputs (should be identical)
- Verify GUI displays correctly with API backend
- Test full workflow: Edit graph in Neo4j → re-analyze → GUI updates

## Design Decisions

### 1. Authentication: Backend Proxy ✓

**Decision:** Use backend proxy between GUI and Neo4j.

**Rationale:**
- Data is hosted online (Neo4j Aura), requires proper authentication
- Credentials must stay server-side, not exposed in browser
- Enables future features: caching, rate limiting, user auth
- Small implementation cost (50-100 lines of Express.js/FastAPI)

**Implementation:** Create lightweight Node.js or Python backend that queries Neo4j and exposes REST API to GUI.

### 2. Analysis Versioning: Latest Only ✓

**Decision:** Store only the most recent analysis, overwrite on each run.

**Rationale:**
- Simpler queries (no timestamp filtering needed)
- Fixed storage requirements
- Sufficient for current use case (single-user tool)
- Can add versioning later if temporal analysis needed

**Implementation:** Each analysis write overwrites previous data for that graph.

### 3. Incremental Updates: Not Needed ✓

**Decision:** Always run full analysis, no incremental updates.

**Rationale:**
- Current scale (30-50 people) makes full re-analysis fast (<5 seconds)
- Simpler code, easier to debug
- Avoids complexity of dependency tracking
- Can optimize later if performance becomes issue

**Implementation:** No special handling needed, standard re-analysis workflow.

### 4. Single Ego Graph Model ✓

**Decision:** System supports exactly ONE ego graph (the user's own graph).

**Rationale:**
- Aligns with "epistemic ego graph" philosophy: you own your graph
- Massive simplification: no need for graph names or multi-graph logic
- Easier to find: just query for `is_focal: true` node
- File system: graph data lives directly in `data/ego_graphs/` (no subdirectory)
- Neo4j: no `graph_name` property needed anywhere

**Breaking change from Phase 1:** Need to remove `graph_name` from schema and simplify queries.

**Implementation changes needed:**
- Remove `graph_name` parameter from all functions
- File system: move `data/ego_graphs/fronx/*` → `data/ego_graphs/*`
- Neo4j schema: remove `graph_name` property from all nodes
- Queries simplified: `MATCH (focal:Person {is_focal: true})` (no graph_name filter)

## References

- Neo4j documentation already created: [docs/NEO4J_BACKEND.md](NEO4J_BACKEND.md)
- Existing analysis code: [src/semantic_flow.py](../src/semantic_flow.py)
- Existing GUI loader: [gui/src/lib/egoGraphLoader.ts](../gui/src/lib/egoGraphLoader.ts)
- Neo4j storage implementation: [src/neo4j_storage.py](../src/neo4j_storage.py)
