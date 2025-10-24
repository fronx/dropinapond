# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Drop in a Pond is a semantic network navigation system that uses geometric reasoning over embedding spaces to help users understand and strategically navigate their social/professional networks. It's distributed and privacy-preserving - each person maintains their own "epistemic ego graph" with predictions of neighbors' semantic fields.

## Common Commands

### Development

```bash
# Install dependencies (uses uv for package management)
uv sync

# Run analysis on example ego graph
uv run python src/ego_ops.py fronx

# Run analysis on a different ego graph
uv run python src/ego_ops.py <graph_name>
# (looks for data/ego_graphs/<graph_name>/ directory)
```

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

**src/ego_ops.py**: Navigation metrics and analysis
- Utility functions (cosine similarity, normalization, R² metrics)
- Semantic landscape picture: `ego_clusters()` for cluster detection
- Public legibility: `public_legibility_r2()` for ridge regression reconstruction
- Subjective attunement: `subjective_attunement_r2()` (includes gated rank-2 variant)
- Heat-residual novelty: `heat_residual_norm()` for diffusion on graph Laplacian
- Translation vectors: `pocket_centroid()` for semantic centroid differences
- Orientation scores: `orientation_score_breakdowns()` for composite metric
- Command-line runner

**src/translation_hints.py**: Finds lexical bridges between people's vocabularies using phrase-level semantic alignment

**src/embeddings.py**: ChromaDB integration for phrase embeddings

**data/ego_graphs/**: Ego graph data in modular directory format
- Each ego graph is a directory: `name/`
- Files are split for better editability and git tracking

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

## Adding New Metrics

1. Define computation function in `src/ego_ops.py`:
   ```python
   def compute_new_metric(ego_data: EgoData, clusters: dict) -> dict:
       """Docstring with clear explanation"""
       # Computation here
       return results
   ```

2. Integrate in main pipeline (see `if __name__ == "__main__"` block at end of file)

3. Optionally incorporate into orientation scores (see `orientation_score_breakdowns()` function)

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
