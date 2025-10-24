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
# (looks for fixtures/ego_graphs/<graph_name>.json)
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

**Continuous semantic fields**: The system uses phrase-level embeddings as the primary representation (not single vectors per person). Operates on kernel-weighted neighborhoods instead of discrete clusters. Clusters are computed on-demand for visualization only, never stored.

**Temporal dynamics**: Phrase weights and edge strengths decay exponentially (τ ≈ 40 days). Re-mentioning a phrase bumps its weight back up, creating a "living graph" that naturally forgets dormant topics.

### Six Navigation Metrics

All metrics operate on continuous semantic fields using kernel methods:

1. **Semantic landscape picture**: Kernel-based density and soft neighborhoods (Gaussian kernels)
2. **Public legibility (R²_in)**: Kernel-weighted reconstruction of your semantic field by neighbors
3. **Subjective attunement (R²_out)**: How well you reconstruct neighbors' fields (with legibility threshold)
4. **Heat-residual novelty**: Topological distance using continuous diffusion geometry
5. **Semantic gradients**: Local field translation (not discrete centroid differences)
6. **Orientation scores**: Composite metric for choosing next interactions

### Code Organization

**src/ego_ops.py** (20KB, ~680 lines): Core module containing all navigation metrics
- Lines 1-50: Data structures (`EgoData` dataclass, JSON loading)
- Lines 51-150: Utility functions (cosine similarity, normalization, R² metrics)
- Lines 151-250: Semantic landscape picture (cluster detection + overlap/attention)
- Lines 251-320: Public legibility (ridge regression)
- Lines 321-450: Subjective attunement (includes gated rank-2 variant at 440-449)
- Lines 451-550: Heat-residual novelty (diffusion on graph Laplacian)
- Lines 551-620: Translation vectors (centroid differences)
- Lines 621-680: Orientation scores (composite metric)
- Lines 681-end: Command-line runner

**src/translation_hints.py** (1.6KB): Finds lexical bridges between people's vocabularies using phrase-level semantic alignment

**fixtures/ego_graphs/*.json**: Example ego graph data files

### Data Format (v0.2 - Phrase-Level with ChromaDB)

Ego graphs are JSON files with this structure (embeddings stored separately in ChromaDB):

```json
{
  "version": "0.2",
  "focal_node": "F",
  "nodes": [
    {
      "id": "F",
      "name": "Your Name",
      "is_self": true,
      "phrases": [
        {
          "text": "topic phrase",
          "weight": 1.0,
          "last_updated": "2025-10-24"
        }
      ]
    }
  ],
  "edges": [
    {
      "source": "F",
      "target": "neighbor1",
      "actual": 0.8,
      "potential": 0.65
    }
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
- Temporal decay not yet implemented (planned v0.3)

## Current Implementation Status

**✅ Works now (v0.2)**:
- All six navigation metrics with discrete clustering (working with real embeddings)
- Phrase-level semantic fields (ChromaDB integration)
- Sentence-transformer embeddings (`all-MiniLM-L6-v2`, 384 dims)
- JSON schema validation
- Example fixture with 8 people, realistic semantic clusters
- Command-line analysis tool
- Keyphrase translation hints
- Backward compatibility with v0.1 format

**🔜 In progress (current sprint)**:
- Conversational interface (Claude-based ego graph builder)

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

**Continuous over discrete**: Avoid premature hardening of semantic structure. Clusters create artificial boundaries and feedback loops. Keep representation continuous, compute structure on-demand.

**Living memory**: Phrase weights and edge strengths decay exponentially. Re-mentioning bumps weight back up. The graph represents a "living present," not an archive.

## Adding New Metrics

1. Define computation function in `src/ego_ops.py`:
   ```python
   def compute_new_metric(ego_data: EgoData, clusters: dict) -> dict:
       """Docstring with clear explanation"""
       # Computation here
       return results
   ```

2. Integrate in main pipeline (lines 681-end)

3. Optionally incorporate into orientation scores (lines 621-680)

## Mathematical Dependencies

The system combines:
- **Spectral graph theory**: Laplacians, heat kernels, diffusion distance
- **Statistical learning**: Ridge regression, R² metrics (scikit-learn)
- **Information theory**: Shannon entropy for attention distribution
- **Differential geometry**: Embeddings as manifolds, translation as tangent vectors
- **Kernel methods**: Gaussian kernels for soft neighborhoods

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
