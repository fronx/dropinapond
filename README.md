# Drop in a Pond

A **semantic network navigation system** for understanding and strategically navigating your social/professional network using geometric reasoning over embedding spaces.

**Distributed and privacy-preserving**: each person runs local computations without revealing their full network to others.

## Quick Start

### Option 1: Conversational Interface (Recommended)

Build your ego graph through natural conversation with Claude:

```bash
# Install dependencies
uv sync

# In a Claude Code session, run:
/ego-session
```

Claude will guide you through building your network graph conversationally. See [Conversational Interface](docs/CONVERSATIONAL_INTERFACE.md) for details.

### Option 2: Analyze Existing Graph

```bash
# Install dependencies
uv sync

# Run analysis on example ego graph
uv run python src/ego_ops.py fronx
```

This analyzes the [example ego graph](data/ego_graphs/fronx.json) and outputs all navigation metrics.

## Documentation

- **[Documentation Index](docs/INDEX.md)** - Start here for full overview
- **[Conversational Interface](docs/CONVERSATIONAL_INTERFACE.md)** - Build your graph through natural dialogue
- **[Vision & User Experience](docs/VISION.md)** - What is this and what's it like to use?
- **[Architecture](docs/ARCHITECTURE.md)** - How does it work internally?
- **[Distributed Protocol](docs/DISTRIBUTED.md)** - Privacy model & federation
- **[Implementation Guide](docs/IMPLEMENTATION.md)** - Building and extending the system

## What It Does

The system helps you answer questions like:

- What semantic regions exist in my network landscape?
- Who could bridge me to new communities?
- How should I frame my message to reach different regions?
- Where is the novelty - who offers perspectives I don't yet understand?

It operates on **continuous semantic fields** using kernel methods, avoiding premature clustering. Structure emerges from analysis, not from stored categories.

It combines **spectral graph theory**, **statistical learning**, **kernel methods**, and **differential geometry** to provide actionable navigation advice.

## Core Metrics (Continuous Field Version)

1. **Semantic landscape picture**: Kernel-based density and soft neighborhoods (no hard clustering)
2. **Public legibility (R²_in)**: Kernel-weighted reconstruction of your semantic field
3. **Subjective attunement (R²_out)**: How well you reconstruct neighbors' fields (with legibility threshold)
4. **Heat-residual novelty**: Topological distance using continuous diffusion geometry
5. **Semantic gradients**: Local field translation (not discrete centroid differences)
6. **Orientation scores**: Composite metric for choosing next interactions

See [Architecture](docs/ARCHITECTURE.md) for detailed explanations and [Continuous Fields](docs/CONTINUOUS_FIELDS.md) for the architectural shift.

## Ego Graph Format (v0.2 - Continuous Fields)

Define your network in `data/ego_graphs/<name>.json`:

```json
{
  "focal_node": "F",
  "nodes": [
    {
      "id": "F",
      "name": "Your Name",
      "phrases": [
        {
          "text": "topic1",
          "embedding": [0.82, 0.31, -0.15, 0.41, 0.52],
          "weight": 1.0
        },
        {
          "text": "topic2",
          "embedding": [0.78, 0.25, -0.09, 0.38, 0.49],
          "weight": 0.9
        }
      ],
      "embedding": {
        "mean": [0.80, 0.28, -0.12, 0.40, 0.50]
      },
      "is_self": true
    },
    {
      "id": "neighbor1",
      "name": "Neighbor Name",
      "phrases": [
        {
          "text": "topic3",
          "embedding": [0.41, 0.61, 0.08, -0.19, 0.35],
          "weight": 1.0
        }
      ],
      "embedding": {
        "mean": [0.41, 0.61, 0.08, -0.19, 0.35]
      },
      "is_self": false,
      "prediction_confidence": 0.6
    }
  ],
  "edges": [
    {
      "source": "F",
      "target": "neighbor1",
      "actual": {"present": 0.8},
      "potential": 0.65
    }
  ]
}
```

**Key change**: `phrases` array is now the **primary representation**. Each phrase has its own embedding. Person-level `embedding.mean` is optional summary.

**Edge types**:
- `potential`: Semantic alignment (auto-computed from embeddings if not provided)
- `actual`: Real interaction strength
  - `past`: Historical interactions (0-1, decays slowly)
  - `present`: Current interactions (0-1, decays quickly)
  - `future`: Planned interactions (0-1, medium decay)

The gap between `potential` and `actual` reveals latent opportunities.

**Temporal model**: Phrase weights and edge strengths decay exponentially over time (τ ≈ 40 days). The graph represents a **living present**, not an archive. See [Temporal Dynamics](docs/TEMPORAL_DYNAMICS.md).

## Dependencies

- Python ≥3.9
- numpy ≥1.24
- networkx ≥3.0
- scipy ≥1.10
- scikit-learn ≥1.2

Managed with [uv](https://github.com/astral-sh/uv) for fast, reliable package management.

## Project Status

**Current (v0.1)**: Core metrics working with discrete clustering, example fixture, command-line analysis

**In progress (v0.2)**: Refactoring to continuous semantic fields (see [CONTINUOUS_FIELDS.md](docs/CONTINUOUS_FIELDS.md))

**Next**: Conversational interface (Claude-based), embedding pipeline, temporal dynamics, feedback loops

See [Implementation Guide](docs/IMPLEMENTATION.md) for roadmap and contributing info.

## Why Continuous Fields?

Previous approach used discrete clustering, which created artificial boundaries and feedback loops. New approach:
- Keeps phrase-level embeddings (richer representation)
- Uses kernel methods for soft neighborhoods
- Computes clusters on-demand for visualization only
- Preserves semantic continuity

See [Continuous Semantic Fields](docs/CONTINUOUS_FIELDS.md) for full explanation.
