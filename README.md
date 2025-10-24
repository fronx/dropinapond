# Drop in a Pond

A **semantic network navigation system** for understanding and strategically navigating your social/professional network using geometric reasoning over embedding spaces.

**Distributed and privacy-preserving**: each person runs local computations without revealing their full network to others.

## Quick Start

```bash
# Install dependencies
uv sync

# Run analysis on example ego graph
uv run python src/ego_ops.py fronx
```

This analyzes the [example ego graph](fixtures/ego_graphs/fronx.json) and outputs all navigation metrics.

## Documentation

- **[Documentation Index](docs/INDEX.md)** - Start here for full overview
- **[Vision & User Experience](docs/VISION.md)** - What is this and what's it like to use?
- **[Architecture](docs/ARCHITECTURE.md)** - How does it work internally?
- **[Distributed Protocol](docs/DISTRIBUTED.md)** - Privacy model & federation
- **[Implementation Guide](docs/IMPLEMENTATION.md)** - Building and extending the system

## What It Does

The system helps you answer questions like:

- Which clusters exist in my network?
- Who could bridge me to new communities?
- How should I frame my message to reach different groups?
- Where is the novelty - who offers perspectives I don't yet understand?

It combines **spectral graph theory**, **statistical learning**, and **differential geometry** to provide actionable navigation advice.

## Core Metrics

1. **Ego picture**: Cluster detection, overlap scores, attention entropy
2. **Public legibility (R²_in)**: How well neighbors can reconstruct your position
3. **Subjective attunement (R²_out)**: How well you can reconstruct neighbors' positions
4. **Heat-residual novelty**: Topological distance using diffusion geometry
5. **Translation vectors**: Semantic shifts between clusters
6. **Orientation scores**: Composite metric for choosing next interactions

See [Architecture](docs/ARCHITECTURE.md) for detailed explanations with code examples.

## Ego Graph Format

Define your network in `fixtures/ego_graphs/<name>.json`:

```json
{
  "focal_node": "F",
  "nodes": [
    {
      "id": "F",
      "name": "Your Name",
      "embedding": [0.8, 0.2, -0.1, 0.3, 0.5],
      "keyphrases": {
        "topic1": 1.0,
        "topic2": 0.9
      }
    },
    {
      "id": "neighbor1",
      "name": "Neighbor Name",
      "embedding": [0.4, 0.6, 0.1, -0.2, 0.3],
      "keyphrases": {
        "topic3": 1.0
      }
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

**Edge types**:
- `potential`: Semantic alignment (auto-computed from embeddings if not provided)
- `actual`: Real interaction strength
  - `past`: Historical interactions (0-1)
  - `present`: Current interactions (0-1)
  - `future`: Planned interactions (0-1)

The gap between `potential` and `actual` reveals latent opportunities.

## Dependencies

- Python ≥3.9
- numpy ≥1.24
- networkx ≥3.0
- scipy ≥1.10
- scikit-learn ≥1.2

Managed with [uv](https://github.com/astral-sh/uv) for fast, reliable package management.

## Project Status

**Current (v0.1)**: Core metrics working, example fixture, command-line analysis

**Next**: Conversational interface (Claude-based), embedding pipeline, temporal dynamics, feedback loops

See [Implementation Guide](docs/IMPLEMENTATION.md) for roadmap and contributing info.
