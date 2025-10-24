# Drop in a Pond

Ego network analysis toolkit for measuring social orientation, legibility, and attunement.

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

```bash
# Install dependencies
uv pip install -e .
```

## Running the Example

```bash
# Run with a JSON fixture (e.g., fronx.json)
uv run python src/ego_ops.py fronx
```

## JSON Format

Define your ego network in `fixtures/ego_graphs/<name>.json`:

```json
{
  "F": {
    "name": "Your Name",
    "embedding": [0.25, 0.41, 0.52, 0.20, 0.10],
    "keyphrases": {
      "topic1": 0.9,
      "topic2": 0.8
    }
  },
  "neighbor1": {
    "name": "Neighbor Name",
    "embedding": [...],
    "keyphrases": {...}
  },
  "edges": [
    {"u": "F", "v": "neighbor1", "actual": 0.9, "potential": 0.7}
  ]
}
```

### Multi-dimensional Edges

Edges support multiple dimensions:
- **potential**: Semantic alignment (auto-derived from embeddings if not provided)
- **actual**: Real interaction strength (messages, meetings, etc.)
- **temporal**: past, present, future dimensions

If edges are not provided, the system automatically derives "potential" edges from embedding cosine similarity.

## What It Computes

1. **Ego picture**: overlaps, clustering, attention entropy
2. **Public legibility**: how well neighbors can reconstruct your position
3. **Subjective attunement**: how well you can reconstruct neighbors' positions
4. **Heat-residual novelty**: topological distance from pockets
5. **Orientation scores**: combined metric for choosing next interactions
