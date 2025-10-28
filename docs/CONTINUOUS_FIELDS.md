# Phrase-Level Semantic Fields

## Why Phrase-Level Embeddings?

Instead of representing each person as a single embedding vector, the system uses **multiple weighted phrases**. This preserves semantic nuance and avoids premature aggregation.

### Single Vector Problems

If you represent someone as a single 384-dimensional vector:
- "machine learning" + "embodied cognition" + "regenerative agriculture" → one averaged blob
- Loses which specific topics they care about
- Can't identify phrase-level overlaps with others
- Forces premature compression

### Phrase-Level Solution

Store each phrase separately with its own embedding:

```json
{
  "id": "blake",
  "name": "Blake",
  "phrases": [
    {"text": "music cognition", "weight": 1.0},
    {"text": "pattern recognition", "weight": 0.8},
    {"text": "embodied presence", "weight": 0.7}
  ]
}
```

**Benefits**:
- Preserves semantic granularity
- Enables phrase-pair matching across the network
- Weighted phrases capture importance/recency
- Mean embedding computed on-demand when needed
- Human-readable data (no embedding arrays in JSON)

## Implementation

### Storage Architecture

**JSON files** (human-readable):
```
data/ego_graphs/fronx/
├── self.json              # Your phrases
└── connections/
    ├── blake.json         # Your model of Blake's phrases
    └── ...
```

Each file contains phrases as plain text + weights. NO embeddings stored in JSON.

**ChromaDB** (machine representation):
```
chroma_db/
└── fronx/                 # Collection per ego graph
    ├── phrase_1_hash      # Embedding for "music cognition"
    ├── phrase_2_hash      # Embedding for "pattern recognition"
    └── ...
```

Embeddings computed using `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dims) and cached in ChromaDB.

### Why This Split?

**JSON benefits**:
- Version control friendly (git diff shows phrase changes)
- Human-readable and editable
- No 384-dimensional arrays cluttering files
- Clean separation of semantics from representation

**ChromaDB benefits**:
- Fast similarity search
- Persistent caching (don't recompute unchanged phrases)
- Automatic sync (detects JSON changes via content hashing)
- Efficient nearest-neighbor queries

### Phrase Operations

**Computing affinity** between two people:

1. Load phrase embeddings from ChromaDB
2. Compute pairwise cosine similarity for all phrase pairs
3. Filter by threshold (default: cosine ≥ 0.25)
4. Weight by phrase importance
5. Aggregate to person-level affinity

This reveals **which specific phrases** connect two people, not just an overall similarity score.

**Mean embeddings** are computed on-demand for:
- Distance calculations (semantic gap between people)
- Visualization positioning
- Quick similarity filters (before phrase-level refinement)

## Current Usage

The analysis pipeline (`src/semantic_flow.py`) uses phrase-level embeddings for:

1. **Semantic affinity** (F): Phrase-pair matching with weighted aggregation
2. **Distance** (D): Mean embedding cosine distance
3. **Phrase contributions**: Which of your phrases drive affinity with each neighbor
4. **Standout phrases**: What makes each person unique in your network
5. **Connection suggestions**: High-affinity non-edges based on phrase overlap

See [SEMANTIC_FLOW_GUIDE.md](SEMANTIC_FLOW_GUIDE.md) for metric details.

## Design Trade-offs

**Computation cost**: Phrase-level operations are O(n_phrases × m_phrases) vs O(1) for single vectors. But with typical phrase counts (5-20 per person), this is negligible.

**Memory**: Storing embeddings separately in ChromaDB adds ~150 bytes per phrase. For a 100-person network with 10 phrases each = ~150KB. Acceptable.

**Complexity**: Two-tier storage (JSON + ChromaDB) adds architectural complexity. Worth it for the benefits of human-readable data.

## Future Directions

Ideas for extending phrase-level representations:

- **Phrase clustering**: Group similar phrases across the network to identify themes
- **Phrase evolution**: Track how phrases emerge, merge, and fade over time
- **Phrase attribution**: Which phrases came from which interactions
- **Cross-network phrase flow**: How ideas propagate between people

These are exploratory ideas, not committed roadmap items.

## Getting Started

To build an ego graph with phrase-level embeddings:

```bash
# Conversational interface
/ego-session

# Or edit JSON files directly in data/ego_graphs/<name>/
```

Embeddings are computed automatically when you run analysis:

```bash
uv run python src/semantic_flow.py <name>
```

See [CONVERSATIONAL_INTERFACE.md](CONVERSATIONAL_INTERFACE.md) for building graphs through dialogue.
