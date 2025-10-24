# v0.2 Migration Guide

## What Changed

Drop in a Pond has migrated from **discrete single-vector embeddings (v0.1)** to **phrase-level semantic fields (v0.2)** with ChromaDB integration.

### Key Differences

**v0.1 (Legacy)**:
- Single embedding vector per person (lossy compression)
- Embeddings stored directly in JSON
- Keyphrases stored separately, not embedded

**v0.2 (Current)**:
- Multiple phrase embeddings per person (continuous field)
- Embeddings computed and cached in ChromaDB
- JSON stores only text + metadata (human-readable)
- Uses sentence-transformers (`all-MiniLM-L6-v2`)
- 384-dimensional embeddings

## New Architecture

```
User data (JSON)          ChromaDB (embeddings)
    ↓                            ↓
fronx.json  ←────────────→  chroma_db/
(text only)              (phrase vectors)
                                ↓
                         EmbeddingService
                                ↓
                          ego_ops.py
                         (all 6 metrics)
```

### Benefits

1. **Human-readable JSON**: No huge embedding arrays cluttering files
2. **Faster loading**: Embeddings computed once, cached forever
3. **Better semantic fidelity**: Phrase-level representation preserves nuance
4. **Easy model swapping**: Change embedding model without touching data
5. **Similarity search**: ChromaDB provides built-in vector search

## Using v0.2

### Installation

```bash
uv add chromadb sentence-transformers
```

### Loading Ego Graphs

```python
from src.ego_ops import load_ego_from_json
from src.embeddings import get_embedding_service

# Initialize embedding service
service = get_embedding_service()

# Load v0.2 graph
ego = load_ego_from_json("data/ego_graphs/fronx.json", embedding_service=service)
```

### JSON Format

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
          "text": "semantic navigation",
          "weight": 0.9,
          "last_updated": "2025-10-24"
        }
      ]
    }
  ],
  "edges": [
    {"source": "F", "target": "neighbor1", "actual": 0.8}
  ]
}
```

### Validation

```python
from src.validation import validate_ego_graph_file

is_valid, errors = validate_ego_graph_file("path/to/graph.json", version="0.2")
if not is_valid:
    print("Errors:", errors)
```

## Backward Compatibility

The system still supports v0.1 format:

```python
# v0.1 files load automatically without embedding_service
ego = load_ego_from_json("old_graph.json")
```

Format is auto-detected by checking for `"version": "0.2"` in JSON.

## Migration Path

To convert v0.1 → v0.2:

1. Extract keyphrases into `phrases` array
2. Add `text`, `weight`, `last_updated` to each phrase
3. Remove `embedding` field from nodes
4. Update edge format: `u/v` → `source/target`
5. Add `version: "0.2"` and `focal_node` fields
6. Embeddings will be computed automatically on first load

## ChromaDB Storage

- **Location**: `./chroma_db/` (gitignored)
- **Collections**: One per ego graph (`ego_graph_fronx`)
- **Persistence**: Automatic to disk
- **Phrase IDs**: `{node_id}:phrase_{index}`

### Managing ChromaDB

```python
service = get_embedding_service()

# List all graphs
graphs = service.list_collections()

# Delete a graph's embeddings
service.delete_collection("fronx")
```

## Performance

- **First load**: ~1-2s (computes embeddings)
- **Subsequent loads**: ~100ms (cached in ChromaDB)
- **Embedding model**: Runs on CPU, fast on M3 MacBook Air
- **Dimensions**: 384 (vs 5 in old fixture)

## Next Steps

With v0.2 in place, you can now:

1. ✅ Use real semantic similarity (not toy 5-d vectors)
2. ✅ Build conversational interface to collect phrases
3. 🔜 Implement temporal dynamics (decay + refresh)
4. 🔜 Add kernel-based continuous field operations
5. 🔜 Build distributed protocol for inter-node sync

See `docs/IMPLEMENTATION.md` for the full roadmap.
