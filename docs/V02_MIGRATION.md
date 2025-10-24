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
User data (modular)      ChromaDB (embeddings)
    ↓                            ↓
fronx/          ←────────────→  chroma_db/
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
from src.storage import load_ego_graph
from src.embeddings import get_embedding_service

# Initialize embedding service
service = get_embedding_service()

# Load v0.2 graph (directory path, not file)
ego = load_ego_graph("data/ego_graphs/fronx", embedding_service=service)
```

### Directory Structure

v0.2 uses a modular directory structure:

```
data/ego_graphs/fronx/
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

### Validation

Schema validation infrastructure exists in `src/validation.py`, though schema files are still being developed.

## Breaking Changes

**v0.2 is NOT backward compatible with v0.1.** Legacy v0.1 support was removed to keep the codebase lean.

If you have v0.1 data, you will need to manually convert it using the migration path below. This is a one-way upgrade.

## Manual Migration Path

**Note:** This is a manual process. No automated migration tool exists.

To convert v0.1 → v0.2:

1. Create directory structure: `data/ego_graphs/<name>/`
2. Create `metadata.json` with version "0.2" and format "modular"
3. Extract focal node into `self.json`
4. Create `connections/` directory
5. Split each connection into individual `connections/{id}.json` files
6. Extract keyphrases into `phrases` array for each person
7. Add `text`, `weight`, `last_updated` to each phrase
8. Remove `embedding` field from all nodes (embeddings will be in ChromaDB)
9. Update edge format: `u/v` → `source/target`, add `channels` if known
10. Create `edges.json` with all relationship edges
11. Create `contact_points.json` (optional, for temporal context)
12. Embeddings will be computed automatically on first load

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

# Delete a graph's embeddings (use graph name, not collection name)
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
