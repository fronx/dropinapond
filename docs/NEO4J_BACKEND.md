# Neo4j Aura Backend for Drop in a Pond

This document describes how to use Neo4j Aura as an alternative storage backend for ego graphs in Drop in a Pond.

**Note:** This documentation reflects the **single-graph model** introduced in Phase 1B. The system now supports exactly one ego graph (your own epistemic graph), eliminating the need for graph names in all operations.

## Overview

Neo4j Aura provides a cloud-hosted graph database that offers several advantages for Drop in a Pond:

- **Graph-native storage**: Natural representation of ego graphs with nodes and relationships
- **Vector search**: Built-in vector indexes for semantic similarity queries
- **Scalability**: Better performance for larger ego graphs (100+ people)
- **Collaborative editing**: Potential for multi-user access (future feature)
- **Rich queries**: Cypher query language enables complex graph traversals

The Neo4j backend maintains full compatibility with the existing analysis pipeline by returning the same `EgoData` structure.

## Setup

### 1. Create a Neo4j Aura Instance

1. Sign up for [Neo4j Aura](https://neo4j.com/cloud/aura/) (free tier available)
2. Create a new database instance
3. Save your connection URI, username, and password

**Recommended instance settings:**
- **Size**: 8GB or larger (for vector optimization)
- **Version**: Neo4j 5.18+ (for relationship vector indexes)
- **Vector optimization**: Enable in instance settings (improves embedding performance)

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
# Neo4j Aura connection
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password-here

# OpenAI API (for embeddings)
OPENAI_API_KEY=sk-your-openai-api-key-here
```

### 3. Install Dependencies

The neo4j driver is already included in `pyproject.toml`:

```bash
uv sync
```

## Embedding Options

Neo4j backend supports two embedding strategies:

### Option 1: OpenAI Embeddings (Recommended)

**Advantages:**
- **Higher quality**: MTEB score 62.3 vs 56.3 for local models
- **Built into Neo4j**: Uses `genai.vector.encode()` procedures
- **Better semantic understanding**: Critical for professional/intellectual interests
- **Low cost**: ~$0.0004 per typical ego graph (100 people × 10 phrases each)

**Models available:**
- `text-embedding-3-small` (1536 dims, $0.02/1M tokens) - **recommended**
- `text-embedding-3-large` (3072 dims, $0.13/1M tokens) - highest quality

**Configuration:**
```python
save_ego_graph_to_neo4j(
    embedding_model='openai',
    openai_model='text-embedding-3-small',
    openai_dimensions=1536  # optional: reduce dimensions
)
```

### Option 2: Local Embeddings (Sentence-Transformers)

**Advantages:**
- **Free**: No API costs (compute only)
- **Privacy**: All computation local
- **Offline**: No internet required

**Model used:**
- `all-MiniLM-L6-v2` (384 dims, MTEB score 56.3)

**Configuration:**
```python
save_ego_graph_to_neo4j(
    embedding_model='local'
)
```

### Comparison

| Feature            | OpenAI    | Local       |
| ------------------ | --------- | ----------- |
| **Quality (MTEB)** | 62.3      | 56.3        |
| **Dimensions**     | 1536      | 384         |
| **Cost per graph** | ~$0.0004  | Free        |
| **Latency**        | 100-500ms | 10-100ms    |
| **Privacy**        | API call  | Fully local |
| **Offline**        | No        | Yes         |

**Recommendation**: Use OpenAI for production graphs where semantic quality matters. Use local for development/testing.

## Usage

### Importing Existing Graph to Neo4j

Migrate the JSON-based ego graph to Neo4j (single-graph model):

```bash
# Import with OpenAI embeddings (recommended)
python scripts/import_to_neo4j.py

# Import with local embeddings
python scripts/import_to_neo4j.py --embedding-model local

# Replace existing graph
python scripts/import_to_neo4j.py --clear-existing
```

The import script:
1. Loads the ego graph from `data/ego_graph/`
2. Computes/retrieves embeddings (based on chosen model)
3. Creates Neo4j nodes and relationships
4. Creates vector indexes for similarity search

### Running Analysis from Neo4j

Use the Neo4j backend as a drop-in replacement for file-based storage:

```bash
python scripts/analyze_from_neo4j.py
```

This:
1. Loads the ego graph from Neo4j
2. Returns an `EgoData` structure (same as file-based loader)
3. Runs the standard semantic flow analysis pipeline
4. Outputs to `data/analyses/analysis_latest.json` as usual

**No changes needed** to the analysis pipeline!

### Programmatic Usage

```python
from src.neo4j_storage import (
    load_ego_graph_from_neo4j,
    save_ego_graph_to_neo4j
)
from src.storage import load_ego_graph
from src.embeddings import get_embedding_service

# Load from Neo4j (single-graph model)
ego_data = load_ego_graph_from_neo4j()

# Load from files and save to Neo4j
ego_dir = Path('data/ego_graph')
embedding_service = get_embedding_service()
ego_data = load_ego_graph(ego_dir, embedding_service)
node_details = {...}  # Load from JSON files
save_ego_graph_to_neo4j(
    ego_data=ego_data,
    node_details=node_details,
    embedding_model='openai'
)
```

## Graph Schema

### Nodes

**Person (single-graph model):**
```cypher
{
  id: STRING,
  name: STRING,
  is_focal: BOOLEAN,
  embedding: LIST<FLOAT>  // 384 or 1536 dimensions
}
```

**Phrase:**
```cypher
{
  text: STRING,
  weight: FLOAT,
  last_updated: STRING (ISO timestamp),
  embedding: LIST<FLOAT>  // 384 or 1536 dimensions
}
```

**Event:**
```cypher
{
  id: STRING,
  type: STRING,  // 'past', 'present', or 'potential'
  date: STRING,
  content: STRING
}
```

### Relationships

- `(Person)-[:HAS_PHRASE]->(Phrase)` - Person's semantic field
- `(Person)-[:CONNECTED_TO {actual, channels}]->(Person)` - Network edges
- `(Person)-[:HAS_CAPABILITY {capability}]->(Person)` - Self-loop for skills
- `(Person)-[:HAS_NOTE {date, content}]->(Person)` - Self-loop for observations
- `(Person)-[:AVAILABILITY {date, score, content}]->(Person)` - Self-loop for availability
- `(Event)-[:INVOLVES]->(Person)` - Contact point participation

### Vector Indexes

Automatically created for semantic similarity search (single-graph model):

```cypher
// Person embeddings (for finding similar people)
CREATE VECTOR INDEX person_embedding
FOR (p:Person) ON p.embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,  // or 384 for local
    `vector.similarity_function`: 'cosine'
  }
}

// Phrase embeddings (for finding similar phrases)
CREATE VECTOR INDEX phrase_embedding
FOR (phrase:Phrase) ON phrase.embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }
}
```

## Example Queries

### Find Similar People

```cypher
// Find people most similar to Alice (single-graph model)
MATCH (alice:Person {name: 'Alice'})
CALL db.index.vector.queryNodes('person_embedding', 5, alice.embedding)
YIELD node AS similar_person, score
RETURN similar_person.name, score
ORDER BY score DESC
```

### Find Shared Phrases

```cypher
// Find phrases shared between Alice and Bob
MATCH (alice:Person {name: 'Alice'})-[:HAS_PHRASE]->(p1:Phrase)
MATCH (bob:Person {name: 'Bob'})-[:HAS_PHRASE]->(p2:Phrase)
WHERE p1.text = p2.text
RETURN p1.text, p1.weight AS alice_weight, p2.weight AS bob_weight
```

### Get Markov Blanket

```cypher
// Get immediate neighbors (Markov blanket) of focal node
MATCH (focal:Person {is_focal: true})
MATCH (focal)-[:CONNECTED_TO]-(neighbor:Person)
RETURN neighbor.name, neighbor.embedding
```

### Temporal Queries

```cypher
// Find recent interactions involving Alice
MATCH (alice:Person {name: 'Alice'})
      <-[:INVOLVES]-(event:Event)
WHERE event.type = 'past' AND event.date > '2024-01-01'
MATCH (event)-[:INVOLVES]->(other:Person)
WHERE other <> alice
RETURN event.date, event.content, collect(other.name) AS participants
ORDER BY event.date DESC
```

## Performance Considerations

### Vector Optimization

For best performance with embeddings:

1. **Enable vector optimization** in Neo4j Aura instance settings
2. **Choose appropriate instance size**:
   - 8GB: ~0.9M 768-D vectors
   - 32GB: ~7.3M 768-D vectors
   - For 1536-D OpenAI embeddings, divide capacity by 2

### Batch Operations

The implementation uses batch operations where possible:
- `genai.vector.encodeBatch()` for computing multiple embeddings
- Phrase embeddings computed in batches per person
- Person embeddings computed as weighted mean in Cypher

### Query Optimization

- Vector indexes are created automatically
- Single-graph model simplifies queries (no graph_name filtering needed)
- Use parameterized queries to leverage query caching

## Migration Strategy

To migrate from file-based to Neo4j storage (single-graph model):

1. **Import your ego graph**: Run the import script
   ```bash
   python scripts/import_to_neo4j.py --embedding-model openai
   ```

2. **Validate**: Run analysis on both backends and compare results
   ```bash
   uv run python src/semantic_flow.py  # File-based
   python scripts/analyze_from_neo4j.py  # Neo4j-based
   ```

3. **Compare outputs**: Verify both produce identical `analysis_latest.json` files
   ```bash
   diff data/analyses/analysis_latest.json <path-to-neo4j-output>
   ```

4. **Keep both backends**: JSON files remain source of truth, Neo4j as performance layer

## Future Enhancements

Potential improvements with Neo4j backend:

1. **Temporal dynamics**: Use Neo4j's date/time functions for exponential decay
2. **Collaborative editing**: Multi-user access with role-based permissions
3. **Real-time updates**: Subscribe to graph changes for live dashboard
4. **Advanced queries**: Leverage Cypher for complex graph traversals
5. **Graph algorithms**: Use Neo4j GDS library for centrality, pathfinding, etc.
6. **Vector similarity queries**: Direct semantic search without loading full graph

## Troubleshooting

### Connection Issues

```python
# Test connection
from neo4j import GraphDatabase
driver = GraphDatabase.driver(uri, auth=(username, password))
driver.verify_connectivity()
```

### Vector Index Not Found

Wait for indexes to populate after creation:
```cypher
SHOW INDEXES
YIELD name, state, populationPercent
WHERE name CONTAINS 'embedding'
```

### OpenAI API Errors

- Check `OPENAI_API_KEY` is set correctly
- Verify API quota/billing status
- Rate limit: 3,000 requests/minute (paid tier)

### Memory Issues

For large graphs (100+ people):
- Enable vector optimization in instance settings
- Consider upgrading instance size
- Use dimension reduction: `openai_dimensions=512`

## Cost Estimation

### Neo4j Aura

- **Free tier**: 1 instance, limited storage
- **Professional**: $65/month for 8GB
- **Enterprise**: Custom pricing

### OpenAI Embeddings

For typical ego graph (100 people × 10 phrases × 20 words/phrase = 20K tokens):
- text-embedding-3-small: **$0.0004 per graph**
- text-embedding-3-large: **$0.0026 per graph**

Annual cost for 10 graphs updated monthly:
- text-embedding-3-small: **$0.05/year** (negligible)
- Neo4j Aura 8GB: **$780/year**

**Total cost dominated by Neo4j hosting, not embeddings.**

## See Also

- [Neo4j Cypher Manual](https://neo4j.com/docs/cypher-manual/current/)
- [Neo4j Vector Indexes](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/)
- [Neo4j GenAI Procedures](https://neo4j.com/docs/cypher-manual/current/genai-integrations/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
