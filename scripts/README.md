# Scripts

Utility scripts for Drop in a Pond.

## Neo4j Backend Scripts

### import_to_neo4j.py

Import ego graphs from JSON files to Neo4j Aura.

**Usage:**
```bash
# Import with OpenAI embeddings (recommended)
python scripts/import_to_neo4j.py fronx

# Import with local sentence-transformers embeddings
python scripts/import_to_neo4j.py fronx --embedding-model local

# Replace existing graph in Neo4j
python scripts/import_to_neo4j.py fronx --clear-existing

# Use specific OpenAI model
python scripts/import_to_neo4j.py fronx --openai-model text-embedding-3-large
```

**Requirements:**
- Neo4j Aura credentials in environment variables (see `.env.example`)
- OpenAI API key (if using `--embedding-model openai`)

### analyze_from_neo4j.py

Run semantic flow analysis using Neo4j as the data source.

**Usage:**
```bash
# Run analysis on Neo4j-stored graph
python scripts/analyze_from_neo4j.py fronx

# Customize analysis parameters
python scripts/analyze_from_neo4j.py fronx --alpha 0.5 --cos-min 0.3
```

**Output:**
Same as file-based analysis - creates timestamped analysis files in `data/analyses/`.

**Note:** The analysis pipeline works identically regardless of storage backend (JSON files or Neo4j).
