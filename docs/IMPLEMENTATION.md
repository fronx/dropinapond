# Implementation Guide

## Current Status

### What Works Now (v0.1)

- [x] Core ego graph data structure (JSON format)
- [x] All six navigation metrics implemented:
  - Ego picture (cluster detection)
  - Public legibility (R²_in)
  - Subjective attunement (R²_out, including gated rank-2 variant)
  - Heat-residual novelty
  - Translation vectors
  - Orientation scores
- [x] Keyphrase translation hints
- [x] Example fixture ([fronx.json](../data/ego_graphs/fronx.json)) with 8 people, 2 clusters
- [x] Command-line runner for analysis
- [x] Dependencies managed with `uv`

### What's Next

**Structural changes** (breaking):
- [ ] **Continuous semantic fields**: Refactor data schema to store phrase-level embeddings (primary representation)
- [ ] **Kernel-based neighborhoods**: Replace discrete clustering with Gaussian kernel similarities
- [ ] **Gradient-based translation**: Replace centroid-difference translation vectors with local field gradients
- [ ] **On-demand clustering**: Compute "pockets" transiently for visualization, never store

**New features**:
- [ ] Conversational interface (Claude-based ego graph builder)
- [ ] Embedding computation pipeline (sentence transformers integration)
- [ ] **Temporal dynamics**: Exponential phrase decay, edge weight decay, trajectory prediction (see [TEMPORAL_DYNAMICS.md](../docs/TEMPORAL_DYNAMICS.md))
- [ ] **Rank-2 attunement adapter**: Flexible attunement across pockets with residual directions
- [ ] **Active inference loop**: Predict → interact → update belief cycle
- [ ] **Prediction-error exchange protocol**: Cross-node handshake exchanging epistemic updates
- [ ] **Projected mutual predictability**: Refine potential edges beyond cosine similarity
- [ ] Confidence tracking with decay for predicted embeddings
- [ ] Multi-hop navigation (bridges to distant regions)
- [ ] Periodic checkpointing (optional, for longitudinal analysis)
- [ ] Web UI (optional, for visualization)

## Running the Example

### Setup

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repo
git clone <repo-url>
cd dropinapond

# Sync dependencies
uv sync
```

### Run Analysis

```bash
# Analyze the example ego graph
uv run python src/ego_ops.py fronx

# This will output:
# - Cluster assignments
# - Overlap scores per cluster
# - Attention distribution and entropy
# - Legibility scores (R²_in) per cluster
# - Attunement scores (R²_out) per cluster
# - Novelty residuals per neighbor
# - Translation vectors between clusters
# - Orientation scores per neighbor (sorted, highest first)
```

### Example Output

```
=== EGO PICTURE ===
Cluster 0: ['L', 'P', 'K'] (overlap: 0.43)
Cluster 1: ['S', 'T', 'R'] (overlap: 0.43)
Cluster 2: ['B'] (overlap: 0.14)
Attention entropy: 1.52

=== PUBLIC LEGIBILITY (R²_in) ===
Cluster 0: 0.78 (audio-tech cluster understands you well)
Cluster 1: 0.45 (organizational cluster has partial understanding)
Cluster 2: 0.12 (Blake is mysterious to you, or vice versa)

=== SUBJECTIVE ATTUNEMENT (R²_out) ===
Cluster 0: 0.82 (you understand audio-tech cluster well)
Cluster 1: 0.38 (room to learn about organizational cluster)
Cluster 2: 0.15 (Blake offers high novelty)

=== HEAT-RESIDUAL NOVELTY ===
B: 0.91 (Blake is topologically distant)
T: 0.67 (Taylor is moderately novel)
L: 0.12 (Lara is well-connected)
...

=== ORIENTATION SCORES ===
1. B (Blake): 0.73
2. T (Taylor): 0.61
3. S (Sam): 0.54
...
```

## Code Structure

### Core Module: [src/ego_ops.py](../src/ego_ops.py) - NEEDS REFACTORING

**Current implementation** (v0.1): Uses discrete clustering and single-vector-per-person representation.

**Target implementation** (v0.2): Will use continuous semantic fields with phrase-level embeddings.

**Main components** (current):

1. **Data structures** (lines 1-50):
   - `EgoData`: Dataclass holding the ego graph
   - JSON loading utilities
   - **TO REFACTOR**: Add support for phrase-level embeddings

2. **Utility functions** (lines 51-150):
   - `cosine_similarity()`: Compute cosine between vectors
   - `normalize()`: L2 normalization
   - `r_squared()`: R² metric for regression quality
   - **TO ADD**: `gaussian_kernel()`: K[i,j] = exp(-||x_i - x_j||² / 2σ²)

3. **Semantic landscape picture** (lines 151-250):
   - **CURRENT**: `compute_ego_picture()`: Cluster detection + overlap/attention metrics
   - **TO REFACTOR**: `compute_landscape_picture()`: Kernel similarities + density + attention
   - Replace spectral clustering with on-demand density peak detection

4. **Public legibility** (lines 251-320):
   - **CURRENT**: Ridge regression on cluster members
   - **TO REFACTOR**: Kernel-weighted ridge regression on continuous field

5. **Subjective attunement** (lines 321-450):
   - **CURRENT**: Reconstruct neighbors from focal node
   - **TO REFACTOR**: Kernel-weighted reconstruction with legibility threshold check
   - Keep gated rank-2 variant (lines 440-449)

6. **Heat-residual novelty** (lines 451-550):
   - **CURRENT**: Diffusion on discrete clusters
   - **TO REFACTOR**: Continuous diffusion with kernel-weighted Laplacian
   - Optimize with sparse solvers for scalability

7. **Semantic gradients** (lines 551-620):
   - **CURRENT**: `compute_translation_vectors()`: Centroid differences
   - **TO REFACTOR**: `compute_semantic_gradients()`: Local field gradients
   - Weighted by kernel similarity and density

8. **Orientation scores** (lines 621-680):
   - **CURRENT**: Composite score using cluster-based metrics
   - **TO REFACTOR**: Update to use kernel-based metrics (should be mostly compatible)

9. **Command-line runner** (lines 681-end):
   - Keep mostly unchanged, update output format

### On-Demand Clustering Utilities (NEW)

```python
# src/clustering_utils.py (to be created)

def compute_on_demand_clusters(kernel_matrix, method='density_peaks', k=None):
    """
    Compute clusters transiently from kernel matrix for visualization/analysis.
    Never store the cluster assignments - recompute as needed.

    Args:
        kernel_matrix: K[i,j] = exp(-||x_i - x_j||² / 2σ²)
        method: 'density_peaks' or 'spectral'
        k: number of clusters (if None, auto-detect)

    Returns:
        cluster_assignments: temporary mapping of nodes to cluster IDs
    """
    if method == 'density_peaks':
        density = kernel_matrix.sum(axis=1)
        peaks = find_local_maxima(density)
        assignments = assign_to_nearest_peak(kernel_matrix, peaks)
    elif method == 'spectral':
        assignments = spectral_clustering(kernel_matrix, n_clusters=k)

    return assignments  # Use immediately, don't store

def visualize_semantic_landscape(ego_data, kernel_matrix):
    """
    Create smooth 2D visualization using UMAP or t-SNE on phrase embeddings.
    Show density heatmap with kernel weights, not discrete cluster colors.
    """
    # Reduce dimensionality
    embeddings_2d = umap.UMAP().fit_transform(all_phrase_embeddings)

    # Compute density field
    density = compute_kernel_density(embeddings_2d, kernel_matrix)

    # Plot as continuous heatmap
    plt.imshow(density, cmap='viridis', interpolation='bilinear')
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)

    return fig
```

### Translation Hints: [src/translation_hints.py](../src/translation_hints.py)

**Purpose**: Find lexical bridges between two people's vocabularies

**Main function**:
```python
def translation_hints(
    person_A_keyphrases: dict,
    person_B_keyphrases: dict,
    embedding_fn=None  # optional, defaults to random
) -> list[tuple[str, float, float]]
```

**Returns**: List of (phrase, score, alignment) sorted by score

## Adding New Metrics

To add a new navigation metric:

1. **Define the computation function** in `src/ego_ops.py`:
```python
def compute_my_new_metric(ego_data: EgoData, clusters: dict) -> dict:
    """
    Compute some new metric per cluster or per neighbor.

    Args:
        ego_data: The ego graph
        clusters: Cluster assignments from compute_ego_picture()

    Returns:
        Dictionary mapping cluster_id or neighbor_id to metric value
    """
    results = {}
    # Your computation here
    return results
```

2. **Integrate into the main pipeline** (lines 681-end):
```python
# After computing existing metrics
my_metric = compute_my_new_metric(ego_data, clusters)
print("\n=== MY NEW METRIC ===")
for key, value in my_metric.items():
    print(f"{key}: {value:.2f}")
```

3. **Optionally incorporate into orientation scores** (lines 621-680):
```python
orientation[neighbor] = (
    # existing terms
    + w_new * my_metric[neighbor]
)
```

## Integrating Sentence Transformers

To compute embeddings from keyphrases automatically:

### Step 1: Add dependency

```toml
# In pyproject.toml
dependencies = [
    "numpy>=1.24,<2.0",
    "networkx>=3.0",
    "scipy>=1.10",
    "scikit-learn>=1.2",
    "sentence-transformers>=2.2",  # Add this
]
```

### Step 2: Create embedding script

```python
# scripts/compute_embeddings.py
import json
import numpy as np
from sentence_transformers import SentenceTransformer

def compute_embeddings(input_path: str, model_name: str, output_path: str):
    # Load ego graph
    with open(input_path) as f:
        data = json.load(f)

    # Load model
    model = SentenceTransformer(model_name)

    # For each node
    for node in data['nodes']:
        keyphrases = list(node['keyphrases'].keys())
        weights = np.array([node['keyphrases'][k] for k in keyphrases])

        # Embed keyphrases
        phrase_embeddings = model.encode(keyphrases)

        # Weighted average
        node_embedding = np.average(phrase_embeddings, axis=0, weights=weights)

        # Normalize
        node_embedding = node_embedding / np.linalg.norm(node_embedding)

        # Store
        node['embedding'] = node_embedding.tolist()

    # Compute potential edges
    for i, node_i in enumerate(data['nodes']):
        for node_j in data['nodes'][i+1:]:
            emb_i = np.array(node_i['embedding'])
            emb_j = np.array(node_j['embedding'])
            sim = np.dot(emb_i, emb_j)

            if sim > 0.1:  # threshold
                data['edges'].append({
                    'source': node_i['id'],
                    'target': node_j['id'],
                    'potential': float(sim)
                })

    # Save
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == '__main__':
    import sys
    compute_embeddings(
        input_path=sys.argv[1],
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        output_path=sys.argv[2]
    )
```

### Step 3: Use it

```bash
uv run python scripts/compute_embeddings.py \
  data/ego_graphs/my_graph_plaintext.json \
  data/ego_graphs/my_graph.json
```

## Building the Conversational Interface

The vision: users talk to Claude, which builds/updates their ego graph.

### Architecture

```
User <-> Claude (via Claude Code)
         |
         v
    Ego graph JSON (plain text + keyphrases)
         |
         v
    Embedding script (batch process)
         |
         v
    Ego graph JSON (with embeddings)
         |
         v
    Navigation metrics (ego_ops.py)
         |
         v
    Claude (interprets results for user)
```

### Implementation Plan

1. **Create a conversational session tracker**:
```python
# src/conversation_tracker.py
class ConversationTracker:
    def __init__(self, ego_graph_path: str):
        self.ego_graph_path = ego_graph_path
        self.load_graph()

    def load_graph(self):
        # Load existing graph or create new one
        pass

    def extract_update(self, user_message: str) -> dict:
        # Use Claude to parse user message into structured update
        # Could use function calling or structured output
        pass

    def apply_update(self, update: dict):
        # Update ego graph JSON
        pass

    def recompute_embeddings(self):
        # Call embedding script
        pass

    def get_navigation_advice(self, query: str) -> str:
        # Run ego_ops.py metrics
        # Format results for user
        pass
```

2. **Integrate with Claude Code**:
- User has conversational sessions
- Claude calls `ConversationTracker` methods as needed
- Background process watches for graph updates and recomputes embeddings
- User queries navigation system naturally in conversation

3. **Example flow**:
```
User: "Had coffee with Sarah this week, we talked about urban design"
Claude: [calls extract_update() -> applies update -> recomputes embeddings]
        "I've updated Sarah in your ego graph. Want to see your current network picture?"

User: "Yes"
Claude: [calls get_navigation_advice() with no specific query]
        "You have two main clusters: urban planning (Sarah, ...) and tech (...).
         Blake could be a bridge between them - orientation score 0.73."

User: "How should I frame my transit accessibility project for Sarah?"
Claude: [calls get_navigation_advice() with query about transit]
        "Use phrases like 'walkable neighborhoods' and 'public space design'
         rather than 'routing algorithms' - this aligns with her keyphrase space."
```

## Testing

### Unit Tests (Future)

```python
# tests/test_ego_ops.py
import pytest
from src.ego_ops import EgoData, compute_ego_picture, cosine_similarity

def test_cosine_similarity():
    v1 = [1, 0, 0]
    v2 = [0, 1, 0]
    assert cosine_similarity(v1, v2) == 0.0

    v3 = [1, 0, 0]
    v4 = [1, 0, 0]
    assert cosine_similarity(v3, v4) == 1.0

def test_cluster_detection():
    # Load fixture
    ego_data = EgoData.from_json('data/ego_graphs/fronx.json')

    # Compute clusters
    clusters, overlaps, attention = compute_ego_picture(ego_data, n_clusters=3)

    # Assertions
    assert len(clusters) == 3
    assert 'L' in clusters[0]  # Lara in audio-tech cluster
    # etc.
```

Run with:
```bash
uv run pytest tests/
```

### Integration Tests

Create additional fixtures with known properties:

```json
// data/ego_graphs/test_perfect_clusters.json
// Two perfectly separated clusters, no overlap
// Expected: overlap ≈ 0.5 for each, attention entropy ≈ 1.0
```

## Performance Optimization

Current implementation is suitable for ego graphs with 10-1000 neighbors. For larger graphs:

### 1. Sparse Matrix Operations

```python
# Instead of dense numpy arrays
from scipy.sparse import csr_matrix

# Build sparse adjacency matrix
A_sparse = csr_matrix((data, (row, col)), shape=(n, n))

# Sparse Laplacian
L_sparse = sparse_laplacian(A_sparse)
```

### 2. Approximate Clustering

```python
# For very large graphs, use approximate methods
from sklearn.cluster import MiniBatchKMeans

# Instead of spectral clustering
clusters = MiniBatchKMeans(n_clusters=3).fit_predict(embeddings)
```

### 3. Incremental Updates

Instead of recomputing all metrics on every change:

```python
# Track what changed
changed_nodes = ['A', 'B']

# Only recompute metrics affected by those nodes
# Most metrics are local (only depend on neighbors of changed nodes)
```

## Extending to Multi-Hop

Current system is ego-centric (1-hop). To navigate to distant parts of the network:

### Approach 1: Recursive Ego Graphs

```python
def find_path_to_target(
    ego_data: EgoData,
    target_embedding: np.ndarray,
    max_hops: int = 3
) -> list[str]:
    """
    Find path from focal node to target using recursive exploration.
    """
    # At each hop:
    # 1. Compute orientation scores toward target
    # 2. Choose highest-scoring neighbor
    # 3. Recursively explore from that neighbor
    pass
```

### Approach 2: Federated Query (see [DISTRIBUTED.md](DISTRIBUTED.md))

Nodes exchange privacy-preserving queries to search beyond their ego graphs.

## Deployment Options

### Option 1: Local CLI (Current)

User runs scripts locally, stores ego graph on their machine.

**Pros**: Maximum privacy, full control
**Cons**: Manual process, no real-time updates

### Option 2: Claude Code Integration (Planned)

User has conversational sessions with Claude, graph updates automatically.

**Pros**: Natural interface, automatic updates
**Cons**: Requires Claude access

### Option 3: Web App (Future)

Local-first web app (e.g., using PouchDB for local storage).

**Pros**: Visual interface, easier onboarding
**Cons**: More complex, potential privacy concerns if poorly implemented

### Option 4: Distributed Network (Long-term)

Peer-to-peer network of nodes with federated protocol.

**Pros**: Full vision realized, network effects
**Cons**: Complex protocol design, requires critical mass

## Contributing

To contribute:

1. Fork the repo
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code follows existing style (type hints, docstrings)
5. Submit PR with clear description

### Code Style

- Use type hints (Python 3.9+)
- Docstrings in NumPy format
- Max line length: 100 characters
- Use `black` for formatting (future)

### Documentation

When adding features:
- Update relevant docs in `docs/`
- Add examples to docstrings
- Update [README.md](../README.md) if user-facing

## Roadmap

### v0.2 (Next)
- Embedding computation script
- Confidence tracking
- Basic conversational interface prototype

### v0.3
- Temporal dynamics (edge decay, trajectory prediction)
- Feedback loop (interaction updates)
- Multi-hop navigation (local algorithm)

### v0.4
- Inter-node protocol (JSON schema)
- Privacy-preserving queries
- Federated consensus (basic version)

### v1.0
- Production-ready distributed system
- Web UI (optional)
- Full documentation
- Security audit
