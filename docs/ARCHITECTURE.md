# Architecture

## The Epistemic Ego Graph

Each person maintains a JSON file representing their **epistemic ego network** - their own state (ground truth) plus their **predictive models** of immediate neighbors:

```json
{
  "focal_node": "F",
  "nodes": [
    {
      "id": "F",
      "name": "Fronx",
      "embedding": [0.8, 0.2, -0.1, 0.3, 0.5],  // Ground truth (your actual state)
      "keyphrases": {
        "audio embeddings": 1.0,
        "semantic search": 0.9,
        "navigation interfaces": 0.7
      },
      "is_self": true
    },
    {
      "id": "B",
      "name": "Blake",
      "embedding": [0.4, 0.6, 0.1, -0.2, 0.3],  // Predicted (your best guess)
      "keyphrases": {
        "music cognition": 1.0,
        "pattern recognition": 0.8
      },
      "is_self": false,
      "prediction_confidence": 0.6,
      "last_updated": "2025-03-15"
    }
  ],
  "edges": [
    {
      "source": "F",
      "target": "B",
      "actual": {"present": 0.3},
      "potential": 0.65
    }
  ]
}
```

### Key Components

**Nodes**: Your position (ground truth) and your models of others (predictions)
- `id`: Unique identifier
- `name`: Human-readable name
- `embedding`: Semantic vector (typically 5-100 dimensions)
  - If `is_self=true`: Ground truth (your actual position)
  - If `is_self=false`: Predicted (your best guess of their position)
- `keyphrases`: Weighted terms capturing interests/expertise
- `prediction_confidence`: (0-1) How certain you are about this prediction (only for neighbors)
- `last_updated`: When you last refined this prediction (only for neighbors)

**Critical insight**: This is an **epistemic graph**, not an objective one. Blake also maintains their own ego graph with their ground truth and their prediction of you. Your prediction of Blake may differ from Blake's ground truth. This asymmetry is fundamental to the privacy model.

**Edges**: Two types of relationships
- `actual`: Real interaction strength across time
  - `past`: Historical interactions (weight 0-1)
  - `present`: Current interactions (weight 0-1)
  - `future`: Planned/desired interactions (weight 0-1)
- `potential`: **Projected mutual predictability** - how well you and they can model each other's semantic field (not just cosine similarity)

The gap between `potential` and `actual` reveals **latent opportunities** - connections that would be mutually intelligible but haven't been actualized.

## The Six Navigation Metrics

The system computes multiple signals and combines them into **orientation scores** that guide your next interactions.

### 1. Ego Picture: Cluster Detection

**Goal**: Identify communities in your network

**Method**: Spectral clustering on the embedding similarity graph

```python
# Build similarity matrix from embeddings
S[i,j] = cosine_similarity(embedding_i, embedding_j)

# Apply threshold to get adjacency matrix
A[i,j] = 1 if S[i,j] > threshold else 0

# Detect clusters using graph Laplacian eigenvectors
clusters = spectral_clustering(A, n_clusters=3)

# For each cluster, compute:
overlap[c] = |neighbors_in_c| / |all_neighbors|
attention[c] = sum(interaction_weights_in_c) / total_interaction
attention_entropy = -sum(p * log(p)) for p in attention_distribution
```

**Output**:
- Cluster assignments for each neighbor
- Overlap scores (how much of your network is in each cluster)
- Attention distribution (where you actually spend time)
- Shannon entropy of attention (how focused vs. distributed)

**Interpretation**:
- High entropy = You spread attention across clusters (exploration)
- Low entropy = You focus on one cluster (exploitation)

### 2. Public Legibility (R²_in)

**Goal**: "How well can this cluster reconstruct my embedding from their own positions?"

**Method**: Ridge regression from cluster members to focal node

```python
# For each cluster C:
X = np.array([emb for neighbor in C for emb in [embeddings[neighbor]]])
y = embeddings[focal_node]

model = Ridge(alpha=0.1).fit(X, y)
R²_in[C] = model.score(X, y)
```

**Output**: R² scores per cluster (0 to 1)

**Interpretation**:
- High R²_in (>0.7) = You're predictable/understandable to this cluster
- Low R²_in (<0.3) = You're mysterious to them, potential for surprise
- Medium R²_in (0.3-0.7) = Partial alignment, room for nuance

**Use case**: If you want to be understood, engage with high-R²_in clusters. If you want to surprise/challenge, engage with low-R²_in clusters.

### 3. Subjective Attunement (R²_out)

**Goal**: "How well can I reconstruct this cluster's positions from my own embedding?"

**Method**: Ridge regression from focal node to each cluster member

```python
# For each neighbor n in cluster C:
X = embeddings[focal_node].reshape(1, -1)
y = embeddings[n]

model = Ridge(alpha=0.1).fit(X, y)
R²_out[n] = model.score(X, y)

# Average across cluster
R²_out[C] = mean(R²_out[n] for n in C)
```

**Output**: R² scores per cluster (0 to 1)

**Interpretation** (nuanced):
- High R²_out (>0.7) = You understand this cluster well
- **Low R²_out + High R²_in** (e.g., 0.2 out, 0.7 in) = **Exploratory opportunity** - they're intelligible to you but offer novelty worth investigating
- **Low R²_out + Low R²_in** (e.g., 0.2 out, 0.2 in) = **Noise/misalignment** - mutual unintelligibility, hard to bridge
- Medium R²_out (0.3-0.7) = Partial understanding, room for refinement

**Key refinement**: Low attunement is only valuable if there's a **minimum threshold of legibility**. Otherwise it's not "learning opportunity," it's just incompatibility. The system should check that R²_in > 0.3 before treating low R²_out as exploratory.

**Use case**: If you want to learn, engage with low-R²_out, high-R²_in clusters (they understand you, but you don't fully understand them yet). If you want mutual understanding, engage with high-R²_out, high-R²_in clusters.

**Advanced variant**: Gated rank-2 attunement
- If rank-1 model fails (R² < 0.1), try rank-2 model with residual direction
- This detects when a cluster has a "hidden dimension" orthogonal to your main interests
- Adds a second basis vector to capture structure not aligned with your primary embedding
- See `src/ego_ops.py:440-449` for implementation

### 4. Heat-Residual Novelty

**Goal**: Measure topological distance from a cluster using diffusion geometry

**Method**: Heat kernel smoothing on the interaction graph

```python
# Build graph Laplacian from interaction strengths
A = adjacency_matrix(interaction_graph)
D = degree_matrix(A)
L = D - A

# Solve heat equation: (I + t*L) * x = indicator
# where indicator = 1 for cluster members, 0 otherwise
heat_kernel = inv(I + t * L)
smoothed = heat_kernel @ cluster_indicator

# Novelty = what's left after smoothing
residual[neighbor] = abs(actual_position - smoothed_position)
```

**Output**: Residual scores per neighbor (higher = more novel)

**Interpretation**:
- High residual = Structurally distant from cluster, not reached by diffusion
- Low residual = Well-connected to cluster through multiple paths
- This captures "topological novelty" distinct from semantic novelty

**Parameters**:
- `t` (diffusion time): Controls how far heat spreads. Higher `t` = more smoothing.
- Typical values: `t = 0.1` to `t = 1.0`

**Use case**: If you want to reach structurally distant parts of your network, engage with high-residual neighbors.

**Implementation note on scalability**: The naive implementation `inv(I + t*L)` is O(n³) for matrix inversion. For larger ego graphs (>100 neighbors), use:
- **Truncated spectral decomposition**: Compute only top-k eigenvectors of L
- **Local heat propagation**: Iteratively diffuse from cluster without full inversion
- **Sparse solvers**: Use conjugate gradient if L is sparse
- Target: O(n²) or better for practical use

### 5. Translation Vectors

**Goal**: How to shift your message toward a target cluster

**Method**: Compute centroid difference and translate query embedding

```python
# Compute cluster centroids
centroid_A = mean(embeddings[n] for n in cluster_A)
centroid_B = mean(embeddings[n] for n in cluster_B)

# Translation = direction from A to B
translation_vector = centroid_B - centroid_A

# Shift your query
query_for_B = your_query + alpha * translation_vector

# Compute post-translation similarities
similarity[n] = cosine_similarity(query_for_B, embeddings[n])
```

**Output**:
- Translation vectors between clusters
- Post-translation similarity scores per neighbor

**Interpretation**:
- Translation vectors encode "semantic shifts" between communities
- High post-translation similarity = This neighbor is well-positioned to receive your shifted message
- This is how the system suggests **how to frame** your message for different audiences

**Use case**: "I want to talk about X with cluster B, but I'm currently in cluster A. How should I frame it?"

### 6. Orientation Score (Composite)

**Goal**: Combine all signals into a single navigation metric

**Method**: Weighted linear combination

```python
orientation[n] = (
    w1 * (1 - overlap[cluster(n)])        # Favor exploration (low overlap)
    + w2 * legibility[cluster(n)]          # Favor understandable clusters
    + w3 * (1 - attunement[cluster(n)])    # Favor learning opportunities
    + w4 * post_translation_similarity[n]  # Favor semantic alignment
    - w5 * instability_penalty[n]          # Penalize weak connections
)
```

**Default weights**: `w1=0.2, w2=0.3, w3=0.2, w4=0.2, w5=0.1`

**Output**: Orientation scores per neighbor (higher = better next target)

**Interpretation**: This is the **main navigation signal**. High-scoring neighbors are optimal next interactions given your current position and goals.

**Locality property**: All terms in this computation are derived from your **ego subgraph** (at most two hops from you):
- Overlap, legibility, attunement: computed from your immediate neighbors
- Translation vectors: derived from cluster centroids in your ego graph
- Instability: based on your direct interaction history

**This is fundamental to the Markov blanket principle**: You navigate using only local information, not global graph structure. You don't need to see the full network to make good decisions.

**Customization**: Users can adjust weights to change navigation strategy:
- Increase `w1` for more exploration
- Increase `w2` for more mutual understanding
- Increase `w3` for more learning (if legibility threshold is met)
- Increase `w4` for semantic alignment
- Increase `w5` for stability/reliability

## Keyphrase Translation Hints

Beyond abstract metrics, the system provides **lexical bridges** for practical communication.

**Goal**: Find phrases that bridge two people's vocabularies

**Method**: Score phrases by target-heavy weighting and semantic alignment

```python
def translation_hints(person_A_keyphrases, person_B_keyphrases, embedding_fn):
    hints = []

    for phrase_B, weight_B in person_B_keyphrases.items():
        # Embed the phrase
        emb_B = embedding_fn(phrase_B)

        # Compute semantic alignment with A's vocabulary
        alignment = max(
            cosine_similarity(emb_B, embedding_fn(phrase_A))
            for phrase_A in person_A_keyphrases
        )

        # Score = target weight * semantic alignment
        score = weight_B * alignment
        hints.append((phrase_B, score, alignment))

    return sorted(hints, key=lambda x: x[1], reverse=True)
```

**Output**: Ranked list of phrases to use when communicating with person B

**Example**:
```
To reach Blake (music cognition), use:
  1. "pattern recognition" (score: 0.82, alignment: 0.91)
  2. "perceptual features" (score: 0.76, alignment: 0.84)
  3. "temporal structure" (score: 0.68, alignment: 0.79)

Avoid:
  - "vector databases" (score: 0.15, alignment: 0.23)
  - "infrastructure" (score: 0.12, alignment: 0.18)
```

## Embedding Computation Pipeline

**Input**: Ego graph JSON with plain-text keyphrases

**Process**:
1. Extract all keyphrases across all nodes
2. Use sentence transformer to embed each phrase (e.g., `all-MiniLM-L6-v2`)
3. For each person, compute weighted average of keyphrase embeddings
4. Compute potential edges from embedding similarity (cosine > threshold)
5. Write updated JSON with embeddings and potential edges

**Future implementation**:
```bash
uv run python scripts/compute_embeddings.py \
  --input ego_graph.json \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --threshold 0.1
```

## Mathematical Foundations

The system combines multiple branches of mathematics:

- **Spectral graph theory**: Laplacians, heat kernels, diffusion distance
- **Statistical learning**: Ridge regression for reconstruction, R² metrics
- **Information theory**: Shannon entropy for attention distribution
- **Differential geometry**: Embeddings as manifolds, translation as tangent vectors
- **Linear algebra**: Cosine similarity, projections, rank analysis

It's not just a social graph tool - it's a **cognitive prosthetic** for navigating semantic space.

## Implementation Notes

**Current stack**:
- Data structures: Python dataclasses, NetworkX graphs
- Linear algebra: NumPy
- Machine learning: scikit-learn (ridge regression)
- Graph algorithms: NetworkX (spectral clustering, Laplacians)

**Performance considerations**:
- Current implementation loads full ego graph into memory
- Suitable for graphs with 10-1000 neighbors
- For larger graphs, would need incremental/streaming computation

**Extensibility**:
- New metrics can be added by implementing functions in `src/ego_ops.py`
- Custom orientation score weights can be passed via config
- Alternative clustering methods can swap in easily (modularity, Louvain, etc.)
