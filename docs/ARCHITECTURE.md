# Architecture

## The Epistemic Ego Graph: Continuous Semantic Fields

Each person maintains a modular directory structure representing their **epistemic ego network** - their own semantic field (ground truth) plus their **predictive models** of immediate neighbors' semantic fields.

**Directory Structure (v0.2)**:
```
data/ego_graphs/fronx/
├── metadata.json
├── self.json
├── connections/
│   ├── blake.json
│   └── ...
├── edges.json
└── contact_points.json
```

**self.json** (your ground truth semantic field):
```json
{
  "id": "fronx",
  "name": "Fronx",
  "phrases": [
    {
      "text": "audio embeddings",
      "weight": 1.0,
      "last_updated": "2025-10-15"
    },
    {
      "text": "semantic search",
      "weight": 0.9,
      "last_updated": "2025-10-12"
    },
    {
      "text": "navigation interfaces",
      "weight": 0.7,
      "last_updated": "2025-09-28"
    }
  ]
}
```

**connections/blake.json** (your prediction of Blake's semantic field):
```json
{
  "id": "blake",
  "name": "Blake",
  "phrases": [
    {
      "text": "music cognition",
      "weight": 1.0,
      "last_updated": "2025-03-15"
    },
    {
      "text": "pattern recognition",
      "weight": 0.8,
      "last_updated": "2025-03-15"
    }
  ],
  "capabilities": ["audio ML", "pattern recognition"],
  "availability": [
    {"date": "2025-10-24", "score": 0.7, "content": "Generally available"}
  ],
  "notes": [
    {"date": "2025-03-15", "content": "Met at music-tech conference"}
  ]
}
```

**edges.json**:
```json
[
  {
    "source": "fronx",
    "target": "blake",
    "actual": 0.3,
    "channels": ["email", "in_person"]
  }
]
```

**Embeddings** (stored separately in ChromaDB, not in JSON):
- Phrase embeddings computed via sentence-transformers (`all-MiniLM-L6-v2`, 384 dims)
- Cached in `./chroma_db/` directory (gitignored)
- Mean embeddings computed on-demand from weighted phrase embeddings
- Potential edge weights computed from embedding cosine similarity

### Key Components

**Nodes**: Phrase-level semantic fields
- `id`: Unique identifier
- `name`: Human-readable name
- `phrases`: Array of phrases (the **primary representation**)
  - `text`: The phrase itself (embeddings stored separately in ChromaDB)
  - `weight`: Current activation (0-1)
  - `last_updated`: Timestamp of most recent mention
- `capabilities`: (Optional) Skills/expertise the person can help with
- `availability`: (Optional) Timestamped availability observations
- `notes`: (Optional) Timestamped qualitative observations

**Embeddings** (separate from JSON):
- Stored in ChromaDB, not in JSON files
- Computed via sentence-transformers (all-MiniLM-L6-v2, 384 dimensions)
- Mean embeddings computed on-demand from weighted phrase embeddings
- Used for semantic similarity and navigation metrics

**Temporal dynamics** (planned for v0.3): Phrase weights will decay exponentially (`w *= exp(-Δt / τ)` with τ ≈ 40 days). Re-mentioning a phrase will bump its weight back up. This will create a **living semantic field** that naturally forgets dormant topics and remembers actively-used concepts. Currently timestamps are captured but decay is not yet implemented. See [TEMPORAL_DYNAMICS.md](TEMPORAL_DYNAMICS.md) for design.

**Critical insight**: This is an **epistemic graph** of **phrase-level semantic fields**, not discrete position vectors. Blake also maintains their own ego graph with their ground truth field and their prediction of your field. Your prediction of Blake's field may differ from Blake's ground truth field. This asymmetry is fundamental to the privacy model.

### Why Phrase-Level Embeddings?

**Previous approach (v0.1)**: Single embedding vector per person stored in JSON.

**Problems**:
- **Lossy compression**: A person's semantic field reduced to one point
- **Loss of nuance**: Multiple interests/facets collapsed together
- **Hard to update**: Changing one topic required recomputing entire vector

**Current approach (v0.2)**: Phrase-level embeddings stored in ChromaDB, mean computed on-demand.

**Benefits**:
- Preserves semantic richness (multiple phrases per person)
- Human-readable JSON (no embedding arrays cluttering files)
- Easy to update (add/remove phrases independently)
- Foundation for future continuous field operations

**Future direction (v0.3)**: Operate on **continuous kernel-weighted neighborhoods** instead of discrete clusters for navigation metrics.

**Future benefits** (when kernel methods are implemented):
- Preserves semantic continuity and expressiveness
- Structure emerges from ad-hoc analysis, not baked into stored data
- Aligns with cognitive metaphor: navigating a **smooth meaning landscape**, not hopping between islands
- Supports emergent detection of themes and bridges
- Maintains Markov blanket principle (local inference, no global categorization)

**Edges**: Relationship strength
- `actual`: Real interaction strength (scalar 0-1)
  - Represents current/recent interaction intensity
  - Set manually based on actual communication patterns
- `potential`: Semantic alignment (computed from embeddings)
  - Cosine similarity of mean embeddings
  - Computed automatically when loading ego graph
- `channels`: (Optional) Communication channels (e.g., "video_calls", "in_person", "email")

**Note**: Future versions may support temporal edge dimensions (past/present/future). See [TEMPORAL_DYNAMICS.md](TEMPORAL_DYNAMICS.md).

The gap between `potential` and `actual` reveals **latent opportunities** - connections that would be mutually intelligible but haven't been actualized.

## The Six Navigation Metrics

The system computes multiple signals and combines them into **orientation scores** that guide your next interactions.

**Current implementation (v0.2)**: Uses discrete clustering with phrase-level embeddings.

**Planned evolution (v0.3)**: Transition to continuous kernel-weighted neighborhoods. The mathematical descriptions below represent the target architecture.

### 1. Semantic Landscape Picture: Cluster Detection

**Goal**: Understand the topology of your network and attention distribution

**Current Method (v0.2)**: Discrete clustering using greedy modularity on mean embeddings

```python
# Build graph from neighbor subgraph
# Apply greedy modularity community detection
clusters = greedy_modularity_communities(neighbor_subgraph)

# Attention distribution (where you actually spend time)
attention[i] = interaction_strength[i] / sum_all(interaction_strength)
attention_entropy = -sum(p * log(p)) for p in attention_distribution
```

**Planned Method (v0.3)**: Kernel-based soft neighborhoods

```python
# For each pair of phrase embeddings, compute kernel similarity
K[i,j] = exp(-||phrase_i.embedding - phrase_j.embedding||^2 / (2 * sigma^2))

# Aggregate kernel weights to person level
K_person[i,j] = sum over phrases (
    K[phrase_i_k, phrase_j_l] * weight_i_k * weight_j_l
)

# Compute local density for each person
density[i] = sum_j(K_person[i,j] * interaction_strength[i,j])
```

**Output**:
- Kernel similarity matrix (continuous neighborhood structure)
- Local density per neighbor (centrality in the semantic landscape)
- Attention distribution (where you spend time)
- Shannon entropy of attention (how focused vs. distributed)

**Interpretation**:
- High density = Person is central to a semantic neighborhood (hub)
- Low density = Person is on the periphery (novel, exploratory)
- High entropy = You spread attention across the landscape (exploration)
- Low entropy = You focus on one region (exploitation)

**On-demand clustering**: If you need discrete "pockets" for visualization or analysis, compute them transiently:
```python
# Option 1: Kernel density peaks
pockets = find_local_maxima(density_field)

# Option 2: Spectral clustering on kernel matrix
pockets = spectral_clustering(K_person, n_clusters=k)
```

**Critical**: These clusters are **computed on demand, never stored**. The underlying representation stays continuous.

### 2. Public Legibility (R²_in): Kernel-Weighted Reconstruction

**Goal**: "How well can my semantic neighborhood reconstruct my field from their own positions?"

**Method**: Kernel-weighted ridge regression from neighbors to focal node

```python
# For each neighbor n, compute kernel weight based on semantic closeness
K_weights[n] = K_person[focal_node, n]

# Weighted regression: reconstruct your phrases from neighbors' phrases
X = np.array([phrase_emb for neighbor in neighbors
              for phrase_emb in neighbor.phrases])
y = np.array([phrase_emb for phrase_emb in focal_node.phrases])

# Weight each training sample by kernel similarity
sample_weights = np.array([K_weights[neighbor] * phrase.weight
                          for neighbor in neighbors
                          for phrase in neighbor.phrases])

model = Ridge(alpha=0.1).fit(X, y, sample_weight=sample_weights)
R²_in = model.score(X, y, sample_weight=sample_weights)
```

**Output**: R² score (0 to 1) - how well your semantic field is legible to your neighborhood

**Interpretation**:
- High R²_in (>0.7) = You're predictable/understandable from your neighborhood
- Low R²_in (<0.3) = You're mysterious, potential for surprise
- Medium R²_in (0.3-0.7) = Partial alignment, room for nuance

**Neighborhood-specific legibility** (optional): Compute R²_in for specific semantic regions:
```python
# Only use neighbors with high kernel weight in a specific semantic region
region_neighbors = [n for n in neighbors if K_person[focal, n] > threshold
                   and in_semantic_region(n, region_center)]
R²_in[region] = compute_weighted_regression(focal, region_neighbors)
```

**Use case**: If you want to be understood, engage with high-kernel-weight neighbors where R²_in is high. If you want to surprise, engage where R²_in is low.

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

### 5. Semantic Gradients: Local Field Translation

**Goal**: How to shift your message toward a target semantic region

**Method**: Estimate local gradient of the embedding field rather than discrete centroid differences

```python
# Given a target semantic region (defined by high-density neighbors)
target_neighbors = [n for n in neighbors if density[n] > threshold
                   and in_region(n, target_region)]

# Compute weighted gradient: direction of steepest ascent toward target
gradient = np.zeros_like(your_embedding)
for neighbor in target_neighbors:
    # Direction from you to neighbor
    direction = neighbor.embedding - your_embedding

    # Weight by kernel similarity and density
    weight = K_person[you, neighbor] * density[neighbor]

    gradient += weight * direction

# Normalize
gradient = gradient / np.linalg.norm(gradient)

# Translate your query smoothly along gradient
query_translated = your_query + alpha * gradient

# Compute post-translation kernel similarities
similarity[n] = exp(-||query_translated - n.embedding||^2 / (2 * sigma^2))
```

**Output**:
- Semantic gradient vector (direction toward target region)
- Post-translation kernel similarities per neighbor
- Recommended step size (alpha) based on field curvature

**Interpretation**:
- Gradients encode **local semantic shifts** in a continuous field
- High post-translation similarity = This neighbor is well-positioned to receive your shifted message
- This is how the system suggests **how to frame** your message for different semantic regions

**Advantages over discrete translation vectors**:
- Smooth, continuous navigation (no discontinuities at cluster boundaries)
- Respects local field topology (doesn't assume linear structure)
- Works even when "clusters" are ambiguous or overlapping

**Use case**: "I want to talk about X with people in semantic region B. How should I frame it?" The system computes the gradient from your current position toward region B and suggests phrase-level translations.

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
