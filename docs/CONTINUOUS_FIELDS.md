# Continuous Semantic Fields: Architectural Shift

This document explains the shift from **discrete clustering** to **continuous semantic fields** - a fundamental refinement to keep the system fluid and prevent premature categorization.

## Motivation

### Problems with Discrete Clustering

The initial implementation computed single embeddings per person and used spectral clustering to identify "pockets":

```python
# Old approach
embedding_per_person = weighted_average(keyphrase_embeddings)
clusters = spectral_clustering(similarity_matrix, n_clusters=3)
```

**Issues**:
1. **Premature hardening**: Clustering creates artificial boundaries between facets that might actually converge or synergize
2. **Feedback loops**: Discrete clusters amplify their own boundaries over time as metrics reinforce cluster structure
3. **Loss of nuance**: A person's semantic field is richer than a single point or cluster label
4. **Discontinuities**: Navigation across cluster boundaries involves jumps, not smooth transitions

### Benefits of Continuous Fields

New approach: Keep all phrase-level embeddings, compute metrics using **kernel-weighted neighborhoods**:

```python
# New approach
person.phrases = [
    {text: "audio embeddings", embedding: [...], weight: 1.0},
    {text: "semantic search", embedding: [...], weight: 0.9},
    ...
]

# Compute kernel similarities (soft neighborhoods)
K[i,j] = exp(-||phrase_i - phrase_j||^2 / (2 * sigma^2))
```

**Advantages**:
1. **Preserves semantic continuity**: No hard boundaries, themes blend naturally
2. **Structure emerges on-demand**: Clusters are computed transiently when needed, not baked into data
3. **Respects field topology**: Metrics operate on local curvature, not global categories
4. **Prevents feedback loops**: No stored structure to amplify over time
5. **Aligns with cognitive metaphor**: Smooth landscape navigation, not island-hopping

## Data Structure Changes

### Before: Single Embedding Per Person

```json
{
  "id": "F",
  "name": "Fronx",
  "embedding": [0.8, 0.2, -0.1, 0.3, 0.5],
  "keyphrases": {
    "audio embeddings": 1.0,
    "semantic search": 0.9
  }
}
```

Problems:
- `embedding` is a lossy compression of rich semantic field
- No way to see phrase-level structure
- Forces single-point representation

### After: Phrase-Level Embeddings (Primary)

```json
{
  "id": "F",
  "name": "Fronx",
  "phrases": [
    {
      "text": "audio embeddings",
      "embedding": [0.82, 0.31, -0.15, 0.41, 0.52],
      "weight": 1.0
    },
    {
      "text": "semantic search",
      "embedding": [0.78, 0.25, -0.09, 0.38, 0.49],
      "weight": 0.9
    }
  ],
  "embedding": {
    "mean": [0.77, 0.30, -0.12, 0.36, 0.48],  // Optional summary
    "covariance": null                         // Optional uncertainty
  }
}
```

Key changes:
- `phrases` is now the **primary representation**
- `embedding.mean` is optional convenience (for quick similarity checks), not ground truth
- Can add `embedding.covariance` to represent uncertainty/spread of semantic field

## Metric Adaptations

### 1. Ego Picture → Semantic Landscape Picture

**Before**: Discrete clustering into pockets

```python
clusters = spectral_clustering(similarity_matrix, n_clusters=3)
overlap[c] = |neighbors_in_c| / |all_neighbors|
```

**After**: Kernel-based density and soft neighborhoods

```python
K[i,j] = exp(-||phrase_i - phrase_j||^2 / (2 * sigma^2))
density[i] = sum_j(K[i,j] * interaction_strength[j])

# If clusters needed for visualization, compute on-demand
temp_clusters = find_density_peaks(density)  # Don't store!
```

**Benefits**: Continuous density field, no forced categorization.

### 2. Public Legibility → Kernel-Weighted Reconstruction

**Before**: Ridge regression on cluster members

```python
# For cluster C:
X = embeddings[neighbors_in_C]
y = embedding[focal_node]
R²_in[C] = Ridge().fit(X, y).score(X, y)
```

**After**: Kernel-weighted ridge regression on all neighbors

```python
# Weight training samples by kernel similarity
K_weights[n] = K_person[focal_node, n]
sample_weights = [K_weights[n] * phrase.weight for all phrase pairs]

R²_in = Ridge().fit(X, y, sample_weight=sample_weights).score(X, y)
```

**Benefits**: Smooth weighting, no cluster boundary artifacts.

### 3. Translation Vectors → Semantic Gradients

**Before**: Centroid differences between clusters

```python
centroid_A = mean(embeddings[cluster_A])
centroid_B = mean(embeddings[cluster_B])
translation = centroid_B - centroid_A
```

**After**: Local gradient estimation in continuous field

```python
# Compute weighted gradient toward target region
gradient = sum(
    K[you, neighbor] * density[neighbor] * (neighbor.emb - your_emb)
    for neighbor in target_region
)
gradient = gradient / ||gradient||
```

**Benefits**:
- Respects local field topology (curvature)
- Works when "clusters" are ambiguous
- No discontinuities at boundaries

### 4. Heat-Residual Novelty → Continuous Diffusion

**Before**: Discrete Laplacian per cluster

```python
L_discrete = build_laplacian(cluster_adjacency)
heat_kernel = inv(I + t * L_discrete)
```

**After**: Kernel-weighted Laplacian on continuous field

```python
# Build Laplacian from kernel similarities
L_continuous = degree_matrix(K) - K
heat_kernel = inv(I + t * L_continuous)
```

**Benefits**: Smooth diffusion, respects fine-grained structure.

## On-Demand Clustering

**Philosophy**: Clusters are **emergent phenomena**, not fundamental structure. Compute them when needed for visualization or human interpretation, but never store them.

### When to Compute Clusters On-Demand

1. **Visualization**: User wants to see "pockets" in 2D projection
2. **High-level summaries**: "Your network has 3 main themes"
3. **Debugging**: Understanding metric behavior

### How to Compute Clusters On-Demand

```python
def get_clusters_for_display(ego_data, kernel_matrix, method='density_peaks'):
    """
    Compute clusters transiently for visualization.
    Call this when rendering UI, discard results immediately.
    """
    if method == 'density_peaks':
        density = kernel_matrix.sum(axis=1)
        peaks = find_local_maxima(density, threshold=0.1)
        assignments = assign_to_nearest_peak(kernel_matrix, peaks)
    elif method == 'spectral':
        assignments = spectral_clustering(kernel_matrix, n_clusters=3)

    return assignments  # Use immediately, don't store

# Usage
clusters = get_clusters_for_display(ego_data, K_matrix)
plot_with_cluster_colors(embeddings_2d, clusters)
# clusters go out of scope, never persisted
```

### Visualization: Smooth Density Heatmaps

Instead of discrete cluster colors, visualize the **continuous density field**:

```python
# Reduce to 2D for visualization
embeddings_2d = umap.UMAP().fit_transform(all_phrase_embeddings)

# Compute kernel density at each point in 2D space
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
density_grid = np.zeros_like(xx)

for i, j in product(range(100), range(100)):
    point = [xx[i,j], yy[i,j]]
    density_grid[i,j] = sum(exp(-||point - emb_2d||^2 / (2*sigma^2))
                           for emb_2d in embeddings_2d)

# Plot as smooth heatmap
plt.imshow(density_grid, cmap='viridis', interpolation='bilinear', alpha=0.5)
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='white', alpha=0.8)
```

Result: User sees a **continuous landscape** with ridges and valleys, not discrete colored regions.

## Implementation Roadmap

### Phase 1: Data Schema Migration

1. Update JSON schema to support `phrases` array
2. Make `embedding.mean` optional
3. Add migration script to convert old format to new format:
   ```python
   # For existing graphs with single embedding
   old_embedding = node['embedding']
   node['phrases'] = [
       {text: phrase, embedding: old_embedding, weight: weight}
       for phrase, weight in node['keyphrases'].items()
   ]
   node['embedding'] = {'mean': old_embedding, 'covariance': null}
   ```

### Phase 2: Kernel Infrastructure

1. Implement `gaussian_kernel()` utility
2. Implement `compute_kernel_matrix()` for person-level similarities
3. Add `sigma` parameter to config (default: 1.0)

### Phase 3: Metric Refactoring

1. **Semantic landscape picture**: Replace clustering with density computation
2. **Legibility/attunement**: Add kernel weighting to ridge regression
3. **Translation**: Replace centroids with gradient estimation
4. **Heat-residual**: Use continuous Laplacian

### Phase 4: On-Demand Clustering

1. Create `clustering_utils.py` module
2. Implement `compute_on_demand_clusters()` with multiple methods
3. Add visualization utilities with density heatmaps

### Phase 5: Update Documentation & Examples

1. Update example fixtures to use phrase-level embeddings
2. Revise README to explain continuous field representation
3. Update all metric descriptions in docs

## Backward Compatibility

**Breaking change**: v0.2 will not be backward compatible with v0.1 JSON format.

**Migration path**:
```bash
# Convert old ego graphs to new format
uv run python scripts/migrate_to_continuous_fields.py \
  --input data/ego_graphs/fronx_v01.json \
  --output data/ego_graphs/fronx_v02.json
```

**Alternative**: Keep v0.1 implementation in `src/ego_ops_legacy.py` for comparison.

## Open Questions

1. **Kernel bandwidth selection**: How to choose σ automatically? Options:
   - Fixed value (σ=1.0)
   - Adaptive per person (σ = median distance to k-nearest neighbors)
   - Learn from interaction feedback

2. **Phrase embedding quality**: Should we re-embed phrases from scratch or trust pre-computed embeddings?
   - Re-embedding ensures consistency
   - Pre-computed saves computation
   - Hybrid: re-embed only when keyphrases change

3. **Covariance estimation**: When/how to compute `embedding.covariance`?
   - Compute from phrase distribution
   - Update from prediction errors
   - Store as diagonal (variance only) or full matrix?

4. **Performance**: Kernel matrix is O(n²) in number of phrases. For large ego graphs:
   - Sparse kernels (threshold small similarities to zero)
   - Approximate nearest neighbors
   - Hierarchical summaries

## References

- **Kernel methods**: Gaussian kernel for soft similarity
- **Density estimation**: Kernel density estimation for continuous fields
- **Manifold learning**: UMAP/t-SNE for visualization
- **Spectral graph theory**: Continuous Laplacians for diffusion
- **Differential geometry**: Gradients on embedding manifolds

## Summary

This shift is both:
- A **philosophical simplification**: Don't force structure prematurely
- A **technical generalization**: Metrics operate on continuous fields, clusters emerge on-demand

It makes the system:
- More **robust** (no feedback loops amplifying boundaries)
- More **faithful to data** (preserves phrase-level nuance)
- More **extensible** (easier to add new phrase-based features)
- More **aligned with cognitive metaphor** (smooth landscape navigation)

**Next steps**: Implement Phase 1 (data schema migration), then incrementally refactor metrics in Phase 3.
