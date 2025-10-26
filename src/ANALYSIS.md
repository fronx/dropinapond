# Semantic–Structural Flow Analysis

This analysis combines two sources of information about your network: who you actually interact with and who thinks and talks about similar things as you.

To measure "thinking about similar things," we represent each person's interests as a collection of **phrases**. Each phrase gets converted into a numeric vector (embedding) that captures its meaning:

```json
{
  "consciousness exploration":     [0.23, -0.15, 0.89, "...", 0.42],
  "watching math videos for fun":  [0.11, 0.67, -0.22, "...", 0.38],
  "dinosaurs":                     [-0.45, 0.12, 0.56, "...", -0.19],
  "topological network analysis":  [0.34, -0.21, 0.73, "...", 0.51]
}
```

Semantically similar phrases end up with similar vectors (close together in this high-dimensional space). Your **semantic field** is the collection of all your phrase vectors—a cloud of points scattered through the embedding space.

When two people's **phrase clouds** overlap substantially—meaning they have many phrases that are semantically close—they share conceptual territory. We can measure this overlap numerically to compute a **semantic affinity** score between any pair of people. We then blend these affinity scores with your actual interaction strengths to create a combined network.

The result is a graph where edge weights reflect both real relationships and conceptual alignment. This reveals where information is likely to flow, which clusters form around shared meaning, and where high-affinity connections don't exist yet.

---

## 1. Mathematical Model

We need a way to assign a weight to each connection that combines both interaction strength and semantic alignment.

To do that, we represent your network as a matrix where each row is a person, each column is a person, and entry \( W_{ij} \) is the effective connection strength from person \( i \) to person \( j \).

We construct \( W \) by blending two matrices:

\[
W = \alpha S + (1 - \alpha) A
\]

where:
- \( W \) = blended weight matrix (effective connection strengths)
- \( S \) = structural weight matrix (from actual edges)
- \( A \) = semantic affinity matrix (from phrase embeddings)
- \( \alpha \in [0,1] \) = blending parameter

At \( \alpha = 1 \), you get pure topology. At \( \alpha = 0 \), pure semantics. In between, you get both.

---

## 2. Processing Stages

How do we actually build S, A, and W?
1. Construct \( S \) (structural matrix) from your actual interaction edges
2. Compute \( A \) (affinity matrix) from pairwise phrase-level semantic similarity
3. Blend them according to \( \alpha \)

### Technical Details

#### 2.1 Load ego graph

Read structural edges from `edges.json` and fetch phrase embeddings from ChromaDB.

#### 2.2 Structural weights (\( S \))

\( S_{ij} \) is the edge's `actual` field (defaults to 0.3 if missing), clipped to \([0,1]\).

#### 2.3 Semantic affinities (\( A \))

For the affinity matrix \( A \), we compute affinities only along existing directed edges \( i \to j \). This keeps the graph structure intact—we're re-weighting existing connections, not creating new ones yet. (Later, in connection suggestions, we'll look beyond existing edges.)

For each existing edge:

1. Retrieve phrase embeddings for both nodes
2. Compute all pairwise cosine similarities between phrases from \( i \) and phrases from \( j \)
3. Filter out pairs with similarity below `--cos-min` (default 0.2)
4. Take the weighted mean of the remaining similarities (each pair weighted by the product of its phrase weights)

This quantifies how much the two phrase clouds overlap.

#### 2.4 Blend (\( W \))

\[
W = \alpha S + (1 - \alpha) A
\]

- `--alpha 1.0`: ignore semantics, use pure topology
- `--alpha 0.0`: ignore topology, use pure semantic affinity
- Intermediate values combine both

---

## 3. Derived Analyses

### Intuition

With \( W \) in hand, we can simulate diffusion (attention flow), detect communities that reflect both structure and semantics, and identify high-affinity pairs with no existing edge.

### Technical Details

#### 3.1 Diffusion Simulation

Once we have these blended weights, we can model how attention ripples through the network—more likely to flow between people who are both connected and conceptually close.

To simulate this, we convert weights into probabilities. We normalize each row of \( W \) so it sums to 1:

\[
P = \frac{W}{\text{row-sum}(W)}
\]

Now \( P_{ij} \) represents the probability that attention flows from \( i \) to \( j \) in one step (row-stochastic matrix).

We compute and export:
- \( P^1 \) (1-step diffusion)
- \( P^2 \) (2-step diffusion)
- \( P^3 \) (3-step diffusion)

The UI's "Show Diffusion Flow" overlay uses these matrices to visualize where information spreads—directly or through intermediaries.

#### 3.2 Clustering (community detection)

Most clustering algorithms expect undirected graphs. We symmetrize the weight matrix by summing:

\[
W_{undirected} = W + W^\top
\]

Then run greedy modularity maximization to partition nodes into clusters.

Clusters reflect combined semantic and structural coherence—people who are both connected and aligned. Boundary nodes sit between semantically distinct regions.

#### 3.3 Connection suggestions

Find pairs with no edge but high semantic affinity:

1. For all non-edges, compute affinities between phrase clouds. (For speed, we use each person's average embedding rather than full phrase-by-phrase comparison)
2. Rank by affinity
3. Export top `--suggest-k` per node

Output format:
```json
"recommendations": {
  "semantic_suggestions": [
    { "source": "...", "target": "...", "affinity": 0.89 }
  ]
}
```

These are potential bridges—people whose ideas align but who aren't yet connected.

---

## 4. Output Structure

All results are written to `data/analyses/<name>_latest.json`. The structure separates raw inputs (structural edges, semantic affinities) from derived outputs (effective weights, clusters, diffusion):

```json
{
  "version": "semantic-flow-1.0",
  "parameters": { "alpha": 0.6, "cos_min": 0.2 },
  "metrics": {
    "clusters": [ ["a","b","c"], ["d","e"] ],
    "layers": {
      "structural_edges": { "a": {"b": 0.5} },
      "semantic_affinity": { "a": {"b": 0.7} },
      "effective_edges": { "a": {"b": 0.62} }
    },
    "kernel_neighborhoods": {
      "diffusion_heatmap": {
        "node_order": [],
        "matrices": { "t1": [], "t2": [], "t3": [] }
      }
    }
  },
  "recommendations": {
    "semantic_suggestions": [
      { "source": "fronx", "target": "alice", "affinity": 0.89 },
      { "source": "fronx", "target": "ben", "affinity": 0.73 }
    ]
  }
}
```

The `layers` object preserves both input matrices (`structural_edges`, `semantic_affinity`) and the blended result (`effective_edges`), making it easy to compare them. The `kernel_neighborhoods.diffusion_heatmap` holds the \( P^t \) matrices (historical naming from an earlier version of the system).

---

## 5. UI Interpretation

| Visual Element    | Data Source                                      | Meaning                                    |
| ----------------- | ------------------------------------------------ | ------------------------------------------ |
| Node color        | `metrics.clusters`                               | Cluster membership (semantic + structural) |
| Edge thickness    | `metrics.layers.effective_edges`                 | Blended weight (\( W \))                   |
| Diffusion overlay | `metrics.kernel_neighborhoods.diffusion_heatmap` | \( P^t \) matrices showing attention flow  |
| Suggestions       | `recommendations.semantic_suggestions`           | High-affinity non-edges                    |

---

## 6. Tuning Parameters

| Parameter     | Effect                   | Typical Range |
| ------------- | ------------------------ | ------------- |
| `--alpha`     | Structure vs. semantics  | 0.4–0.8       |
| `--cos-min`   | Min similarity threshold | 0.15–0.3      |
| `--suggest-k` | Max suggestions per node | 2–5           |

Low \( \alpha \) emphasizes semantic affinity. High \( \alpha \) preserves topology. Run multiple analyses to see how clusters shift.

---

## 7. Key Takeaway

Standard network analysis asks "who's connected?" This analysis asks "who's connected *and* who thinks alike?"

By blending structural and semantic layers, we see clusters that reflect actual communities of practice—not just interaction patterns—and we surface high-value connections that don't exist yet.