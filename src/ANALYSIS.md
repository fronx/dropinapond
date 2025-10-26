# Semantic–Structural Flow Analysis

This analysis combines two sources of information about your network: who you actually interact with and who thinks and talks about similar things as you.

To measure "thinking about similar things," we represent each person's semantic field as a cloud of points in a high-dimensional space organized by meaning—phrases that mean similar things sit closer together. Your semantic field is the collection of topics and ideas you care about, scattered through this space. When your cloud overlaps substantially with someone else's, you share conceptual territory.

We can measure this overlap numerically: for any pair of people, we compute how much their phrase clouds align. This gives us a semantic affinity score for each potential connection. We then blend these affinity scores with your actual interaction strengths to create a combined network.

The result is a graph where edge weights reflect both real relationships and conceptual alignment. This reveals where information is likely to flow, which clusters form around shared meaning, and where high-affinity connections don't exist yet.

---

## 1. Concept

### Intuition

Standard network analysis treats connections as binary facts: you're linked or you're not. But some people you rarely speak to might be working on exactly the problems you care about. Others you talk to frequently might occupy completely different semantic worlds.

This analysis addresses the gap by computing two separate weight matrices—one from actual edges, one from semantic affinity—and blending them. The result is a composite network that captures both who you interact with and who thinks like you.

### Mathematical Model

\[
W = \alpha S + (1 - \alpha) A
\]

where:
- \( S \) = structural weights from actual edges
- \( A \) = semantic affinities from phrase embeddings
- \( \alpha \in [0,1] \) = blending parameter

At \( \alpha = 1 \), you get pure topology. At \( \alpha = 0 \), pure semantics. In between, you get both.

---

## 2. Processing Stages

### Intuition

We construct \( S \) from your actual edges, \( A \) from pairwise phrase-level semantic similarity, and blend them according to \( \alpha \).

### Technical Details

#### 2.1 Load ego graph

Read structural edges from `edges.json` and fetch phrase embeddings from ChromaDB.

#### 2.2 Structural weights (\( S \))

\( S_{ij} \) is the edge's `actual` field, normalized and clipped to \([0,1]\).

#### 2.3 Semantic affinities (\( A \))

For each directed edge \( i \to j \):

1. Retrieve phrase embeddings for both nodes
2. Compute weighted mean of pairwise cosine similarities:

\[
A_{ij} = \text{mean}_{p \in i, q \in j} \left(\max(0, \cos(p,q))\right)
\]

weighted by phrase weights.

3. Filter out affinities below `--cos-min` (default 0.2)

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

Normalize \( W \) to make it row-stochastic:

\[
P = \frac{W}{\text{row-sum}(W)}
\]

\( P \) is a Markov transition matrix: \( P_{ij} \) is the probability that attention flows from \( i \) to \( j \) in one step.

We compute and export:
- \( P^1 \) (1-step diffusion)
- \( P^2 \) (2-step diffusion)
- \( P^3 \) (3-step diffusion)

The UI's "Show Diffusion Flow" overlay uses these matrices to visualize where information spreads—directly or through intermediaries.

#### 3.2 Clustering (community detection)

Symmetrize \( W \):

\[
W_{undirected} = W + W^\top
\]

Run greedy modularity maximization to partition nodes into clusters.

Clusters reflect combined semantic and structural coherence—people who are both connected and aligned. Boundary nodes sit between semantically distinct regions.

#### 3.3 Connection suggestions

Find pairs with no edge but high semantic affinity:

1. Compute affinities for all non-edges (using mean embeddings as a fast approximation)
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

All results are written to `data/analyses/<name>_latest.json`:

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

---

## 5. UI Interpretation

| Visual Element   | Data Source                                | Meaning                                         |
|------------------|--------------------------------------------|-------------------------------------------------|
| Node color       | `metrics.clusters`                         | Cluster membership (semantic + structural) |
| Edge thickness   | `metrics.layers.effective_edges`       | Blended weight (\( W \)) |
| Diffusion overlay| `metrics.kernel_neighborhoods.diffusion_heatmap` | \( P^t \) matrices showing attention flow |
| Suggestions      | `recommendations.semantic_suggestions`       | High-affinity non-edges              |

---

## 6. Tuning Parameters

| Parameter   | Effect                                | Typical Range  |
|-------------|-------------------------------------|----------------|
| `--alpha`     | Structure vs. semantics | 0.4–0.8    |
| `--cos-min`   | Min similarity threshold | 0.15–0.3   |
| `--suggest-k` | Max suggestions per node | 2–5     |

Low \( \alpha \) emphasizes semantic affinity. High \( \alpha \) preserves topology. Run multiple analyses to see how clusters shift.

---

## 7. Key Takeaway

Standard network analysis asks "who's connected?" This analysis asks "who's connected *and* who thinks alike?"

By blending structural and semantic layers, we see clusters that reflect actual communities of practice—not just interaction patterns—and we surface high-value connections that don't exist yet.