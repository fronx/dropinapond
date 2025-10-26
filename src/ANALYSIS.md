# Semantic–Structural Flow Analysis

> **File:** `src/semantic_flow.py`  
> **Purpose:** Compute a blended semantic–structural model of an ego network and export metrics for visualization in the existing EgoGraph UI.

---

## 1. Concept

Traditional social-graph analysis treats edges as **factual connections** — *who interacts with whom* and *how often*.  
But human networks also contain **semantic structure** — *who thinks or talks about similar things*.

The **Semantic–Structural Flow** analysis combines both:

\[
W = \alpha S + (1 - \alpha) A
\]

where

- \( S \) = **structural weights** from `edges.json` (actual relationships)  
- \( A \) = **semantic affinities** between nodes, derived from their phrase embeddings  
- \( \alpha \in [0,1] \) = blending factor (`--alpha` CLI argument)

This yields an **effective weighted graph** `W` that expresses both the *existing topology* (what edges exist) and *how semantically conductive* those edges are.

The resulting graph supports:  
- **Diffusion simulation** (how information or “attention energy” would flow),  
- **Community detection** (emergent clusters of semantic-social resonance),  
- **Connection suggestions** (potential edges not yet present but semantically strong).

---

## 2. Processing Stages

### 2.1 Load ego graph  
Reads:  
- `/data/ego_graphs/<name>/edges.json` (directed structural edges)  
- Each person’s phrases and embeddings (via ChromaDB through `embeddings.get_embedding_service()`).

### 2.2 Structural weights (`S`)  
Matrix \( S_{ij} \) is the normalized **connection strength** from node *i* to *j*  
(from `edge["actual"]`, clipped to `[0,1]`).

### 2.3 Semantic affinities (`A`)  
For every existing directed edge:  
1. Collect all phrase embeddings for the source and target nodes.  
2. Compute a **weighted cosine similarity** between their phrases:  
   \[
   A_{ij} = \text{mean}_{p \in i, q \in j}
       (\max(0, \cos(p,q))) \text{ weighted by phrase weights}
   \]  
3. Ignore weak similarities below `--cos-min` (default 0.2).

`A_{ij}` quantifies how semantically aligned two people’s “conceptual fields” are.

### 2.4 Blending: effective weights (`W`)  
The final weight matrix:

\[
W = \alpha S + (1 - \alpha) A
\]

- `--alpha 1.0`: purely structural (status quo)  
- `--alpha 0.0`: purely semantic (but still only along existing edges)  
- intermediate values mix both realities.

---

## 3. Derived Analyses

### 3.1 Diffusion Simulation

A **row-stochastic matrix**

\[
P = \frac{W}{\text{row-sum}(W)}
\]

is treated as a **Markov transition operator** — the probability that “attention” or “information” flows from one node to another in one step.

We export:  
- `t1` = \( P \)  
- `t2` = \( P^2 \)  
- `t3` = \( P^3 \)  

These represent 1-, 2-, and 3-step diffusion snapshots used by the *“Show Diffusion Flow”* overlay in the UI.

Interpretation:  
- Thick arrows → strong potential for information flow.  
- As *t* increases, diffusion shows how ideas could spread indirectly across the network.

### 3.2 Clustering (community detection)

We symmetrize the weights:

\[
W_{undirected} = W + W^\top
\]

and run **greedy modularity optimization** on that undirected graph.

- The algorithm knows nothing about embeddings directly.  
- It simply groups nodes where edge weights (now semantically informed) are dense.  
- The output clusters reflect **regions of high combined semantic and structural coherence**.

Interpretation:  
- **High-coherence cluster** → group of people who are both connected and semantically resonant.  
- **Boundary nodes** → people linking otherwise distinct semantic regions.

### 3.3 Connection suggestions

The script also computes **high-affinity non-edges**:  
- Find pairs of nodes without an existing edge.  
- Rank them by semantic affinity (using mean embeddings as a fast pre-filter).  
- Export the top `--suggest-k` per node as:

```json
"recommendations": { "semantic_suggestions": [ { "source": ..., "target": ..., "affinity": ... } ] }
```

Interpretation:  
- These are potential bridges — people who aren’t yet connected but whose ideas or focuses are highly compatible.

---

## 4. Output Structure

Example outline of `data/analyses/<name>_latest.json`:

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

## 5. Interpreting Results in the UI

| Visual Element   | Data Source                                | Meaning                                         |
|------------------|--------------------------------------------|-------------------------------------------------|
| Node color       | `metrics.clusters`                         | Membership in a semantically-structurally coherent region |
| Edge thickness / brightness | `metrics.layers.effective_edges`       | Blended strength of factual + semantic connection |
| Diffusion overlay| `metrics.kernel_neighborhoods.diffusion_heatmap` | Simulated potential information flow            |
| Suggestions (future) | `recommendations.semantic_suggestions`       | Potential new edges worth exploring              |

---

## 6. Tuning Parameters

| Parameter   | Effect                                | Typical Range  |
|-------------|-------------------------------------|----------------|
| --alpha     | Balance between structure and semantics | 0.4 – 0.8    |
| --cos-min   | Similarity threshold for phrase pairs | 0.15 – 0.3   |
| --suggest-k | How many high-affinity non-edges to suggest | 2 – 5     |

Low alpha → semantic clustering dominates.  
High alpha → original topology dominates.  
Use several runs to observe how clusters and diffusion change.

---

## 7. Key Takeaway

The clustering itself remains purely graph-based,  
but the graph it operates on has been reshaped by semantics.  
That’s why the communities now align with shared meaning rather than just shared history.

This analysis bridges two views of a network:  
- Topology — the skeleton of who is connected to whom.  
- Semantics — the resonance pattern of what people are about.

Blending them reveals where meaning can flow most naturally.

---

Would you like me to add one more section at the end showing **example before/after diagrams** (schematic graphs showing how semantic blending changes cluster boundaries)? That could make it even easier for future readers to grasp visually.