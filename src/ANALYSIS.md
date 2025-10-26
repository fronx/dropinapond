# Semantic–Structural Flow Analysis

Imagine a network not just as dots connected by lines, but as a living web where ideas, meanings, and attention flow like currents through rivers. Each connection carries not only the fact of interaction but also a resonance of shared understanding and themes. This analysis captures that dynamic interplay — how information and influence ripple through the network’s structure, shaped by both who talks to whom and what they talk about. It reveals hidden pathways where meaning spreads naturally, and communities form not just by links, but by shared semantic harmony.

---

## 1. Concept

### Intuition

Traditional social network analysis looks at who interacts with whom — the concrete, factual connections that tie people together. But human relationships are richer than mere contact; they carry layers of meaning and shared ideas. Imagine two people who may rarely speak but think along similar lines, their minds resonating like tuning forks. Capturing this semantic harmony alongside structural ties lets us see not only the skeleton of the network but also its living pulse.

The Semantic–Structural Flow analysis blends these two realities into a single picture — a network where edges reflect both actual relationships and the strength of shared meaning. This combined view lets us simulate how attention flows, discover communities bound by both connection and concept, and suggest new links where ideas align but ties have yet to form.

### Mathematical Model

The combined network weight matrix \( W \) is a blend of:

\[
W = \alpha S + (1 - \alpha) A
\]

where

- \( S \) = structural weights from actual edges  
- \( A \) = semantic affinities derived from phrase embeddings  
- \( \alpha \in [0,1] \) = blending factor controlling the balance

This formula produces an effective weighted graph that expresses both existing topology and semantic conductance.

---

## 2. Processing Stages

### Intuition

To build this blended network, we start with the known connections — who talks to whom and how strongly. Then, we enrich those links with semantic information, measuring how closely aligned people's ideas are by comparing their phrase embeddings. By combining these, we create a nuanced map where edges carry both factual and conceptual weight.

### Technical Details

#### 2.1 Load ego graph

We read:

- Structural edges from `/data/ego_graphs/<name>/edges.json`  
- Each person’s phrases and embeddings via ChromaDB through `embeddings.get_embedding_service()`.

#### 2.2 Structural weights (\( S \))

Matrix \( S_{ij} \) represents the normalized connection strength from node *i* to *j*, based on `edge["actual"]`, clipped to the range \([0,1]\).

#### 2.3 Semantic affinities (\( A \))

For each existing directed edge:

1. Collect phrase embeddings for source and target nodes.  
2. Compute a weighted cosine similarity between their phrases:

\[
A_{ij} = \text{mean}_{p \in i, q \in j} \left(\max(0, \cos(p,q))\right) \text{ weighted by phrase weights}
\]

3. Ignore weak similarities below the threshold `--cos-min` (default 0.2).

This quantifies how semantically aligned two people’s conceptual fields are.

#### 2.4 Blending: effective weights (\( W \))

The final weight matrix is:

\[
W = \alpha S + (1 - \alpha) A
\]

- `--alpha 1.0` yields purely structural weights  
- `--alpha 0.0` yields purely semantic weights (along existing edges)  
- Intermediate values mix both.

---

## 3. Derived Analyses

### Intuition

With this blended network, we can explore how ideas and attention might flow, how communities emerge, and where new connections could form. The analysis simulates diffusion — like watching a drop of ink spread through water — revealing paths of influence. It also detects clusters where semantic and structural ties reinforce each other, and suggests promising new edges where semantic affinity is high but no link exists.

### Technical Details

#### 3.1 Diffusion Simulation

We convert the effective weights \( W \) into a row-stochastic matrix:

\[
P = \frac{W}{\text{row-sum}(W)}
\]

This matrix \( P \) acts as a Markov transition operator, representing the probability that attention flows from one node to another in a single step.

We export:

- \( t1 = P \)  
- \( t2 = P^2 \)  
- \( t3 = P^3 \)

These represent 1-, 2-, and 3-step diffusion snapshots, used by the “Show Diffusion Flow” overlay in the UI.

Interpretation:  
- Thick arrows indicate strong potential for information flow.  
- Increasing \( t \) shows how ideas spread indirectly across the network.

#### 3.2 Clustering (community detection)

We symmetrize the weights to create an undirected graph:

\[
W_{undirected} = W + W^\top
\]

Then, we run greedy modularity optimization on this graph.

- The algorithm groups nodes with dense, semantically informed connections.  
- Clusters represent regions of high combined semantic and structural coherence.

Interpretation:  
- High-coherence clusters are groups of people both connected and semantically resonant.  
- Boundary nodes link otherwise distinct semantic regions.

#### 3.3 Connection suggestions

We identify high-affinity non-edges by:

- Finding pairs of nodes without existing edges.  
- Ranking them by semantic affinity (using mean embeddings as a fast pre-filter).  
- Exporting the top `--suggest-k` per node as:

```json
"recommendations": { "semantic_suggestions": [ { "source": ..., "target": ..., "affinity": ... } ] }
```

Interpretation:  
- These suggestions highlight potential bridges — people not yet connected but with highly compatible ideas.

---

## 4. Output Structure

### Intuition

The output organizes the analysis into a structured format that the EgoGraph UI can use to visualize clusters, edge strengths, diffusion flows, and recommendations. It captures the blended reality of the network, ready for exploration.

### Technical Details

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

### Intuition

The UI translates these metrics into visual cues that help users grasp the network’s semantic-structural dynamics at a glance. Colors, thickness, and overlays reveal communities, connection strengths, and potential new links, making the abstract analysis tangible.

| Visual Element   | Data Source                                | Meaning                                         |
|------------------|--------------------------------------------|-------------------------------------------------|
| Node color       | `metrics.clusters`                         | Membership in a semantically-structurally coherent region |
| Edge thickness / brightness | `metrics.layers.effective_edges`       | Blended strength of factual + semantic connection |
| Diffusion overlay| `metrics.kernel_neighborhoods.diffusion_heatmap` | Simulated potential information flow            |
| Suggestions (future) | `recommendations.semantic_suggestions`       | Potential new edges worth exploring              |

---

## 6. Tuning Parameters

### Intuition

Adjusting parameters lets you explore different balances between structure and semantics, and control sensitivity thresholds. This tuning reveals how the network’s character shifts — from topology-driven to meaning-driven — and helps tailor the analysis to your needs.

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

### Intuition

This analysis reveals that communities are not just about who talks to whom, but also about who shares meaning. By reshaping the graph with semantic information, it uncovers clusters and pathways where ideas flow naturally — bridging the gap between social structure and shared understanding.

The network becomes a map of both connections and concepts, showing where meaning can travel most freely.

---

Would you like me to add one more section at the end showing **example before/after diagrams** (schematic graphs showing how semantic blending changes cluster boundaries)? That could make it even easier for future readers to grasp visually.