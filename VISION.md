# Drop in a Pond: Project Documentation

This file has been split into multiple focused documents for easier navigation.

## Quick Links

- **[Documentation Index](docs/INDEX.md)** - Start here
- **[Vision & User Experience](docs/VISION.md)** - What is this system?
- **[Architecture](docs/ARCHITECTURE.md)** - How does it work internally?
- **[Distributed Protocol](docs/DISTRIBUTED.md)** - Privacy model & federation
- **[Implementation Guide](docs/IMPLEMENTATION.md)** - Building and extending the system

## Quick Start

```bash
uv sync
uv run python src/ego_ops.py fronx
```

See [README.md](README.md) for more details.

## The User Experience

### Building Your Ego Graph

You don't manually create JSON files. Instead, you have **conversational sessions** where you talk about your week, your interests, people you've met, and past experiences. An AI assistant (like Claude) listens and builds a structured representation:

**Example interaction:**
```
You: "Had coffee with Sarah this week - we talked about her new podcast
      on urban design. She's really into walkable cities. I also caught
      up with Marcus from the old startup days. He's pivoting to climate
      tech now."

Claude: I've updated your ego graph:
        - Sarah: Added keyphrases "urban design", "podcasting", "walkable cities"
          Interaction: present=0.8 (recent coffee meeting)
        - Marcus: Updated from past=0.7 to present=0.4, future=0.6
          New keyphrases: "climate tech", "startup experience"
```

Behind the scenes, Claude:
1. Extracts structured information (names, topics, interaction context)
2. Stores it as plain text in your ego graph JSON
3. Runs a background process that computes semantic embeddings for all keyphrases
4. Updates the vector representations that power the navigation metrics

### Getting Navigation Insights

Once your graph is populated, you can ask:

```
You: "I want to learn more about regenerative agriculture. Who should I talk to?"

Claude: Based on your network analysis:
        1. Blake (orientation score: 0.73) - bridges your audio-tech cluster
           and organizational cluster. High novelty residual suggests they
           have perspectives you haven't encountered.

           Translation hint: Frame it as "sensing ecosystems" rather than
           "data analytics" - this aligns with their keyphrase space.

        2. Taylor (orientation score: 0.61) - in your organizational cluster
           but low attunement (R²_out=0.32) suggests you don't fully
           understand their position yet.
```

The system surfaces **who** to talk to, **why** they're valuable bridges, and **how** to frame your message for resonance.

### Privacy Model

Your ego graph lives locally. When you interact with someone else who also uses the system:

1. You maintain a **predictive model** of their embedding (what you think they're about)
2. After real interactions, you update your prediction based on what you learned
3. You never need their actual embedding - just your evolving best guess
4. This is the **Markov blanket**: you infer their state through interaction, not direct access

## How It Works Internally

### The Ego Graph Structure

Each person maintains a JSON file representing their **ego network**:

```json
{
  "focal_node": "F",
  "nodes": [
    {
      "id": "F",
      "name": "Fronx",
      "embedding": [0.8, 0.2, -0.1, 0.3, 0.5],
      "keyphrases": {
        "audio embeddings": 1.0,
        "semantic search": 0.9,
        "navigation interfaces": 0.7
      }
    },
    {
      "id": "B",
      "name": "Blake",
      "embedding": [0.4, 0.6, 0.1, -0.2, 0.3],
      "keyphrases": {
        "music cognition": 1.0,
        "pattern recognition": 0.8
      }
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

**Key components:**

- **Nodes**: People in your network with semantic embeddings (typically 5-100 dimensions)
- **Keyphrases**: Weighted terms that capture someone's interests/expertise
- **Edges**: Two types:
  - `actual`: Real interaction strength (past/present/future dimensions)
  - `potential`: Computed from embedding similarity - how well you *could* connect

### The Six Navigation Metrics

The system computes multiple signals and combines them into **orientation scores** that guide your next interactions:

#### 1. Ego Picture: Cluster Detection

Uses spectral clustering on the embedding similarity graph to identify communities:

```python
# Detect clusters using graph Laplacian eigenvectors
clusters = detect_clusters(similarity_graph)

# For each cluster, compute:
overlap_score = jaccard_similarity(neighbors_in_cluster, all_neighbors)
attention_entropy = shannon_entropy(interaction_distribution)
```

**Output**: Your network has 2-3 major "pockets" with varying density and overlap.

#### 2. Public Legibility (R²_in)

"How well can this cluster reconstruct my embedding from their own positions?"

Uses ridge regression:
```python
# For each cluster C:
X = embeddings_of_neighbors_in_C
y = your_embedding
model = RidgeRegression(alpha=0.1).fit(X, y)
R²_in[C] = model.score(X, y)
```

**Interpretation**:
- High R²_in = You're predictable/understandable to this cluster
- Low R²_in = You're mysterious to them, potential for surprise

#### 3. Subjective Attunement (R²_out)

"How well can I reconstruct this cluster's positions from my own embedding?"

Same method, reversed:
```python
# For each neighbor n in cluster C:
X = [your_embedding]
y = neighbor_n_embedding
R²_out[C] = average(regression_scores)
```

**Interpretation**:
- High R²_out = You understand this cluster well
- Low R²_out = They offer novelty, room to learn

#### 4. Heat-Residual Novelty

Uses diffusion geometry to measure topological distance:

```python
# Build graph Laplacian
L = compute_laplacian(interaction_graph)

# Solve heat equation: (I + t*L) * x = delta_cluster
heat_kernel = inv(I + t * L)
smoothed = heat_kernel @ cluster_indicator

# Novelty = what's left after smoothing
residual[neighbor] = abs(actual_position - smoothed_position)
```

**Interpretation**: High residual = structurally distant from cluster, not reached by diffusion.

#### 5. Translation Vectors

How to shift your message toward a target cluster:

```python
# Compute cluster centroids
centroid_A = mean(embeddings_in_cluster_A)
centroid_B = mean(embeddings_in_cluster_B)

# Translation = direction from A to B
translation_vector = centroid_B - centroid_A

# Shift your query
query_for_B = your_query + alpha * translation_vector
```

**Output**: Similarity scores after semantic shifting, used in orientation score.

#### 6. Orientation Score (Composite)

Combines all signals into a single navigation metric:

```python
orientation[neighbor] = (
    w1 * (1 - overlap[cluster])          # Favor exploration
    + w2 * legibility[cluster]            # Favor understandable clusters
    + w3 * (1 - attunement[cluster])      # Favor learning opportunities
    + w4 * post_translation_similarity    # Favor semantic alignment
    - w5 * instability_penalty            # Penalize weak connections
)
```

Higher scores = better next targets for interaction.

### Keyphrase Translation Hints

Beyond abstract metrics, the system provides **lexical bridges**:

```python
# Given two people A and B, find phrases that:
# 1. Are heavy in B's keyphrase distribution
# 2. Align semantically with A's vocabulary (via embeddings)

translation_hints = find_bridging_phrases(
    person_A_keyphrases,
    person_B_keyphrases,
    embedding_function
)
```

**Example output**:
```
To reach Blake (music cognition), frame your audio embedding work as:
  - "pattern recognition in sound" (similarity: 0.82)
  - "perceptual feature extraction" (similarity: 0.76)

Instead of:
  - "vector databases" (similarity: 0.23)
  - "semantic search infrastructure" (similarity: 0.31)
```

## Distributed Computation & Federation

### Local Operations

Each node runs independently:

1. **Ego graph maintenance**: Update your own representation and predictive models of neighbors
2. **Metric computation**: Calculate all six navigation metrics locally
3. **Privacy boundary**: Your full graph never leaves your machine

### Inter-node Protocol (Future)

When two nodes want to exchange information:

```
Node A: "I interacted with you about topic X"
Node B: [updates predictive model of A]
        "Here's my response about topic X"
Node A: [updates predictive model of B]
```

No embeddings are exchanged - only interaction evidence that lets each side refine their predictions.

### The Markov Blanket Principle

Your knowledge of another person is mediated entirely through interactions:

- **Before interaction**: You have a prior (predicted embedding)
- **During interaction**: You observe their responses, framings, interests
- **After interaction**: You update your posterior (refined embedding prediction)

This is **active inference** applied to social navigation: you choose interactions that minimize uncertainty about valuable parts of your network.

## Building Ego Graphs with AI

### The Conversational Pipeline

**Step 1: Natural conversation**
```
You tell Claude about your week, interests, people, projects.
Claude extracts structured information.
```

**Step 2: Plain-text representation**
```json
{
  "interactions": [
    {
      "person": "Sarah",
      "topics": ["urban design", "podcasting", "walkable cities"],
      "context": "coffee meeting, present",
      "strength": "strong"
    }
  ]
}
```

**Step 3: Embedding computation**
```bash
uv run python scripts/compute_embeddings.py --input ego_graph.json
```

This runs a background process that:
- Uses a sentence transformer to embed all keyphrases
- Averages keyphrase embeddings (weighted by importance) to get person embeddings
- Computes potential edges from embedding similarity
- Outputs the final ego graph ready for navigation

**Step 4: Query & navigate**
```
You ask Claude: "Who should I talk to about X?"
Claude runs ego_ops.py metrics and interprets results.
```

### Why Claude as Interface?

1. **Natural interaction**: You describe your network in your own words
2. **Contextual understanding**: Claude knows your history and can infer importance
3. **Semantic lifting**: Claude translates messy human descriptions into clean structured data
4. **Interpretation**: Claude translates mathematical metrics back into actionable advice

The human provides the **semantic content**.
The system provides the **geometric reasoning**.
Claude provides the **translation layer** between the two.

## Current Implementation Status

### What Works Now

- [x] Core ego graph data structure (JSON format)
- [x] All six navigation metrics implemented
- [x] Keyphrase translation hints
- [x] Example fixture (fronx.json) with 8 people, 2 clusters
- [x] Command-line runner for analysis
- [x] Dependencies: numpy, networkx, scipy, scikit-learn

### What's Next

- [ ] Conversational interface (Claude-based ego graph builder)
- [ ] Embedding computation pipeline (sentence transformers)
- [ ] Temporal dynamics (decay, trajectory prediction)
- [ ] Multi-hop navigation (bridges to distant clusters)
- [ ] Feedback loop (interaction updates predictive models)
- [ ] Inter-node protocol (privacy-preserving information exchange)

## Mathematical Foundations

The system combines:

- **Spectral graph theory**: Laplacians, heat kernels, diffusion distance
- **Statistical learning**: Ridge regression for reconstruction, R² metrics
- **Information theory**: Shannon entropy for attention distribution
- **Differential geometry**: Embeddings as manifolds, translation as tangent vectors
- **Active inference**: Markov blankets, predictive models updated through interaction

It's not just a social graph tool - it's a **cognitive prosthetic** for navigating semantic space.

## Design Philosophy

### Principles

1. **Privacy-first**: No centralized graph, no embedding leakage
2. **Cognitively natural**: Interface mirrors how you already think about your network
3. **Actionable**: Outputs are concrete (who to talk to, how to frame it)
4. **Composable**: Metrics can be weighted/combined for different navigation goals
5. **Extensible**: New signals (reputation, trust, temporal dynamics) can plug in

### Non-goals

- Not a social network platform (no central server)
- Not a CRM (doesn't optimize for business outcomes)
- Not a recommender system (doesn't push content)
- Not a surveillance tool (you only model people you actually interact with)

## Use Cases

### 1. Research Collaboration

"I want to explore applications of diffusion models to biology. Who in my network can bridge me there?"

System identifies people at intersection of ML and bio clusters, suggests framings.

### 2. Career Transitions

"I'm moving from software engineering to climate tech. Who should I talk to?"

System finds bridges between clusters, ranks by novelty and alignment.

### 3. Community Building

"I want to connect two groups I'm involved with. Who are the natural bridges?"

System identifies people with high betweenness and semantic overlap between clusters.

### 4. Learning Optimization

"I want to learn about X. Who can teach me most effectively?"

System balances: high expertise (keyphrase match), high attunement (they understand you), structural accessibility (reachable through existing connections).

## Why "Drop in a Pond"?

When you interact with someone, the effect **ripples outward** through the network. The system helps you:

- Choose where to drop your stone (who to engage)
- Predict how the ripples will spread (diffusion through clusters)
- Optimize for desired outcomes (reach, novelty, alignment)

You navigate by making **local perturbations** (individual conversations) that have **global effects** (shifting your position in the network).

---

## Getting Started

See [README.md](README.md) for installation and running the example.

For the conceptual foundations, see the "How It Works Internally" section above.

For understanding the metrics, see `src/ego_ops.py` (heavily commented).

For the future vision, see the "Distributed Computation & Federation" section above.
