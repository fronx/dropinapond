# Distributed Protocol & Privacy Model

## The Markov Blanket Principle

The core insight: **You don't need direct access to someone's embedding to navigate your network effectively.** Instead, you maintain a **predictive model** of their embedding that gets refined through interaction.

This is inspired by the **Markov blanket** in active inference:
- Your knowledge of another person is mediated entirely through interactions
- You infer their state from observable evidence, not direct access
- Your prediction errors drive information-seeking behavior (who to talk to next)

## Local Operations

Each node runs independently with complete privacy:

### What You Store Locally

1. **Your own epistemic ego graph**:
   - Your embedding (ground truth - your actual position)
   - Your keyphrases (ground truth)
   - Your neighbors' predicted embeddings (your best guesses of their positions)
   - Your neighbors' keyphrases (observed through interaction)
   - Interaction history (past, present, future edge weights)
   - Prediction confidence scores per neighbor

2. **Derived metrics**:
   - Cluster assignments (computed from your epistemic graph)
   - Legibility/attunement scores (how well you and they can model each other)
   - Novelty residuals (topological distance in your interaction graph)
   - Translation vectors (semantic shifts between your perceived clusters)
   - Orientation scores (navigation guidance based on your local view)
   - **Potential edges** (projected mutual predictability - see below)

3. **Uncertainty estimates** (future):
   - Confidence intervals on predicted embeddings
   - Last update timestamp per neighbor
   - Interaction count (more interactions = higher confidence)

### Computing Potential Edges: Projected Mutual Predictability

A critical refinement: **potential connection strength is not just embedding cosine similarity**. Instead, it's **projected mutual predictability** - how well you and they can model each other's semantic fields.

**Computation**:
```python
def compute_potential(you_embedding, neighbor_predicted_embedding,
                     you_confidence, neighbor_confidence_in_you):
    # Raw semantic alignment
    cosine_sim = dot(you_embedding, neighbor_predicted_embedding)

    # Mutual predictability = alignment weighted by bidirectional confidence
    # High potential requires:
    # 1. Semantic alignment (high cosine similarity)
    # 2. You're confident about your model of them
    # 3. You estimate they're confident about their model of you
    mutual_predictability = cosine_sim * you_confidence * neighbor_confidence_in_you

    return mutual_predictability
```

**Why this matters**:
- Pure cosine similarity can be high even if you're uncertain about the neighbor
- Pure cosine similarity doesn't account for whether the connection is **mutually intelligible**
- Projected mutual predictability captures: "We're semantically aligned AND we can model each other"

**In practice**: After several interactions, if you both refine your models and confidence increases, potential edges strengthen even if raw embeddings stay similar. This captures the process of **mutual attunement** through interaction.

### What You Compute Locally

All six navigation metrics run on your machine:

1. Cluster detection (spectral clustering on your ego graph)
2. Public legibility (how well your neighbors could reconstruct you)
3. Subjective attunement (how well you can reconstruct them)
4. Heat-residual novelty (diffusion distance in your interaction graph)
5. Translation vectors (semantic shifts between your clusters)
6. Orientation scores (composite navigation metric)

**Privacy guarantee**: No external service sees your full graph. All computation is local.

## Predictive Models: How You Learn About Others

### Initial State (Before Any Interaction)

When you first hear about someone (through a mutual contact or introduction):

```json
{
  "id": "new_person",
  "name": "Alex",
  "embedding": null,
  "keyphrases": {},
  "prediction_confidence": 0.0,
  "last_updated": null
}
```

You have no information. Your predictive model is uniform over the embedding space.

### After First Interaction

You talk to Alex and learn about their interests:

```json
{
  "id": "alex",
  "name": "Alex",
  "embedding": [0.5, 0.3, 0.2, 0.1, 0.4],  // inferred from keyphrases
  "keyphrases": {
    "climate tech": 0.9,
    "renewable energy": 0.8,
    "policy advocacy": 0.6
  },
  "prediction_confidence": 0.3,
  "last_updated": "2025-01-15"
}
```

**How the embedding is inferred**:
1. You extract keyphrases from the conversation
2. You embed each keyphrase using your local sentence transformer
3. You compute a weighted average to get a person-level embedding
4. This is your **predicted embedding** - your best guess at their position in semantic space

**Confidence**: Low after one interaction, increases with more observations.

### After Multiple Interactions

You've now talked to Alex three times over several months:

```json
{
  "id": "alex",
  "name": "Alex",
  "embedding": [0.52, 0.31, 0.23, 0.09, 0.38],  // refined prediction
  "keyphrases": {
    "climate tech": 1.0,
    "renewable energy": 0.9,
    "policy advocacy": 0.7,
    "carbon markets": 0.8,       // newly observed
    "systems thinking": 0.6       // newly observed
  },
  "prediction_confidence": 0.7,
  "last_updated": "2025-03-22"
}
```

**Refinement process**:
1. New keyphrases emerge from additional conversations
2. Existing keyphrase weights are updated (Bayesian-style)
3. Embedding is recomputed from updated keyphrases
4. Confidence increases with interaction count

### Prediction Errors Drive Navigation

Key insight: **You want to interact with people where your prediction is most uncertain or most wrong.**

This is active inference:
- High uncertainty = high information gain potential
- Choose interactions that minimize future uncertainty about valuable parts of your network

The **orientation score** naturally captures this:
- Low attunement (R²_out) means poor prediction → high information gain
- High novelty residual means structurally distant → likely different from your model
- These components drive you toward exploratory interactions

## Inter-Node Protocol (Future)

When two nodes want to exchange information, they follow a **privacy-preserving protocol**.

### Scenario: You and Alex Both Use the System

**Your perspective**:
- You maintain a predicted embedding for Alex
- You compute navigation metrics using this prediction

**Alex's perspective**:
- Alex maintains a predicted embedding for you
- Alex computes navigation metrics using this prediction

**Neither of you shares your actual embedding directly.**

### Information Exchange After Interaction

After you have a conversation with Alex:

**You do**:
1. Update Alex's predicted embedding based on what you learned
2. Recompute navigation metrics
3. (Optional) Send a **prediction-error summary** to Alex

**Alex does**:
1. Update your predicted embedding based on the conversation
2. Recompute navigation metrics
3. (Optional) Receive your prediction-error summary and refine their model of you

### Prediction-Error Exchange Protocol (Future)

Instead of sharing raw embeddings, nodes exchange **prediction-error summaries** - compressed signals about how the interaction updated their models:

```json
{
  "interaction_id": "uuid",
  "from": "you",
  "to": "alex",
  "prediction_error": {
    "magnitude": 0.3,              // How much your model of Alex changed
    "direction_hint": "closer",     // "closer", "orthogonal", "further"
    "surprise_topics": [            // Topics you didn't expect from Alex
      "carbon markets",
      "systems thinking"
    ],
    "confirmed_topics": [           // Topics that matched your prediction
      "climate tech",
      "renewable energy"
    ]
  },
  "timestamp": "2025-03-22T14:30:00Z"
}
```

**Privacy properties**:
- You don't reveal your full embedding
- You don't reveal your full keyphrase distribution
- You only signal **relative changes** to your predictive model
- Alex can use this to calibrate their model of you (e.g., "they were surprised by X, so I should weight X higher in my representation of myself when interacting with them")

**What's NOT shared**:
- Your actual embedding (private)
- Alex's actual embedding (private)
- Your full ego graph (private)
- Your navigation metrics (private)

**Key innovation**: You're exchanging **epistemic updates** (how beliefs changed), not **ground truth** (actual states). This enables distributed learning without revealing raw data.

### Consensus Formation (Advanced)

In the future, two nodes could optionally engage in **active consensus**:

**Protocol**:
1. You share: "I think you're interested in X, Y, Z" (your prediction of Alex)
2. Alex responds: "Mostly correct, but I'm also into W"
3. You update your prediction
4. Repeat until convergence

This is like **federated learning without gradient sharing** - you're aligning predictive models through interaction, not by sharing raw data.

**Privacy properties**:
- Each node controls what they reveal
- No central aggregator
- Opt-in at every step
- Asymmetric disclosure (you can share more or less than the other person)

## Federated Navigation (Future)

### Multi-Hop Queries

**Scenario**: You want to reach someone outside your ego graph (e.g., an expert in field X you don't know personally).

**Current system**: You only see your immediate neighbors, can't navigate beyond them.

**Federated extension**:

1. You issue a query: "Find bridges to experts in X"
2. Your node computes orientation scores locally
3. Your node sends a **privacy-preserving query** to high-scoring neighbors:
   ```json
   {
     "query_id": "uuid",
     "target_profile": [0.7, 0.2, 0.1, 0.5, 0.3],  // anonymized embedding
     "hops_remaining": 2
   }
   ```
4. Your neighbors check their ego graphs locally
5. If they have a good match, they return:
   ```json
   {
     "query_id": "uuid",
     "match_quality": 0.85,
     "path_hint": "via Alex",  // or anonymized
     "contact_opt_in": true    // can I intro you?
   }
   ```
6. You get aggregated results without seeing your neighbors' full graphs

**Privacy properties**:
- Neighbors don't see who else you queried
- Neighbors don't see your full ego graph
- You don't see their full ego graphs
- Only final matches are revealed (with consent)

### Reputation & Trust (Future)

Over time, nodes accumulate **reputation** based on:
- Quality of introductions made
- Responsiveness to queries
- Reciprocity (do they help others who helped them?)

This could be stored as:
```json
{
  "reputation": {
    "alex": {
      "intro_quality": 0.8,    // how good were their intros?
      "response_rate": 0.9,    // how often do they respond?
      "reciprocity": 0.7       // do they return favors?
    }
  }
}
```

This is **local reputation** - your personal assessment, not a global score. Different people can have different views of the same node.

## Comparison to Centralized Systems

| Aspect | Centralized (e.g., LinkedIn) | Drop in a Pond |
|--------|------------------------------|----------------|
| **Graph storage** | Central database | Distributed, each node stores only their ego graph |
| **Privacy** | Company sees full graph | Nobody sees full graph |
| **Recommendations** | Server-side algorithm | Local computation |
| **Data portability** | Locked in | You own your data |
| **Algorithmic transparency** | Opaque | Open source, inspectable |
| **Surveillance** | Possible | Impossible (no central authority) |
| **Federation** | N/A | Opt-in peer-to-peer exchange |

## Active Inference Framework

The system implements **active inference** for social navigation:

1. **Beliefs**: Predicted embeddings for neighbors (your epistemic graph)
2. **Observations**: Interactions that provide evidence
3. **Actions**: Choosing who to talk to next (guided by orientation scores)
4. **Free energy**: Prediction error + uncertainty
5. **Objective**: Minimize free energy by seeking information-rich interactions

**Key insight**: You navigate your network by **actively gathering evidence** to refine your predictive models, not by maximizing utility directly.

This is cognitively natural - it's how humans already navigate social networks (we form impressions, update them through interaction, seek out people who surprise us).

### The Full Active Inference Loop (Future)

1. **Predict**: Before an interaction, form expectations about what the person will say/do based on your current model
2. **Interact**: Have the conversation, observe their actual responses
3. **Compute prediction error**: How much did they differ from your expectations?
4. **Update belief**: Refine your predicted embedding based on the error
5. **Update confidence**: Increase confidence if predictions were accurate, decrease if they were way off
6. **Recompute metrics**: Update orientation scores with new information
7. **Choose next action**: Based on updated orientation scores, decide who to engage next
8. **(Optional) Exchange prediction-error summary**: Send epistemic update to the other person

This loop runs continuously as you interact with your network, creating a **dynamic, self-correcting navigation system**.

## Implementation Status

**Current (v0.1)**:
- Local computation fully working
- No inter-node protocol yet (pure local analysis)

**Next steps**:
- Confidence tracking for predicted embeddings
- Interaction history logging
- Basic inter-node message format (JSON schema)
- Privacy-preserving query protocol

**Future**:
- Multi-hop navigation with privacy guarantees
- Federated consensus formation
- Reputation/trust scoring
- Temporal dynamics (drift, decay, trajectory prediction)
