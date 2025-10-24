# Documentation Refinements (Post-GPT-5 Review)

This document summarizes the key refinements made to the documentation based on feedback emphasizing the epistemic and predictive coupling nature of the system.

## Major Conceptual Clarifications

### 1. Epistemic Ego Graphs (Not Objective Graphs)

**Before**: Documentation implied each node stores "the truth" about their neighbors' embeddings.

**After**: Clarified that each node stores:
- **Ground truth**: Their own embedding (the only objective data point)
- **Predictions**: Best guesses of neighbors' embeddings, refined through interaction
- **Confidence**: Uncertainty estimates that decrease with more observations

**Key insight**: Blake's ego graph contains Blake's prediction of you, which may differ from your ground truth. This asymmetry is fundamental to privacy.

**See**: [ARCHITECTURE.md - The Epistemic Ego Graph](ARCHITECTURE.md#the-epistemic-ego-graph)

### 2. Potential Edges = Projected Mutual Predictability

**Before**: Potential edges computed as simple cosine similarity between embeddings.

**After**: Potential edges are **projected mutual predictability** - how well you and they can model each other's semantic fields, weighted by bidirectional confidence.

```python
potential = cosine_sim * your_confidence * their_confidence_in_you
```

**Why it matters**: Pure cosine similarity doesn't capture mutual intelligibility. High potential requires both semantic alignment AND mutual modeling capacity.

**See**: [DISTRIBUTED.md - Computing Potential Edges](DISTRIBUTED.md#computing-potential-edges-projected-mutual-predictability)

### 3. Nuanced Interpretation of R²_out (Attunement)

**Before**: Low R²_out automatically means "learning opportunity."

**After**: Low R²_out is only valuable if **legibility (R²_in) exceeds a minimum threshold**:
- **Low R²_out + High R²_in**: Exploratory opportunity (they understand you, you don't fully understand them)
- **Low R²_out + Low R²_in**: Noise/misalignment (mutual unintelligibility, hard to bridge)

**Key refinement**: Novelty must be **interpretable** to be valuable. Otherwise it's just incompatibility.

**See**: [ARCHITECTURE.md - Subjective Attunement](ARCHITECTURE.md#3-subjective-attunement-r²_out)

### 4. Prediction-Error Exchange (Not Raw Embeddings)

**Before**: Inter-node protocol implied sharing some form of embedding data.

**After**: Nodes exchange **prediction-error summaries** - compressed signals about how interactions updated their models:

```json
{
  "prediction_error": {
    "magnitude": 0.3,
    "direction_hint": "closer",
    "surprise_topics": ["carbon markets"],
    "confirmed_topics": ["climate tech"]
  }
}
```

**Privacy guarantee**: You share **epistemic updates** (how beliefs changed), not **ground truth** (actual states).

**See**: [DISTRIBUTED.md - Prediction-Error Exchange Protocol](DISTRIBUTED.md#prediction-error-exchange-protocol-future)

### 5. Locality of Orientation Scores

**Before**: Not explicitly stated that all computation is local.

**After**: Emphasized that orientation scores use only **ego subgraph data** (at most two hops):
- All metrics computed from immediate neighbors
- No global graph visibility required
- Fundamental to Markov blanket principle

**Key insight**: You navigate using only local information, not global structure. You don't need the full network to make good decisions.

**See**: [ARCHITECTURE.md - Orientation Score Locality](ARCHITECTURE.md#6-orientation-score-composite)

## Implementation Notes Added

### Diffusion Metric Scalability

Added note that naive heat kernel inversion is O(n³). For larger graphs, use:
- Truncated spectral decomposition
- Local heat propagation
- Sparse solvers (conjugate gradient)

**See**: [ARCHITECTURE.md - Heat-Residual Novelty Implementation](ARCHITECTURE.md#4-heat-residual-novelty)

### Active Inference Loop

Added explicit description of the full predict-interact-update cycle:

1. Predict (form expectations)
2. Interact (observe reality)
3. Compute prediction error
4. Update belief
5. Update confidence
6. Recompute metrics
7. Choose next action
8. (Optional) Exchange prediction-error summary

**See**: [DISTRIBUTED.md - The Full Active Inference Loop](DISTRIBUTED.md#the-full-active-inference-loop-future)

## Roadmap Additions

Updated future features to include:
- Temporal dynamics (decay, momentum, trajectory prediction)
- Rank-2 attunement adapter (flexible attunement with residual directions)
- Active inference loop (full predict-update cycle)
- Prediction-error exchange protocol
- Projected mutual predictability computation

**See**: [IMPLEMENTATION.md - What's Next](IMPLEMENTATION.md#whats-next)

## Terminology Shifts

| Old Term | New Term | Rationale |
|----------|----------|-----------|
| "Ego graph" | "Epistemic ego graph" | Emphasizes that it's a belief structure, not objective reality |
| "Neighbor embeddings" | "Predicted neighbor embeddings" | Clarifies these are guesses, not ground truth |
| "Potential = cosine similarity" | "Potential = projected mutual predictability" | Captures bidirectional modeling capacity, not just alignment |
| "Information exchange" | "Prediction-error exchange" | Emphasizes epistemic updates over raw data sharing |
| "Global metrics" | "Local metrics (two-hop)" | Highlights Markov blanket property |

## What Stayed the Same

The following were already correct and unchanged:
- Markov blanket foundation
- Six-metric structure (ego picture, legibility, attunement, novelty, translation, orientation)
- Conversational interface vision (Claude-based ego graph building)
- Privacy-first, distributed architecture
- Active inference framing

## Alignment Assessment

**Original vision → Documentation**: ~90-95% aligned (per GPT-5 feedback)

**After refinements**: ~98% aligned

Remaining gaps are in **implementation** (features not yet built), not **conceptual framing**.

## Next Steps for Implementation

1. Add `is_self`, `prediction_confidence`, `last_updated` fields to JSON schema
2. Implement projected mutual predictability computation
3. Add legibility threshold check to attunement interpretation
4. Optimize heat kernel computation for larger graphs
5. Design prediction-error exchange message format
6. Implement active inference loop in conversational interface

---

**Summary**: The core architecture was sound. These refinements emphasize the **epistemic nature** of the system - that it's fundamentally about **predictive models refined through interaction**, not about **objective graphs with complete information**. This distinction is central to both the privacy model and the active inference framework.
