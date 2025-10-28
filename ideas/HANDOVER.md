# Handover: UI Metric Explanations

## Current State (2025-10-24)

We've successfully built a sidebar UI that explains what data went into each metric computation. **The phrase similarity computation has been moved to the backend and now uses proper embedding-based semantic similarity.**

## What We Fixed

**Problem**: The UI was computing word-overlap matches (crude, misleading) instead of using the actual embeddings.

Example of **bad match** from old word overlap approach:
```
You: biology and life sciences
Them: daily life and household coordination
29% word overlap ❌ (just the word "life")
```

**Solution**: Backend now computes embedding-based phrase similarities using ChromaDB, exports to JSON, and UI displays the pre-computed results.

Example of **good match** from new embedding approach:
```
You: embodied presence
Them: embodied presence and non-verbal connection
90% semantic similarity ✅ (0.90 cosine in 384-dim space)
```

## The Solution Architecture

**Principle**: The Python analysis should compute and export ALL the data needed for UI explanations. The UI should just display it, not recompute anything.

### What We've Completed ✅

1. **Added `compute_phrase_similarities()` in [ego_ops.py:336-395](src/ego_ops.py#L336-L395)**
   - Computes actual embedding-based cosine similarity between phrase pairs
   - Threshold: cosine > 0.3
   - Sorts by `similarity × focal_weight × neighbor_weight`
   - Returns top-10 pairs per neighbor

2. **Integrated into analysis pipeline** ([ego_ops.py:903-916](src/ego_ops.py#L903-L916))
   - Computes phrase similarities for each neighbor
   - Adds to analysis JSON as `metrics.phrase_similarities`

3. **Updated `save_analysis_json()`** ([ego_ops.py:627](src/ego_ops.py#L627)) to include phrase similarities in output

4. **Frontend updated** ([PersonDetailSidebar.jsx](gui/src/components/PersonDetailSidebar.jsx))
   - ✅ Removed 50 lines of word-overlap computation (lines 69-72)
   - ✅ Now uses `analysisData.metrics.phrase_similarities[personId]`
   - ✅ Displays "semantic similarity" instead of "word overlap"
   - ✅ Added disclaimer that phrase similarities are contextual, not the direct R² computation

## Important Caveat: Phrase Similarities vs. Readability Computation

**What we're showing (phrase similarities)** is NOT exactly what's used in the readability R² computation:

- **Readability metric**: Uses **mean embeddings** (weighted average of all phrases → 384-dim vector), then applies ridge regression: `α × their_mean ≈ your_mean`
- **Phrase similarities**: Shows **individual phrase-to-phrase** cosine similarities

**Why we show phrase similarities anyway:**
- They provide **intuitive context** for why the means are similar
- They're easier to understand than "mean embedding contributors"
- They **correlate strongly** with readability even if not the direct input

The UI now includes a disclaimer: *"The phrase overlaps below help explain why the means are similar, but the actual R² is computed on the aggregated embeddings, not individual phrases."*

### Future Enhancement: True Mean Contributors

For perfect accuracy, we could add `compute_readability_explanation()` ([ego_ops.py:397-486](src/ego_ops.py#L397-L486)) which shows:
- Which phrases contribute most to each person's mean embedding (by weight)
- The ridge coefficient `α` used in the reconstruction
- Which aspects of your interests are well/poorly reconstructed

This is more complex to display but would be the mathematically correct explanation.

## What Needs to Be Done Next (Optional Improvements)

### 1. Apply the Same Pattern to ALL Metrics

The phrase_similarities work is a **template** for how to handle all metric explanations:

| Metric | What to Add to Analysis JSON | Where to Display |
|--------|------------------------------|------------------|
| **Readability** | ✅ `phrase_similarities[neighbor_id]` | Sidebar "gets you" section |
| **Overlap** | Network overlap details: shared neighbors, connection patterns | Sidebar "overlap" section |
| **Subjective Attunement** | Phrase-level reconstruction errors for cluster members | Sidebar cluster section |
| **Heat Residual Novelty** | Semantic distance breakdown, diffusion paths | Sidebar novelty section |
| **Orientation Score** | Component breakdown with actual values used in formula | Sidebar orientation section |

### 2. Detailed Implementation Plan for Each Metric

#### A. Network Overlap (Already Computed, Needs Export Detail)

**Add to `save_analysis_json()`:**
```python
overlap_details = {}
for j in neighbors:
    focal_neighbors = set(G.neighbors(F))
    j_neighbors = set(G.neighbors(j))
    shared = focal_neighbors & j_neighbors
    overlap_details[j] = {
        'shared_neighbors': list(shared),
        'focal_only': list(focal_neighbors - j_neighbors),
        'neighbor_only': list(j_neighbors - focal_neighbors),
        'jaccard': overlaps[j]
    }

# Add to analysis dict:
"overlap_details": overlap_details
```

**Frontend:** Display actual shared neighbor names, not just the Jaccard number.

#### B. Subjective Attunement (Cluster-Level)

**Add function in `ego_ops.py`:**
```python
def compute_attunement_breakdown(ego: EgoData, cluster: Iterable[str], lam: float = 1e-3):
    """
    For each member of cluster, compute reconstruction error: z_j - beta*z_F.
    Returns which phrases are well-predicted vs. poorly predicted.
    """
    F = ego.focal
    zF = ego.embeddings[F]
    denom = np.dot(zF, zF) + lam

    results = []
    for j in cluster:
        zj = ego.embeddings[j]
        beta = float(np.dot(zF, zj) / denom)
        z_hat = beta * zF
        residual = zj - z_hat
        residual_norm = float(np.linalg.norm(residual))

        results.append({
            'node_id': j,
            'beta': beta,
            'residual_norm': residual_norm,
            'r2': _r2_vector(zj, z_hat)
        })

    return results
```

**Integrate**: Call for each cluster, add to analysis JSON under `attunement_breakdown_per_cluster`.

**Frontend**: Show which cluster members are well-understood (low residual) vs. poorly understood (high residual).

#### C. Heat Residual Novelty

**Add function:**
```python
def compute_novelty_breakdown(ego: EgoData, pocket: Iterable[str], t: float = 0.1):
    """
    Return which phrases/dimensions contribute most to the heat residual.
    """
    # (Similar structure to heat_residual_norm, but return per-dimension contribution)
    # This is more advanced - may need to decompose the residual vector
```

**Frontend**: Visualize which semantic dimensions make the cluster "novel" vs. "familiar".

#### D. Orientation Score Components

**Already computed!** Just need to export the breakdown:

```python
# In orientation_scores() function, also return the components:
orientation_breakdown = {}
for j in G.neighbors(F):
    k = target_pocket_by_node.get(j, None)
    cos_term = _cos(qk_by_pocket[k], ego.embeddings[j]) if k is not None else 0.0

    orientation_breakdown[j] = {
        'exploration': weights.lam1 * one_minus_overlap[j],
        'legibility': weights.lam2 * r2_in_per_neighbor.get(j, 0.0),
        'attunement': weights.lam3 * (r2_out_by_pocket.get(k, 0.0) if k is not None else 0.0),
        'semantic_match': weights.lam4 * cos_term,
        'stability_penalty': -weights.lam5 * inst.get(j, 0.0),
        'total': out[j]
    }
```

**Frontend**: Show a bar chart or breakdown of which components contribute most to the orientation score.

## Key Files to Modify

1. **Backend (Python)**:
   - [src/ego_ops.py](src/ego_ops.py) - Add detail extraction functions
   - [src/ego_ops.py:682-700](src/ego_ops.py#L682-L700) - Update analysis dict

2. **Frontend (React)**:
   - [gui/src/components/PersonDetailSidebar.jsx](gui/src/components/PersonDetailSidebar.jsx) - Display the data
   - Remove lines 71-119 (word overlap computation)
   - Replace with direct use of `analysisData.metrics.*`

3. **Data Flow**:
   ```
   Python (ego_ops.py)
     └─> Compute metric
     └─> Compute explanation data
     └─> Export to data/analyses/NAME_latest.json

   React (PersonDetailSidebar.jsx)
     └─> Load analysisData from JSON
     └─> Extract explanation data
     └─> Display in sidebar
   ```

## Testing the Changes

1. Run analysis: `uv run python src/ego_ops.py fronx`
2. Check JSON: `cat data/analyses/fronx_latest.json | jq '.metrics.phrase_similarities.susan | .[0:3]'`
3. Rebuild UI: `cd gui && npm run build`
4. View in browser and click on Susan
5. Verify: No more "biology and life sciences" ↔ "daily life" false matches

## Design Principles

1. **Single source of truth**: Python computes, JSON stores, React displays
2. **No client-side recomputation**: UI never recalculates metrics
3. **Semantic correctness**: Always use embeddings, never word overlap
4. **Explainability**: Every number shown has a data provenance trail

## Current State of Files

- ✅ `src/ego_ops.py`: Has `compute_phrase_similarities()` and `compute_readability_explanation()`, phrase similarities integrated into pipeline
- ✅ `src/embeddings.py`: Has `get_all_node_phrases()` for fetching text + embeddings
- ✅ `gui/src/components/PersonDetailSidebar.jsx`: Uses backend phrase_similarities data
- ✅ Analysis JSON: Has `phrase_similarities` field with embedding-based similarities
- ✅ `data/analyses/fronx_latest.json`: Generated with new phrase_similarities data

## Next Session Goals

1. ✅ **COMPLETED**: Fixed phrase similarities in frontend - now uses backend data
2. **Short-term**: Add overlap_details, orientation_breakdown (1 hour)
3. **Medium-term**: Add attunement and novelty breakdowns (2 hours)
4. **Long-term**: Consider integrating `compute_readability_explanation()` for true mean contributors (if deemed more useful than phrase similarities)

## Questions to Resolve

- **Threshold for phrase similarity**: Currently 0.3 cosine. Too low? Too high?
- **Top-k**: Currently 10 phrase pairs. Enough? Too many?
- **UI presentation**: Should we show similarity scores as percentages? As bars?
- **Granularity**: Do we need per-dimension analysis for residuals?

## Context for Future Claude

The user wants **complete transparency** about what drives each metric. They noticed word overlap was giving false positives ("biology and life sciences" ↔ "daily life") and correctly identified this as a mismatch between backend (embeddings) and frontend (words).

The solution is systematic: **export explanation data from Python analysis, display it verbatim in React**. We've completed this for phrase_similarities (readability metric), but need to apply the same pattern to all other metrics.

User's philosophy: "If I see a metric, I should be able to click it and see exactly which data elements from the ego graph went into its computation."

**Important nuance discovered**: Phrase similarities are **contextual/correlational** with readability, not the direct computational input (which is mean embeddings + ridge regression). The user is okay with showing correlational data as long as there's a disclaimer. For perfect accuracy, we have `compute_readability_explanation()` ready to integrate.

---

**Status**: ✅ Phrase similarities complete (backend + frontend), ready to commit. Optional: extend to other metrics or add mean contributor explanations.
