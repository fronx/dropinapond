**Socio‑Semantic Guide to Semantic Flow**
- Output file: `data/analyses/<name>_latest.json` (example: `data/analyses/fronx_latest.json`)
- Generator: `src/semantic_flow.py`

This guide explains what each number means socially and semantically, using the actual field names present in the JSON.

**Orientation**
- `version`, `ego_graph_file`, `parameters` describe the run context.
- Core results live under `metrics` with three parts: `layers`, `fields`, and `coherence`.
- `recommendations.semantic_suggestions` lists high‑affinity non‑edges to consider.

**Layers**
- `metrics.layers.structural_edges[src][tgt]`
  - What it is: Factual tie strength from `edges.json` (`actual`, clipped to [0,1]). Directed.
  - Read as: How much attention/interaction actually flows from `src` to `tgt` now.
  - Use it to: Anchor analysis in lived relationships and responsibilities.

- `metrics.layers.semantic_affinity[src][tgt]`
  - What it is: Phrase‑level affinity on existing directed edges only. Weighted mean of similar phrase pairs (cosine ≥ `parameters.cos_min`). [0,1].
  - Read as: Conceptual pull from `src` toward `tgt` given what they talk/think about.
  - Use it to: See which existing ties are semantically reinforced (or thin) by shared themes.

- `metrics.layers.effective_edges[src][tgt]`
  - What it is: Blended weight `W = alpha * S + (1 - alpha) * A`. Directed. [0,1].
  - Read as: Net likelihood of attention/collaboration flow when structure and meaning both matter.
  - Use it to: Compare where meaning strengthens or softens structural ties; basis for clustering.

**Edge Fields**
- `metrics.fields.edge_fields[src][tgt].predictability_raw`
  - What it is: Symmetric mutual predictability `F = sqrt(A[src,tgt] * A[tgt,src])`, normalized to [0,1].
  - Read as: Reciprocity of semantic pull; how jointly expectable the pair is as co‑participants.
  - Use it to: Spot pairs with balanced, high mutual alignment (collaboration feels “natural”).

- `metrics.fields.edge_fields[src][tgt].distance_raw`
  - What it is: Semantic distance `D = 1 − cosine(mean_src, mean_tgt)`, normalized to [0,1]. Lower is closer.
  - Read as: Conceptual gap between people’s centers of gravity; 0 = very close, 1 = far.
  - Use it to: Contrast with compatibility metrics; identify adjacent vs. far‑field relations.

**Blanket (Context‑Aware) Fields**
- `metrics.fields.edge_fields_blanket[src][tgt].predictability_blanket`
  - What it is: Markov‑blanket coupling from locally normalized affinities: `F_MB = A_norm[src,tgt] * A_norm[tgt,src]`, [0,1].
  - Read as: Context‑conditioned fit; given each person’s overall attention budget, how tightly these two couple.
  - Use it to: Prioritize pairs that stand out relative to everything else each person could attend to.

- `metrics.fields.edge_fields_blanket[src][tgt].exploration_potential`
  - What it is: `E_MB = F_MB * (1 − D)`. High when fit is strong and semantic distance is small.
  - Read as: Low‑friction opportunity surface; where a next step would be both easy and meaningful.
  - Use it to: Sequence introductions or projects that are likely to “click” quickly.

**Clusters and Coherence**
- `metrics.clusters`
  - What it is: Communities found on the symmetrized blended graph (`W + W^T`) via greedy modularity.
  - Read as: Regions where structure and meaning converge into practice communities.
  - Use it to: Orient the map; not prescriptive—boundary nodes often bridge regions.

- `metrics.coherence.regions[]`
  - `internal_F`, `external_F`: Average mutual similarity inside vs. across the region.
  - `internal_MB`, `external_MB`: Same, but context‑aware (blanket) coupling.
  - `internal_D`, `external_D`: Mean semantic distance inside vs. outside.
  - `conductance_sem`: Cut/volume using the semantic coupling used for communities (lower = cleaner boundary).
  - `silhouette_D`: Distance‑based silhouette (closer to 1 = well‑separated by meaning).
  - `coherence_F`, `coherence_MB`: Ratio‑style cohesion (internal / (internal + external)).
  - Read as: How much a region holds together by meaning (F, F_MB) and how distinct it is (D).

- `metrics.coherence.nodes[node_id]`
  - `region_index`: Which `metrics.clusters` block the node belongs to.
  - `avg_in`, `avg_out`: Average coupling to in‑region vs. out‑region (uses blanket coupling when available).
  - `fit_diff`, `fit_ratio`: Margin and ratio of in‑region vs. out‑region coupling.
  - Read as: Individual fit within the assigned region; large positive margins signal strong anchoring.

**Recommendations**
- `recommendations.semantic_suggestions[]`
  - Fields: `source`, `target`, `affinity`.
  - What it is: High‑affinity non‑edges. Candidates are ranked by mean‑embedding similarity, then refined via phrase‑level affinity with threshold `parameters.cos_min`.
  - Read as: Likely fruitful introductions or collaborations that are not yet connected.
  - Use it to: Create bridges where ideas already rhyme.

**Parameters (levers)**
- `parameters.alpha`
  - Higher emphasizes `structural_edges`; lower emphasizes `semantic_affinity`.
  - Practical: Raise to respect existing responsibilities; lower to surface thematic gravity.

- `parameters.cos_min`
  - Minimum cosine for phrase‑pair inclusion in `semantic_affinity` and suggestions refinement.
  - Practical: Raise to demand sharper topical overlap; lower to admit looser resonance.

- `parameters.suggest_k`, `parameters.suggest_pool`
  - Per‑node suggestion count and candidate pool size.
  - Practical: Tune for breadth vs. focus in recommended non‑edges.

**Reading Patterns (quick heuristics)**
- High `predictability_raw` and low `distance_raw`: deep, mutual conceptual fit; easy collaboration.
- High `predictability_blanket` but moderate `distance_raw`: strong relative pull amid many options; a “magnet” tie.
- High `exploration_potential`: low‑friction next step—great for early wins.
- Region with high `coherence_MB` and low `conductance_sem`: durable practice community.
- Node with high `fit_ratio` and `fit_diff`: strongly anchored; conversely, low margins can indicate bridge/ambassador roles.

**Notes and caveats**
- Direction matters for `structural_edges`, `semantic_affinity`, and `effective_edges`. Mutual metrics (`predictability_raw`, `predictability_blanket`) summarize reciprocity.
- `semantic_affinity` is computed only where an edge exists in `structural_edges`; zeros elsewhere are “not evaluated,” not necessarily “no affinity.”
- All reported fields are normalized to be comparable in [0,1] unless noted.

