# Temporal Dynamics: Living Semantic Fields

This document explains how the system handles time, memory, and semantic drift without accumulating historical baggage.

## Core Design Stance

**The graph represents a living present, not an archive.**

- No ever-growing historical ledger
- No full event replay on updates
- One continuously refreshed semantic field
- Old ideas fade naturally, revive when re-mentioned

This keeps the system **memory-efficient**, **privacy-preserving**, and **cognitively faithful** (mirrors how human attention works).

## The Living Graph Model

### What's Stored

Each phrase in the ego graph has:

```json
{
  "text": "audio embeddings",
  "embedding": [0.82, 0.31, -0.15, 0.41, 0.52],
  "weight": 0.85,              // Current activation (decays over time)
  "last_updated": "2025-10-15"  // Timestamp of most recent mention
}
```

**Key insight**: The `weight` already encodes recency through continuous decay. You don't need separate "how many times mentioned" or "when first added" fields.

### Update Algorithm: Decay + Refresh

When a new conversation arrives:

```python
def update_ego_graph(existing_graph, new_dialogue, tau=40):
    """
    Update semantic field from new evidence without storing full history.

    Args:
        existing_graph: Current graph snapshot
        new_dialogue: New conversation with extracted phrases
        tau: Decay timescale in days (default: 40 days)

    Returns:
        Updated graph with decayed weights and refreshed phrases
    """
    current_time = now()

    # Step 1: Apply exponential decay to all existing phrases
    for phrase in existing_graph.phrases:
        delta_t = (current_time - phrase.last_updated).days
        phrase.weight *= exp(-delta_t / tau)

    # Step 2: Extract new phrases from dialogue
    mentioned_phrases = extract_phrases_from_dialogue(new_dialogue)

    # Step 3: Bump weights for re-mentioned phrases
    for phrase_text in mentioned_phrases:
        if phrase_text in existing_graph.phrases:
            # Phrase exists: boost weight
            existing_phrase = existing_graph.phrases[phrase_text]
            existing_phrase.weight += 0.2  # Additive boost
            # OR: existing_phrase.weight *= 1.5  # Multiplicative boost
            existing_phrase.last_updated = current_time
        else:
            # New phrase: add with small initial weight
            new_phrase = {
                "text": phrase_text,
                "embedding": embed(phrase_text),
                "weight": 0.1,
                "last_updated": current_time
            }
            existing_graph.phrases.append(new_phrase)

    # Step 4: Renormalize weights (optional, depends on metric needs)
    total_weight = sum(p.weight for p in existing_graph.phrases)
    for phrase in existing_graph.phrases:
        phrase.weight /= total_weight

    # Step 5: Prune phrases below threshold (optional)
    existing_graph.phrases = [p for p in existing_graph.phrases
                              if p.weight > 0.01]

    return existing_graph
```

### Decay Timescale (τ)

**τ = 40 days** (default): Half-life of ~28 days

After time Δt, weight becomes: `w * exp(-Δt / τ)`

Example decay rates:
- 1 week: `w * 0.84` (84% retained)
- 1 month: `w * 0.46` (46% retained)
- 3 months: `w * 0.10` (10% retained)
- 6 months: `w * 0.01` (1% retained, likely pruned)

**Adjustable per user**: Some users want faster forgetting (τ=20), others want longer memory (τ=60).

### Boost Strategies

When a phrase is re-mentioned, how much to boost?

**Option 1: Additive boost** (default)
```python
phrase.weight += 0.2
```
Pro: Simple, predictable
Con: Can exceed 1.0 before normalization

**Option 2: Multiplicative boost**
```python
phrase.weight *= 1.5
```
Pro: Naturally bounded
Con: Very low weights barely recover

**Option 3: Ceiling boost**
```python
phrase.weight = min(phrase.weight + 0.2, 1.0)
```
Pro: Bounded, prevents runaway
Con: Slightly more complex

**Recommendation**: Use additive boost + renormalization (most straightforward).

## Seasonal Memory: Long-Term Dynamics

### Dormant Topics

Old interests don't disappear - they asymptotically fade:

```
Week 1: weight = 1.0
Week 4: weight = 0.46
Week 12: weight = 0.10
Week 24: weight = 0.01 (likely pruned)
```

If never mentioned again, the phrase eventually falls below pruning threshold (~0.01) and is removed.

**But if revived**:

```python
# Phrase at weight=0.02 (nearly pruned)
# User mentions it again
weight = 0.02 + 0.2 = 0.22  # Back to mid-range!
last_updated = now()
```

This creates **seasonal memory**:
- Topics cycle in and out naturally
- Nothing is forgotten permanently (can always revive)
- But dormant topics don't distort current representation

### Periodic Checkpoints (Optional)

For longitudinal analysis or debugging, you can snapshot the graph periodically:

```bash
# Every month, save a checkpoint
cp ego_graph.json checkpoints/ego_graph_2025-10.json
```

**Critical**: Checkpoints are **never fed back** into the active model. They're for:
- Understanding semantic drift over time
- Debugging unexpected behavior
- User reflection ("what was I focused on 6 months ago?")

Active graph always operates on the single living file.

## Privacy & Efficiency

### No Historical Event Log

**We don't store**:
- Every conversation transcript
- When each phrase was first mentioned
- Frequency counts over time
- Edit history

**We only store**:
- Current phrase weights (already encode recency)
- Last update timestamp per phrase

**Privacy win**: If user deletes a phrase, it's truly gone. No hidden audit trail.

**Efficiency win**: Graph size stays constant (~100-500 phrases typical), regardless of conversation count.

### Selective Forgetting

Users can manually accelerate forgetting for specific topics:

```python
# User wants to forget about "cryptocurrency"
for phrase in ego_graph.phrases:
    if "crypto" in phrase.text.lower():
        phrase.weight *= 0.1  # Aggressive decay
```

Combined with natural decay, unwanted topics disappear within weeks.

## Interaction with Other Temporal Dimensions

### Edge Weights: Past/Present/Future

Edges also have temporal structure:

```json
{
  "source": "F",
  "target": "B",
  "actual": {
    "past": 0.7,     // Historical relationship strength
    "present": 0.3,   // Current interaction level
    "future": 0.6     // Planned/desired interaction
  }
}
```

**Edge decay**: Similar exponential decay applies

```python
# Every update, decay edge weights
for edge in ego_graph.edges:
    delta_t = (now() - edge.last_updated).days

    # Decay past (memories fade)
    edge.actual.past *= exp(-delta_t / tau_past)  # tau_past = 60 days

    # Decay present (current interactions become past)
    edge.actual.present *= exp(-delta_t / tau_present)  # tau_present = 20 days

    # Decay future (plans become stale)
    edge.actual.future *= exp(-delta_t / tau_future)  # tau_future = 30 days
```

**When user reports interaction**: Boost `present`, transfer old `present` to `past`

```python
# User: "Had coffee with Blake today"
edge.actual.past = 0.5 * edge.actual.past + 0.5 * edge.actual.present
edge.actual.present = 0.8  # Fresh interaction
edge.last_updated = now()
```

### Neighbor Prediction Confidence

Prediction confidence also decays:

```json
{
  "id": "B",
  "prediction_confidence": 0.6,
  "last_updated": "2025-08-15"
}
```

```python
# Confidence decays if no recent interaction
delta_t = (now() - neighbor.last_updated).days
neighbor.prediction_confidence *= exp(-delta_t / tau_confidence)
# tau_confidence = 60 days (slower than phrase decay)
```

**Why slower?**: Phrase interests shift quickly, but fundamental aspects of a person (captured in your model of them) change more slowly.

## Cognitive Alignment

This temporal model mirrors human cognition:

1. **Recency bias**: Recent topics have higher weight
2. **Forgetting curve**: Exponential decay matches Ebbinghaus forgetting
3. **Spaced repetition**: Re-mentioning strengthens memory
4. **Interference**: New topics naturally push out old ones (via normalization)
5. **No perfect recall**: You don't remember every conversation, just weighted impressions

The system's representation of you **drifts** over time, just like your actual semantic field.

## Implementation Example

### Conversational Update Flow

```python
# User has a session with Claude
user: "I've been really into regenerative agriculture lately.
       Also had a great conversation with Sarah about urban farming."

# Claude extracts:
new_phrases = [
    "regenerative agriculture",
    "urban farming",
    "soil health",  # inferred related concept
]
interaction_updates = [
    {"person": "Sarah", "topics": ["urban farming"], "strength": 0.8}
]

# Update ego graph
ego_graph = load_ego_graph("user_graph.json")
ego_graph = update_phrases(ego_graph, new_phrases, tau=40)
ego_graph = update_edges(ego_graph, interaction_updates)
save_ego_graph("user_graph.json", ego_graph)

# Next week, user talks about something else
# Regenerative agriculture weight automatically decays
# No manual cleanup needed
```

### Configuration

Users can tune temporal parameters:

```json
{
  "temporal_config": {
    "phrase_decay_tau": 40,           // Days (faster = quicker forgetting)
    "edge_present_decay_tau": 20,      // Current interactions become past
    "edge_past_decay_tau": 60,         // Memories fade slowly
    "edge_future_decay_tau": 30,       // Plans decay at medium rate
    "confidence_decay_tau": 60,        // Trust in predictions
    "phrase_boost_amount": 0.2,        // How much to boost on re-mention
    "prune_threshold": 0.01            // Remove phrases below this weight
  }
}
```

## Future Extensions

### Momentum & Trajectory

Track **semantic velocity** (rate of change):

```python
# Every update, compute centroid shift
old_centroid = mean(phrase.embedding * phrase.weight for phrase in old_graph)
new_centroid = mean(phrase.embedding * phrase.weight for phrase in new_graph)

velocity = (new_centroid - old_centroid) / delta_t

# Store as graph metadata
ego_graph.metadata.semantic_velocity = velocity
```

This enables:
- "You're moving toward climate tech" (velocity points that direction)
- "Your interests have been stable" (low velocity)
- Trajectory prediction (where will you be in 3 months?)

### Oscillation Detection

Some interests are **periodic** (e.g., academic semester cycles):

```python
# Track weight history via checkpoints
weight_history = [0.2, 0.4, 0.2, 0.5, 0.2, 0.4]  # repeating pattern

# Detect periodicity (Fourier analysis or autocorrelation)
if is_periodic(weight_history):
    phrase.metadata.period = 90  # Days
    phrase.metadata.phase = compute_phase(weight_history)
```

System could then **anticipate**: "Teaching season starts in 2 weeks, you'll probably care about pedagogy again."

### External Time Markers

Link phrases to external events:

```json
{
  "text": "conference talk",
  "weight": 0.8,
  "last_updated": "2025-10-15",
  "time_marker": {
    "type": "event",
    "date": "2025-11-20",
    "description": "Strange Loop conference"
  }
}
```

Weight decays normally, but system knows this phrase is tied to future event. Can surface reminders: "Conference in 1 week - want to prep?"

## Summary

**Core algorithm**: Decay + refresh, no historical replay

**Temporal parameters**:
- Phrase decay: τ = 40 days
- Edge present decay: τ = 20 days
- Edge past decay: τ = 60 days
- Confidence decay: τ = 60 days

**Benefits**:
- Lightweight (one graph file, no logs)
- Privacy-preserving (no hidden history)
- Cognitively faithful (mirrors forgetting curves)
- Self-regulating (dormant topics naturally prune)

**Key insight**: Current weights already encode the past through continuous decay. You don't need to store the past separately.

This creates a **living semantic field** that shifts naturally over time, stays responsive to new evidence, and forgets gracefully when topics become irrelevant.

---

**Next steps**: Implement decay logic in graph update pipeline, add temporal config to user settings, create checkpointing utility for longitudinal analysis.
