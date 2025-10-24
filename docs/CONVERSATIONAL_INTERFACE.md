# Conversational Interface

The conversational interface lets you build and update your ego graph through natural dialogue with Claude, rather than manually editing JSON files.

## Quick Start

In a Claude Code session, run:

```
/ego-session
```

This loads a conversational guide that helps Claude facilitate an ego graph building session with you.

## How It Works

### The Flow

1. **Start the session**: `/ego-session` loads instructions for Claude
2. **Natural conversation**: Claude asks about your interests, network, recent interactions
3. **Automatic extraction**: Claude identifies people, topics, and relationships from your responses
4. **Incremental updates**: Your ego graph JSON is updated as you talk
5. **Embedding computation**: New phrases are sent to ChromaDB for vector caching
6. **Navigation insights**: Claude runs analysis and surfaces strategic suggestions

### Example Session

```
You: /ego-session

Claude: Let's build your ego graph! What would you like to call it?

You: Let's call it "justin"

Claude: Great! What's been on your mind lately?

You: I've been really into audio embeddings and semantic search.
     Also thinking about how to build better navigation interfaces.

Claude: Nice - those sound related. Who in your network are you talking
        to about these ideas?

You: Mainly Lara. She's a DJ and we geek out about music similarity
     and playlist flows.

Claude: What specifically does Lara care about?

You: Playlist transitions, beat matching, and she's curious about
     using embeddings for music visualization.

Claude: Got it. Anyone else in your network working on different things?

You: Yeah, Sam is in org design. We talk about team coordination
     and feedback loops.

Claude: [After more conversation...]

        Let me update your ego graph and see what insights come up.

        [Claude writes to data/ego_graphs/justin.json]
        [Claude computes embeddings via ChromaDB]
        [Claude runs: uv run python src/ego_ops.py justin]

        Your network has two main clusters:
        - Audio-tech: Lara, Pat, Kim
        - Organizational: Sam, Taylor, Riley

        Blake has the highest orientation score (0.73) - they bridge
        both worlds with interests in music cognition and cognitive
        science. Might be worth reaching out to Blake to connect these
        two parts of your network.
```

## What Claude Extracts

### From Your Answers
- **Topics you care about**: Phrases that represent your semantic field
- **Intensity**: How much you talk about something indicates its weight (0.1 to 1.0)
- **Recency**: Timestamps for temporal decay (future feature)

### About Each Person
- **Name and ID**: Full name + short identifier
- **Their interests**: What topics define their semantic field
- **Phrase weights**: How central each topic is to them (your perception)

### About Interactions
- **Connection strength** (0-1 scale):
  - 0.9-1.0: Very close, frequent interaction
  - 0.7-0.8: Regular connection
  - 0.5-0.6: Occasional interaction
  - 0.3-0.4: Distant but real
  - 0.1-0.2: Barely connected

## File Structure

Your ego graph is saved as `data/ego_graphs/{name}.json`:

```json
{
  "version": "0.2",
  "focal_node": "F",
  "metadata": {
    "created_at": "2025-10-24",
    "description": "Ego graph for Justin"
  },
  "nodes": [
    {
      "id": "F",
      "name": "Justin",
      "is_self": true,
      "phrases": [
        {"text": "audio embeddings", "weight": 0.9, "last_updated": "2025-10-24"},
        {"text": "semantic search", "weight": 0.8, "last_updated": "2025-10-24"}
      ]
    },
    {
      "id": "L",
      "name": "Lara",
      "is_self": false,
      "phrases": [
        {"text": "playlist transitions", "weight": 0.8, "last_updated": "2025-10-24"}
      ]
    }
  ],
  "edges": [
    {"source": "F", "target": "L", "actual": 0.9}
  ]
}
```

Embeddings are automatically cached in `./chroma_db/` (not in JSON).

## Navigation Insights

After building your graph, Claude can provide insights:

### Orientation Scores
"Who should I talk to next?"

Claude runs the full analysis and identifies high-value connections based on:
- Novelty (do they offer new perspectives?)
- Bridging potential (do they connect different clusters?)
- Mutual intelligibility (can you understand each other?)
- Strategic alignment (does it serve your goals?)

### Cluster Detection
"What are the main themes in my network?"

The system identifies semantic clusters (groups of people with related interests) and shows:
- Cluster composition
- Your attention distribution across clusters
- Bridge opportunities between clusters

### Translation Hints
"How should I frame my message to reach [person]?"

The system suggests which phrases to use (and avoid) when communicating with specific people, based on vocabulary overlap and semantic alignment.

### Legibility & Attunement
"How well do I understand my network?"

- **Legibility (R²_in)**: How predictable you are from your neighbors' perspectives
- **Attunement (R²_out)**: How well you understand each person's position

Low attunement + high legibility = learning opportunity.

## Privacy

Your ego graph:
- Lives on your local machine (`data/ego_graphs/`)
- Is gitignored by default (won't be committed)
- Never leaves your machine unless you explicitly share it
- Contains only your perceptions of others (not their ground truth)

## Updating Your Graph

You can:

1. **Have another conversation**: Run `/ego-session` again anytime
2. **Edit JSON directly**: The format is human-readable
3. **Manual analysis**: Run `uv run python src/ego_ops.py {your_name}` anytime

Changes are incremental - Claude will load your existing graph and add to it.

## Future Features (v0.3+)

**Temporal decay**: Phrases will automatically fade over time (τ ≈ 40 days) unless re-mentioned. This creates a "living graph" that reflects your current focus, not historical accumulation.

**Edge dynamics**: Track past/present/future interaction dimensions, with separate decay rates.

**Prediction confidence**: Track how certain you are about your model of each person, with confidence decay over time without interaction.

**Trajectory analysis**: Detect semantic velocity (what direction your interests are moving) and predict future positions.

## Tips for Good Sessions

1. **Be natural**: This isn't a form to fill out - just talk about your week
2. **Think recent**: Focus on current interests and active relationships
3. **Be honest**: Connection strengths should reflect reality, not aspiration
4. **Iterate**: You can always come back and refine
5. **Use insights**: The system is most valuable when you act on what it surfaces

## Technical Details

Under the hood, Claude:
- Reads/writes JSON directly (no special API needed)
- Calls `src.embeddings.get_embedding_service()` to cache vectors in ChromaDB
- Runs `src/ego_ops.py` via bash to compute navigation metrics
- Interprets mathematical results into conversational insights

The `/ego-session` slash command is just a markdown file (`.claude/commands/ego-session.md`) with detailed instructions - no custom code required.

## See Also

- [VISION.md](VISION.md): The big picture and use cases
- [ARCHITECTURE.md](ARCHITECTURE.md): Mathematical foundations of the six metrics
- [IMPLEMENTATION.md](IMPLEMENTATION.md): Technical implementation guide
- [V02_MIGRATION.md](V02_MIGRATION.md): Details on phrase-level embeddings and ChromaDB
