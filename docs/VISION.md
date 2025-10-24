# Vision & User Experience

## What Is This?

Drop in a Pond is a **semantic network navigation system** that helps you understand and strategically navigate your social/professional network. It treats relationships as embeddings in semantic space and uses geometric reasoning to answer questions like:

- Which clusters exist in my network?
- Who could bridge me to new communities?
- How should I frame my message to reach different groups?
- Where is the novelty in my network - who offers perspectives I don't yet understand?

The system is **distributed and privacy-preserving**: each person runs local computations on their own ego graph without revealing their full network to others.

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

## Use Cases

### 1. Research Collaboration

**Query**: "I want to explore applications of diffusion models to biology. Who in my network can bridge me there?"

**System**: Identifies people at intersection of ML and bio clusters, suggests framings that resonate with each.

### 2. Career Transitions

**Query**: "I'm moving from software engineering to climate tech. Who should I talk to?"

**System**: Finds bridges between clusters, ranks by novelty and alignment, suggests conversation starters.

### 3. Community Building

**Query**: "I want to connect two groups I'm involved with. Who are the natural bridges?"

**System**: Identifies people with high betweenness and semantic overlap between clusters.

### 4. Learning Optimization

**Query**: "I want to learn about X. Who can teach me most effectively?"

**System**: Balances high expertise (keyphrase match), high attunement (they understand you), and structural accessibility (reachable through existing connections).

## Why "Drop in a Pond"?

When you interact with someone, the effect **ripples outward** through the network. The system helps you:

- Choose where to drop your stone (who to engage)
- Predict how the ripples will spread (diffusion through clusters)
- Optimize for desired outcomes (reach, novelty, alignment)

You navigate by making **local perturbations** (individual conversations) that have **global effects** (shifting your position in the network).

## Why Claude as Interface?

1. **Natural interaction**: You describe your network in your own words
2. **Contextual understanding**: Claude knows your history and can infer importance
3. **Semantic lifting**: Claude translates messy human descriptions into clean structured data
4. **Interpretation**: Claude translates mathematical metrics back into actionable advice

The human provides the **semantic content**.
The system provides the **geometric reasoning**.
Claude provides the **translation layer** between the two.

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

## The Long-Term Vision

This starts as a **personal tool** - you use it to navigate your own network.

Eventually it becomes a **distributed platform** where:
- Each person runs their own local node
- Nodes can exchange limited information through privacy-preserving protocols
- The global structure emerges from local interactions
- No central authority has access to the full graph

This is **decentralized social cognition** - collective intelligence without surveillance.
