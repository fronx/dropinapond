# Drop in a Pond: Documentation

## Overview

Drop in a Pond is a **semantic network navigation system** that helps you understand and strategically navigate your social/professional network using geometric reasoning over embedding spaces.

It's **distributed and privacy-preserving**: each person runs local computations without revealing their full network to others.

## Documentation Structure

### 1. [Vision & User Experience](VISION.md)
**What is this and what's it like to use?**

- The big picture: what problem does this solve?
- User experience: conversational ego graph building
- Use cases: research collaboration, career transitions, community building
- Why "Drop in a Pond"?

### 2. [Architecture](ARCHITECTURE.md)
**How does the system work internally?**

- Epistemic ego graph data structure
- The six navigation metrics (mathematical foundations)
- Keyphrase translation mechanism
- Embedding computation pipeline
- Claude as the interface layer

### 3. [Distributed Protocol](DISTRIBUTED.md)
**How do nodes communicate without compromising privacy?**

- The Markov blanket principle
- Local operations vs. inter-node exchange
- Predictive models updated through interaction
- Prediction-error exchange (not raw embeddings)
- Active inference framework
- Future federation protocol

### 4. [Conversational Interface](CONVERSATIONAL_INTERFACE.md)
**Build your ego graph through natural dialogue**

- Quick start with `/ego-session` command
- What Claude extracts from conversation
- Privacy and file structure
- Getting navigation insights
- Tips for effective sessions

### 5. [Implementation Guide](IMPLEMENTATION.md)
**How to build, run, and extend the system**

- Current implementation status
- Running the example
- Adding new metrics
- Integrating embedding models
- Technical details

### 6. [Refinements Document](REFINEMENTS.md)
**Key conceptual clarifications (post-initial review)**

- Epistemic ego graphs vs. objective graphs
- Projected mutual predictability (not just cosine similarity)
- Nuanced attunement interpretation
- Prediction-error exchange protocol
- Terminology shifts and alignment improvements

### 7. [Continuous Semantic Fields](CONTINUOUS_FIELDS.md)
**Architectural shift from discrete clustering to continuous fields**

- Why continuous fields vs. discrete clusters
- Data structure changes (phrase-level embeddings)
- Metric adaptations (kernel-based neighborhoods)
- On-demand clustering (compute transiently, never store)
- Implementation roadmap and migration path

### 8. [Temporal Dynamics](TEMPORAL_DYNAMICS.md)
**Living semantic fields: memory without historical baggage**

- The living graph model (no event log, just continuous decay)
- Update algorithm: exponential decay + refresh on re-mention
- Seasonal memory (dormant topics fade, revive when mentioned)
- Edge weight decay (past/present/future dimensions)
- Cognitive alignment (mirrors forgetting curves)

## Quick Start

If you just want to see it work:
```bash
uv sync
uv run python src/ego_ops.py fronx
```

This analyzes the example ego graph and outputs all navigation metrics.

## For Different Audiences

**I want to understand the vision**: Start with [Vision & User Experience](VISION.md)

**I want to understand how it works**: Read [Architecture](ARCHITECTURE.md)

**I want to understand the privacy model**: See [Distributed Protocol](DISTRIBUTED.md)

**I want to hack on it**: Go to [Implementation Guide](IMPLEMENTATION.md) and [README.md](../README.md)

**I want the mathematical details**: [Architecture](ARCHITECTURE.md) has all six metrics explained with code examples

## Design Philosophy

- **Privacy-first**: No centralized graph, no embedding leakage
- **Cognitively natural**: Interface mirrors how you think about your network
- **Actionable**: Outputs are concrete (who to talk to, how to frame it)
- **Composable**: Metrics can be weighted/combined for different goals
- **Extensible**: New signals can plug in easily

## Project Status

**Current (v0.2)**:
- All six navigation metrics with phrase-level embeddings
- Modular directory format with ChromaDB integration
- Conversational interface (`/ego-session` command)
- Command-line analysis with JSON export
- Visual interface (React + D3 force-directed graph)
- Translation hints and phrase-level similarity explanations
- Example graph (fronx) with 15 people across 6 clusters

**Planned (v0.3+)**:
- Kernel-based neighborhoods (replace discrete clustering)
- Gradient-based translation (replace centroid differences)
- Temporal dynamics with exponential decay
- Active inference loop (predict → interact → update)
- Inter-node privacy-preserving protocol
