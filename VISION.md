# Drop in a Pond: Vision

> **A tool for navigating the semantic topology of your social network**

## What Problem Does This Solve?

Your professional and social network is rich with latent potential. But most of that potential remains hidden because:

- You can't hold everyone in working memory at once
- You don't know who could bridge you to new ideas or communities
- You struggle to frame messages that resonate across different semantic contexts
- The "right person" for a question often isn't who you'd initially think

Traditional network tools (LinkedIn, contact managers, CRMs) treat relationships as nodes and edges. They tell you *who* you know, but not *what semantic regions your network covers* or *how to navigate between them*.

## The Core Insight

Your network has **semantic structure** that can be analyzed geometrically. People aren't just connected or disconnected - they occupy positions in a continuous space of ideas, interests, and expertise.

Understanding this topology lets you:
- **Identify bridges** between different communities you're part of
- **Find novelty** - people whose perspectives you don't yet understand
- **Navigate strategically** - choose next interactions based on exploration vs exploitation
- **Translate effectively** - frame your message differently for different semantic regions

## The Experience

### Building Your Map

You describe your network conversationally - who you know, what they care about, how you interact. The system captures this in a structured form, representing each person's interests as a semantic field rather than discrete tags.

This isn't networking software - it's a **navigational aid** for the network you already have.

### Getting Insights

Once your network is mapped, you can ask strategic questions:

> "Who should I talk to about regenerative agriculture?"

The system doesn't just match keywords. It identifies:
- People positioned to bridge semantic gaps
- Those offering perspectives you haven't encountered yet
- How to frame your message for different regions of your network
- Whether you need exploration (new territory) or exploitation (deepen existing connections)

### The Privacy Model

Your map lives locally. You maintain **predictive models** of others - your best guess of what they care about based on interactions. When you talk to someone, you update your model.

You never need direct access to others' full networks or semantic fields. The system operates on the **Markov blanket principle**: you can make excellent local navigation decisions with just your immediate neighborhood.

This enables federation: multiple people can run their own instances, share lightweight updates, and collectively benefit from better navigation - all without centralizing data or revealing private information.

## What Makes This Different

**Continuous over discrete**: No premature categorization. The structure emerges from analysis, not from forcing people into folders.

**Geometric reasoning**: Uses spectral graph theory, diffusion geometry, and kernel methods to understand your network as a navigable space.

**Privacy-first**: Distributed by default. Each person controls their own data and decides what to share.

**Living memory**: Designed to represent your current network, not an archive. Interests and connections naturally fade if not maintained.

**Action-oriented**: Built to answer "who should I talk to next?" not "who do I know?"

## Use Cases

**Freelancers & consultants**: Strategically develop your network based on semantic gaps and bridges, not just accumulation.

**Researchers**: Find collaborators positioned at the intersection of fields you care about.

**Community builders**: Understand the semantic topology of your community and identify connectors vs specialists.

**Anyone with a rich network**: Stop relying on memory and start navigating strategically.

## Current State

The system exists as working software you can run locally. You can:
- Build ego graphs through conversational sessions
- Run semantic flow analysis to understand network structure
- Visualize your network and see analysis results
- Get connection suggestions based on semantic proximity

See the [Documentation Index](docs/INDEX.md) for technical details and getting started guides.

## Future Directions

**Temporal dynamics**: Graphs that represent "living memory" - interests naturally fade unless actively maintained.

**Federated protocol**: Lightweight updates between nodes that preserve privacy while enabling collective benefit.

**Active inference**: Predict how interactions will update your model, choose next moves based on information value.

**Semantic gradients**: Navigate smoothly through semantic space rather than hopping between discrete categories.

These aren't roadmap items with versions - they're directions the system could evolve as we better understand the problem space.

## Getting Started

See [README.md](README.md) for installation and running the example.

See [Documentation Index](docs/INDEX.md) for full technical documentation.
