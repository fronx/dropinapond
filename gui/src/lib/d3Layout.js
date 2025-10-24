import * as d3 from 'd3';

/**
 * Apply D3 force-directed layout to xyflow nodes based on edge strengths
 */
export function applyForceLayout(nodes, edges, options = {}) {
  const {
    width = 1200,
    height = 800,
    strength = -400,
    distance = 150,
    iterations = 300,
  } = options;

  // Create a map for quick node lookup
  const nodeMap = new Map(nodes.map(node => [node.id, node]));

  // Convert edges to D3 format with link strength based on actual connection strength
  const d3Links = edges.map(edge => ({
    source: edge.source,
    target: edge.target,
    strength: edge.data?.actualStrength || 0.5,
  }));

  // Create D3 simulation
  const simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(d3Links)
      .id(d => d.id)
      .distance(d => distance / (d.strength + 0.1)) // Stronger connections = closer nodes
      .strength(d => d.strength * 0.7) // Use edge strength for link force
    )
    .force('charge', d3.forceManyBody().strength(strength))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide().radius(60))
    .stop();

  // Run simulation synchronously
  for (let i = 0; i < iterations; i++) {
    simulation.tick();
  }

  // Update node positions
  nodes.forEach(node => {
    const simNode = nodeMap.get(node.id);
    if (simNode) {
      node.position = {
        x: simNode.x || 0,
        y: simNode.y || 0,
      };
    }
  });

  return nodes;
}
