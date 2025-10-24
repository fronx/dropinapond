import * as d3 from 'd3';

/**
 * Apply D3 force-directed layout to xyflow nodes based on edge strengths
 * Returns a simulation that can be used for animated layout
 */
export function createForceSimulation(nodes, edges, onTick, options = {}) {
  const {
    width = 500,
    height = 500,
    strength = -300,
    distance = 150,
  } = options;

  // Convert edges to D3 format with link strength based on actual connection strength
  const d3Links = edges.map(edge => ({
    source: edge.source,
    target: edge.target,
    strength: edge.data?.actualStrength || 0.3,
  }));

  // Create D3 simulation
  const simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(d3Links)
      .id(d => d.id)
      .distance(d => distance * Math.exp(-d.strength * 2)) // Exponential: stronger connections = exponentially closer nodes
      .strength(d => 0.3 * Math.exp(d.strength * 1.5)) // Exponential strength: stronger edges pull harder
    )
    .force('charge', d3.forceManyBody()
      .strength(strength)
      .distanceMax(400) // Limit repulsion range to create clustering while avoiding overlaps
    )
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide()
      .radius(60) // Larger radius to spread nodes out
      .strength(0.7) // Moderate collision strength
      .iterations(2)
    )
    .on('tick', () => {
      // Update node positions on each tick
      const updatedNodes = nodes.map(node => ({
        ...node,
        position: { x: node.x || 0, y: node.y || 0 },
      }));
      onTick(updatedNodes);
    });

  return simulation;
}

/**
 * Synchronous layout (for initial positioning)
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
      .distance(d => distance * Math.exp(-d.strength * 2)) // Exponential: stronger connections = exponentially closer nodes
      .strength(d => Math.exp(d.strength * 2) - 1) // Exponential strength: stronger edges pull harder
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
