/**
 * Loads ego graph JSON data from the data directory
 */
export async function loadEgoGraph(name) {
  try {
    // In development, we'll use Vite's dynamic import
    // In production, you might need to serve the data directory differently
    const response = await fetch(`/data/ego_graphs/${name}.json`);
    if (!response.ok) {
      throw new Error(`Failed to load ego graph: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`Error loading ego graph "${name}":`, error);
    throw error;
  }
}

/**
 * Parses ego graph data into xyflow-compatible nodes and edges
 */
export function parseEgoGraphForFlow(egoData) {
  const nodes = [];
  const edges = [];

  // Add self node (focal node)
  nodes.push({
    id: egoData.self.id,
    type: 'personNode',
    data: {
      person: egoData.self,
      isSelf: true,
    },
    position: { x: 0, y: 0 }, // Will be overridden by D3 layout
  });

  // Add connection nodes
  egoData.connections.forEach((connection) => {
    nodes.push({
      id: connection.id,
      type: 'personNode',
      data: {
        person: connection,
        isSelf: false,
      },
      position: { x: 0, y: 0 }, // Will be overridden by D3 layout
    });
  });

  // Add edges
  egoData.edges.forEach((edge, index) => {
    const actualStrength = typeof edge.actual === 'number' ? edge.actual : 0.5;

    edges.push({
      id: `${edge.source}-${edge.target}-${index}`,
      source: edge.source,
      target: edge.target,
      type: 'default',
      data: {
        actualStrength,
        potential: edge.potential,
        metadata: edge.metadata,
      },
      style: {
        strokeWidth: Math.max(1, actualStrength * 5),
        stroke: `rgba(100, 100, 100, ${0.3 + actualStrength * 0.7})`,
      },
      label: actualStrength > 0 ? `${(actualStrength * 100).toFixed(0)}%` : undefined,
      labelStyle: { fontSize: 10, fill: '#666' },
      labelBgStyle: { fill: 'white', fillOpacity: 0.7 },
    });
  });

  return { nodes, edges };
}
