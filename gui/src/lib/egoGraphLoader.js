import distinctColors from 'distinct-colors';

// Generate visually distinct colors for clusters
function generateClusterColors(count) {
  if (count === 0) return [];

  const palette = distinctColors({
    count,
    lightMin: 40,
    lightMax: 80,
    chromaMin: 40,
    chromaMax: 100,
    quality: 50,
  });

  return palette.map(color => color.hex());
}

const FOCAL_NODE_COLOR = '#777777';

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
 * Loads the latest analysis JSON for a given ego graph
 * Returns null if no analysis exists
 */
export async function loadLatestAnalysis(name) {
  try {
    // Try to load analyses directory listing (you'll need to implement a way to find the latest file)
    // For now, we'll construct the expected latest filename pattern
    // In production, you might want to maintain a "latest.json" symlink or index file

    // For now, try to fetch with a wildcard approach or accept it might fail
    // We'll implement a simple approach: look for a _latest.json file
    const response = await fetch(`/data/analyses/${name}_latest.json`);
    if (!response.ok) {
      console.warn(`No analysis found for "${name}", proceeding without analysis data`);
      return null;
    }
    return await response.json();
  } catch (error) {
    console.warn(`Could not load analysis for "${name}":`, error.message);
    return null;
  }
}

/**
 * Parses ego graph data into xyflow-compatible nodes and edges
 * @param {object} egoData - Ego graph JSON data
 * @param {object|null} analysisData - Optional analysis data with clusters
 */
export function parseEgoGraphForFlow(egoData, analysisData = null) {
  const nodes = [];
  const edges = [];

  // Build a map of connection strengths from focal node to each person
  const strengthMap = new Map();
  egoData.edges.forEach((edge) => {
    if (edge.source === egoData.self.id) {
      strengthMap.set(edge.target, edge.actual || 0.3);
    }
  });

  // Build cluster assignment map if analysis data is available
  const clusterMap = new Map(); // nodeId -> { clusterIndex, color }
  if (analysisData?.metrics?.clusters) {
    const clusterColors = generateClusterColors(analysisData.metrics.clusters.length);
    analysisData.metrics.clusters.forEach((cluster, clusterIndex) => {
      const color = clusterColors[clusterIndex];
      cluster.forEach((nodeId) => {
        clusterMap.set(nodeId, { clusterIndex, color });
      });
    });
  }

  // Add self node (focal node)
  nodes.push({
    id: egoData.self.id,
    type: 'personNode',
    data: {
      person: egoData.self,
      isSelf: true,
      connectionStrength: 1.0, // Focal node is always max strength
      clusterColor: FOCAL_NODE_COLOR,
      clusterIndex: null,
    },
    position: { x: 0, y: 0 }, // Will be overridden by D3 layout
  });

  // Add connection nodes
  egoData.connections.forEach((connection) => {
    const clusterInfo = clusterMap.get(connection.id);
    nodes.push({
      id: connection.id,
      type: 'personNode',
      data: {
        person: connection,
        isSelf: false,
        connectionStrength: strengthMap.get(connection.id) || 0.3,
        clusterColor: clusterInfo?.color || '#d1d5db', // Default gray if no cluster
        clusterIndex: clusterInfo?.clusterIndex ?? null,
      },
      position: { x: 0, y: 0 }, // Will be overridden by D3 layout
    });
  });

  // Add edges
  egoData.edges.forEach((edge, index) => {
    const actualStrength = typeof edge.actual === 'number' ? edge.actual : 0.3;

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
        strokeWidth: Math.max(1, actualStrength * 20), // Thicker based on strength
        stroke: `rgba(100, 100, 100, ${0.4 + actualStrength * 0.6})`, // More opaque for stronger connections
      },
      animated: false,
    });
  });

  return { nodes, edges };
}
