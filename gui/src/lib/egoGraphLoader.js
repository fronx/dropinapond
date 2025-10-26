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
 * Loads ego graph JSON data from the modular directory structure
 */
export async function loadEgoGraph(name) {
  try {
    // Load all files from the modular format
    const basePath = `/data/ego_graphs/${name}`;

    const [metadataRes, selfRes, edgesRes, contactPointsRes] = await Promise.all([
      fetch(`${basePath}/metadata.json`),
      fetch(`${basePath}/self.json`),
      fetch(`${basePath}/edges.json`),
      fetch(`${basePath}/contact_points.json`)
    ]);

    if (!metadataRes.ok) {
      throw new Error(`Failed to load metadata.json: ${metadataRes.status} ${metadataRes.statusText}`);
    }
    if (!selfRes.ok) {
      throw new Error(`Failed to load self.json: ${selfRes.status} ${selfRes.statusText}`);
    }
    if (!edgesRes.ok) {
      throw new Error(`Failed to load edges.json: ${edgesRes.status} ${edgesRes.statusText}`);
    }

    const [metadata, self, edges, contactPoints] = await Promise.all([
      metadataRes.json(),
      selfRes.json(),
      edgesRes.json(),
      contactPointsRes.ok ? contactPointsRes.json() : { past: [], present: [], potential: [] }
    ]);

    // Extract connection IDs from edges (since we don't have a manifest file)
    const uniqueIds = new Set();
    edges.forEach(edge => {
      if (edge.source !== self.id) uniqueIds.add(edge.source);
      if (edge.target !== self.id) uniqueIds.add(edge.target);
    });
    const connectionIds = Array.from(uniqueIds);

    // Load all connection files
    const connections = await Promise.all(
      connectionIds.map(async (id) => {
        const res = await fetch(`${basePath}/connections/${id}.json`);
        if (res.ok) {
          return await res.json();
        }
        console.warn(`Could not load connection ${id}`);
        return null;
      })
    );

    // Filter out failed loads
    const validConnections = connections.filter(c => c !== null);

    // Reconstruct the ego graph format expected by the UI
    return {
      version: metadata.version,
      format: metadata.format,
      metadata,
      self,
      connections: validConnections,
      edges,
      contact_points: contactPoints
    };
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

  // Build rank map from recommendations (handle both old + new formats)
  const rankMap = new Map();

  if (analysisData?.recommendations) {
    const recs = Array.isArray(analysisData.recommendations)
      ? analysisData.recommendations
      : analysisData.recommendations.semantic_suggestions || [];

    recs.forEach((rec, index) => {
      const target = rec.node_id || rec.target;
      if (target) rankMap.set(target, index + 1);
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
    const nodeId = connection.id;

    // Extract analysis metrics for this node
    const analysisMetrics = analysisData?.metrics ? {
      orientationScore: analysisData.metrics.orientation_scores?.[nodeId],
      readability: analysisData.metrics.per_neighbor_readability?.[nodeId],
      overlap: analysisData.metrics.overlaps?.[nodeId],
    } : {};

    // Get rank from recommendations
    const rank = rankMap.get(nodeId);

    nodes.push({
      id: connection.id,
      type: 'personNode',
      data: {
        person: connection,
        isSelf: false,
        connectionStrength: strengthMap.get(connection.id) || 0.3,
        clusterColor: clusterInfo?.color || '#d1d5db', // Default gray if no cluster
        clusterIndex: clusterInfo?.clusterIndex ?? null,
        analysisMetrics,
        rank,
      },
      position: { x: 0, y: 0 }, // Will be overridden by D3 layout
    });
  });

  // Add edges with recommendation-based styling
  egoData.edges.forEach((edge, index) => {
    const actualStrength = typeof edge.actual === 'number' ? edge.actual : 0.3;

    // Check if the target node is a top recommendation (rank 1-5)
    const targetRank = rankMap.get(edge.target);
    const isTopRecommendation = targetRank && targetRank <= 5;
    const isTopThree = targetRank && targetRank <= 3;

    // Style edges to top recommendations differently
    let strokeColor;
    let strokeWidth;

    if (isTopThree) {
      // Top 3: bright, prominent edges
      strokeColor = 'rgba(59, 130, 246, 0.9)'; // Bright blue for top 3
      strokeWidth = Math.max(2.5, actualStrength * 20);
    } else if (isTopRecommendation) {
      // Rank 4-5: medium brightness
      strokeColor = 'rgba(96, 165, 250, 0.7)'; // Medium blue
      strokeWidth = Math.max(2, actualStrength * 20);
    } else {
      // Default: subtle gray
      strokeColor = `rgba(100, 100, 100, ${0.3 + actualStrength * 0.4})`;
      strokeWidth = Math.max(1, actualStrength * 20);
    }

    edges.push({
      id: `${edge.source}-${edge.target}-${index}`,
      source: edge.source,
      target: edge.target,
      type: 'default',
      data: {
        actualStrength,
        potential: edge.potential,
        metadata: edge.metadata,
        rank: targetRank,
      },
      style: {
        strokeWidth,
        stroke: strokeColor,
      },
      animated: false, // Animate top 3 for extra emphasis
    });
  });

  // Extract cluster-level and overall metrics
  const clusterMetrics = analysisData?.metrics ? {
    publicLegibilityPerCluster: analysisData.metrics.public_legibility_per_cluster,
    subjectiveAttunementPerCluster: analysisData.metrics.subjective_attunement_per_cluster,
    heatResidualNovelty: analysisData.metrics.heat_residual_novelty,
  } : null;

  const overallMetrics = analysisData?.metrics ? {
    attentionEntropy: analysisData.metrics.attention_entropy,
    publicLegibilityOverall: analysisData.metrics.public_legibility_overall,
  } : null;

  const recommendations = analysisData?.recommendations || null;

  // Build a map of node IDs to names for display
  const nodeNameMap = {};
  nodeNameMap[egoData.self.id] = egoData.self.name;
  egoData.connections.forEach(connection => {
    nodeNameMap[connection.id] = connection.name;
  });

  return {
    nodes,
    edges,
    clusterMetrics,
    overallMetrics,
    recommendations,
    nodeNameMap
  };
}
