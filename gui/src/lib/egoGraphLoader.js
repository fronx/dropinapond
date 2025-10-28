import distinctColors from 'distinct-colors';
import chroma from 'chroma-js';

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
    const url = `/data/analyses/${name}_latest.json`;
    console.log('Fetching analysis from:', url);
    const response = await fetch(url);
    console.log('Analysis fetch response:', response.status, response.ok);
    if (!response.ok) {
      console.warn(`No analysis found for "${name}", proceeding without analysis data`);
      return null;
    }

    // Get the text first to handle Infinity values
    const text = await response.text();
    // Replace Infinity with a very large number that JSON can handle
    // We'll use a special marker value that we can detect later
    const sanitizedText = text.replace(/:\s*Infinity\b/g, ': 9999999999');
    const data = JSON.parse(sanitizedText);

    console.log('Analysis data loaded:', data);
    return data;
  } catch (error) {
    console.error(`Could not load analysis for "${name}":`, error);
    return null;
  }
}

/**
 * Parses ego graph data into xyflow-compatible nodes and edges
 * @param {object} egoData - Ego graph JSON data
 * @param {object} analysisData - Analysis data with effective edges and clusters
 */
export function parseEgoGraphForFlow(egoData, analysisData) {
  const nodes = [];
  const edges = [];

  console.log('parseEgoGraphForFlow called with:', { egoData, analysisData });

  // Check if we have the necessary data
  if (!analysisData || !analysisData.metrics) {
    console.error('Missing analysisData or analysisData.metrics');
    throw new Error('Analysis data is missing or invalid');
  }

  // Use effective edges from analysis (blend of structural + semantic)
  const effectiveEdges = analysisData.metrics.layers.effective_edges;

  // Build a map of connection strengths from focal node to each person using raw edges
  // (for node sizing - we want this based on actual relationship strength, not computed similarity)
  const strengthMap = new Map();
  egoData.edges.forEach(edge => {
    if (edge.source === egoData.self.id) {
      strengthMap.set(edge.target, edge.actual);
    }
  });

  // Build cluster assignment map
  const clusterMap = new Map(); // nodeId -> { clusterIndex, color }
  const clusterColors = generateClusterColors(analysisData.metrics.clusters.length);
  analysisData.metrics.clusters.forEach((cluster, clusterIndex) => {
    const color = clusterColors[clusterIndex];
    cluster.forEach((nodeId) => {
      clusterMap.set(nodeId, { clusterIndex, color });
    });
  });

  // Build a set of edges to highlight
  // TODO: This can be used to highlight specific edges of interest
  // For example: edges representing new recommended connections, high-leverage introductions,
  // or edges with particular semantic properties
  // Format: Set of "source->target" strings
  const highlightedEdges = new Set([
    // Example: 'alice->bob', 'carol->dave'
  ]);

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

  // Calculate fit thresholds for fit_ratio categorization (if coherence data exists)
  let top30PercentileFitRatio = 3;
  let top30PercentileFitDiff = 0;

  if (analysisData.metrics.coherence?.nodes) {
    const fitRatios = [];
    const fitDiffs = [];
    Object.values(analysisData.metrics.coherence.nodes).forEach(nodeData => {
      // Skip Infinity values (represented as 9999999999 after sanitization)
      if (nodeData.fit_ratio !== 9999999999 && nodeData.fit_ratio !== Infinity) {
        fitRatios.push(nodeData.fit_ratio);
      }
      fitDiffs.push(nodeData.fit_diff);
    });
    fitRatios.sort((a, b) => b - a); // Sort descending
    fitDiffs.sort((a, b) => b - a); // Sort descending
    const top30PercentileRatioIndex = Math.floor(fitRatios.length * 0.3);
    top30PercentileFitRatio = fitRatios[top30PercentileRatioIndex] || 3;
    const top30PercentileDiffIndex = Math.floor(fitDiffs.length * 0.3);
    top30PercentileFitDiff = fitDiffs[top30PercentileDiffIndex] || 0;
  }

  // Add connection nodes
  egoData.connections.forEach((connection) => {
    const clusterInfo = clusterMap.get(connection.id);
    const nodeId = connection.id;

    // Extract analysis metrics for this node (if they exist in this analysis format)
    const analysisMetrics = {
      orientationScore: analysisData.metrics.orientation_scores?.[nodeId],
      readability: analysisData.metrics.per_neighbor_readability?.[nodeId],
      overlap: analysisData.metrics.overlaps?.[nodeId],
    };

    // Calculate fit category from coherence data
    const coherenceData = analysisData.metrics.coherence?.nodes?.[nodeId];
    let fitCategory = 'none'; // none, misfit, borderline, strong
    if (coherenceData) {
      const { fit_ratio, fit_diff } = coherenceData;
      if (fit_ratio === Infinity || fit_ratio >= top30PercentileFitRatio || fit_diff >= top30PercentileFitDiff) {
        fitCategory = 'strong';
      } else if (fit_ratio >= 1 && fit_ratio < top30PercentileFitRatio) {
        fitCategory = 'borderline';
      } else if (fit_ratio < 1 || fit_diff <= 0) {
        fitCategory = 'misfit';
      }
    }

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
        fitCategory, // Add fit category for visualization
        coherenceData, // Include raw coherence data for debugging/tooltip
      },
      position: { x: 0, y: 0 }, // Will be overridden by D3 layout
    });
  });

  // Add edges from effective edges layer (all edges, not just from focal node)
  let edgeIndex = 0;
  Object.entries(effectiveEdges).forEach(([source, targets]) => {
    Object.entries(targets).forEach(([target, effectiveWeight]) => {
      // Check if this edge should be highlighted
      const edgeKey = `${source}->${target}`;
      const isHighlighted = highlightedEdges.has(edgeKey);

      // Check if this edge involves the ego node
      const involvesEgo = source === egoData.self.id || target === egoData.self.id;

      // Check if both nodes are in the same cluster
      const sourceCluster = clusterMap.get(source);
      const targetCluster = clusterMap.get(target);
      const sameCluster = sourceCluster && targetCluster &&
        sourceCluster.clusterIndex === targetCluster.clusterIndex;

      // Style edges based on cluster membership and highlighting
      let strokeColor;
      let strokeWidth;

      if (isHighlighted) {
        // Highlighted edges: bright blue
        strokeColor = 'rgba(59, 130, 246, 0.9)';
        strokeWidth = Math.max(2.5, effectiveWeight * 20);
      } else if (involvesEgo) {
        // Edges involving ego node: neutral gray, don't color by cluster
        strokeColor = `rgba(100, 100, 100, ${0.3 + effectiveWeight * 0.4})`;
        strokeWidth = Math.max(1, effectiveWeight * 20);
      } else if (sameCluster) {
        // Edges within the same cluster: use cluster color with transparency
        const alpha = 0.1 + effectiveWeight * 0.7;
        strokeColor = chroma(sourceCluster.color).alpha(alpha).css();
        strokeWidth = Math.max(1, effectiveWeight * 20);
      } else {
        // Default: subtle gray
        strokeColor = `rgba(100, 100, 100, ${0.3 + effectiveWeight * 0.4})`;
        strokeWidth = Math.max(1, effectiveWeight * 20);
      }

      edges.push({
        id: `${source}-${target}-${edgeIndex}`,
        source: source,
        target: target,
        type: 'default',
        data: {
          effectiveWeight,
          highlighted: isHighlighted,
        },
        style: {
          strokeWidth,
          stroke: strokeColor,
        },
        animated: false,
      });
      edgeIndex++;
    });
  });

  // Extract cluster-level and overall metrics (if they exist in this analysis format)
  const clusterMetrics = {
    publicLegibilityPerCluster: analysisData.metrics.public_legibility_per_cluster,
    subjectiveAttunementPerCluster: analysisData.metrics.subjective_attunement_per_cluster,
    heatResidualNovelty: analysisData.metrics.heat_residual_novelty,
  };

  const overallMetrics = {
    attentionEntropy: analysisData.metrics.attention_entropy,
    publicLegibilityOverall: analysisData.metrics.public_legibility_overall,
  };

  const recommendations = analysisData.recommendations;

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
