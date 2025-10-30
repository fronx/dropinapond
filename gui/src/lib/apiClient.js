/**
 * API client for backend communication
 *
 * The backend auto-detects data source (Neo4j or files) based on environment variables.
 * The frontend doesn't need to know which source is being used.
 */

// Get backend URL from environment variable, default to localhost:3001
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3001';

/**
 * Fetch ego graph from backend
 * Returns graph structure with nodes, edges, focal node ID, and names mapping
 */
export async function fetchGraphFromAPI() {
  try {
    console.log('[API] Fetching graph from:', `${API_URL}/api/graph`);
    const response = await fetch(`${API_URL}/api/graph`);

    if (!response.ok) {
      throw new Error(`Failed to fetch graph: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    console.log('[API] Graph data received:', data);

    // Transform backend format to match what parseEgoGraphForFlow expects
    // Backend returns: { nodes: [...ids], focal: "id", edges: [{source, target, weight}], names: {id: name} }
    // We need to reconstruct the ego graph format with self and connections

    const focalId = data.focal;
    const nodeIds = data.nodes;
    const edges = data.edges;
    const names = data.names;

    // Create self object
    const self = {
      id: focalId,
      name: names[focalId],
      phrases: [] // Backend doesn't return phrases (not needed for visualization)
    };

    // Create connections objects
    const connections = nodeIds
      .filter(id => id !== focalId)
      .map(id => ({
        id: id,
        name: names[id],
        phrases: [],
        capabilities: [],
        availability: [],
        notes: []
      }));

    // Transform edges to match expected format
    const transformedEdges = edges.map(edge => ({
      source: edge.source,
      target: edge.target,
      actual: edge.weight
    }));

    return {
      version: '0.2',
      format: 'api',
      self,
      connections,
      edges: transformedEdges,
      contact_points: { past: [], present: [], potential: [] }
    };
  } catch (error) {
    console.error('[API] Error fetching graph:', error);
    throw error;
  }
}

/**
 * Fetch analysis results from backend
 * Returns analysis data with metrics, clusters, and recommendations
 */
export async function fetchAnalysisFromAPI() {
  try {
    console.log('[API] Fetching analysis from:', `${API_URL}/api/analysis`);
    const response = await fetch(`${API_URL}/api/analysis`);

    if (!response.ok) {
      console.warn('[API] No analysis found');
      return null;
    }

    const data = await response.json();

    // Check for error response
    if (data.error) {
      console.warn('[API] Analysis not available:', data.error);
      return null;
    }

    console.log('[API] Analysis data received');
    return data;
  } catch (error) {
    console.error('[API] Error fetching analysis:', error);
    return null;
  }
}

/**
 * Health check endpoint
 */
export async function checkHealth() {
  try {
    const response = await fetch(`${API_URL}/health`);
    if (!response.ok) {
      return false;
    }
    const data = await response.json();
    return data.status === 'ok';
  } catch (error) {
    console.error('[API] Health check failed:', error);
    return false;
  }
}
