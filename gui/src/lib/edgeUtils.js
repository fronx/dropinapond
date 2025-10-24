/**
 * Calculate optimal connection handles based on relative node positions
 * This makes edges connect to the nearest side of each node
 */
export function calculateOptimalHandles(sourceNode, targetNode) {
  const deltaX = targetNode.position.x - sourceNode.position.x;
  const deltaY = targetNode.position.y - sourceNode.position.y;

  let sourceHandle = 'source-right';
  let targetHandle = 'target-left';

  if (Math.abs(deltaX) > Math.abs(deltaY)) {
    // Horizontal connection
    if (deltaX > 0) {
      sourceHandle = 'source-right';
      targetHandle = 'target-left';
    } else {
      sourceHandle = 'source-left';
      targetHandle = 'target-right';
    }
  } else {
    // Vertical connection
    if (deltaY > 0) {
      sourceHandle = 'source-bottom';
      targetHandle = 'target-top';
    } else {
      sourceHandle = 'source-top';
      targetHandle = 'target-bottom';
    }
  }

  return { sourceHandle, targetHandle };
}

/**
 * Update edges with optimal handle positions based on current node positions
 */
export function updateEdgeHandles(edges, nodes) {
  const nodeMap = new Map(nodes.map(node => [node.id, node]));

  return edges.map(edge => {
    const sourceNode = nodeMap.get(edge.source);
    const targetNode = nodeMap.get(edge.target);

    if (!sourceNode || !targetNode) {
      return edge;
    }

    const { sourceHandle, targetHandle } = calculateOptimalHandles(sourceNode, targetNode);

    return {
      ...edge,
      sourceHandle,
      targetHandle,
    };
  });
}
