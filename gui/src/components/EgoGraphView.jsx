import { useEffect, useState, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { PersonNode } from './PersonNode';
import { loadEgoGraph, parseEgoGraphForFlow } from '../lib/egoGraphLoader';
import { applyForceLayout } from '../lib/d3Layout';

const nodeTypes = {
  personNode: PersonNode,
};

export function EgoGraphView() {
  const { graphName } = useParams();
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [metadata, setMetadata] = useState(null);

  useEffect(() => {
    async function loadGraph() {
      try {
        setLoading(true);
        setError(null);

        // Load ego graph data
        const egoData = await loadEgoGraph(graphName || 'fronx');
        setMetadata(egoData.metadata);

        // Parse into xyflow format
        const { nodes: parsedNodes, edges: parsedEdges } = parseEgoGraphForFlow(egoData);

        // Apply D3 force-directed layout
        const layoutNodes = applyForceLayout(parsedNodes, parsedEdges, {
          width: 1200,
          height: 800,
          strength: -500,
          distance: 200,
          iterations: 300,
        });

        console.log('Loaded nodes:', layoutNodes);
        console.log('Loaded edges:', parsedEdges);

        setNodes(layoutNodes);
        setEdges(parsedEdges);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }

    loadGraph();
  }, [graphName, setNodes, setEdges]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-xl text-gray-600">Loading ego graph...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-screen">
        <div className="text-xl text-red-600 mb-4">Error loading graph</div>
        <div className="text-gray-600">{error}</div>
      </div>
    );
  }

  return (
    <div style={{ width: '100vw', height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <div style={{
        backgroundColor: 'white',
        borderBottom: '1px solid #e5e7eb',
        padding: '1rem 1.5rem'
      }}>
        <h1 style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#1f2937' }}>
          Ego Graph: {graphName || 'fronx'}
        </h1>
        {metadata?.description && (
          <p style={{ fontSize: '0.875rem', color: '#4b5563', marginTop: '0.25rem' }}>
            {metadata.description}
          </p>
        )}
        <div style={{ fontSize: '0.75rem', color: '#9ca3af', marginTop: '0.25rem' }}>
          {nodes.length} nodes, {edges.length} edges
        </div>
      </div>

      {/* Graph visualization */}
      <div style={{ flex: 1, width: '100%' }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.2, minZoom: 0.1, maxZoom: 2 }}
          minZoom={0.1}
          maxZoom={2}
        >
          <Background />
          <Controls />
          <MiniMap
            nodeColor={(node) => node.data?.isSelf ? '#3b82f6' : '#d1d5db'}
            nodeStrokeWidth={3}
            zoomable
            pannable
          />
        </ReactFlow>
      </div>
    </div>
  );
}
