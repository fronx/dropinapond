import { useEffect, useState, useCallback, useRef } from 'react';
import { useParams } from 'react-router-dom';
import {
  ReactFlow,
  Background,
  useNodesState,
  useEdgesState,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { PersonNode } from './PersonNode';
import { PersonDetailSidebar } from './PersonDetailSidebar';
import { loadEgoGraph, loadLatestAnalysis, parseEgoGraphForFlow } from '../lib/egoGraphLoader';
import { createForceSimulation } from '../lib/d3Layout';
import { updateEdgeHandles } from '../lib/edgeUtils';

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
  const [analysisData, setAnalysisData] = useState(null);
  const [egoGraphData, setEgoGraphData] = useState(null);
  const [selectedPerson, setSelectedPerson] = useState(null);
  const simulationRef = useRef(null);

  useEffect(() => {
    async function loadGraph() {
      try {
        setLoading(true);
        setError(null);

        // Load ego graph data
        const egoData = await loadEgoGraph(graphName || 'fronx');
        setMetadata(egoData.metadata);
        setEgoGraphData(egoData);

        // Load analysis data (if available)
        const analysisData = await loadLatestAnalysis(graphName || 'fronx');
        if (analysisData) {
          console.log('Loaded analysis data:', analysisData);
        }

        // Parse into xyflow format with analysis data
        const {
          nodes: parsedNodes,
          edges: parsedEdges,
          clusterMetrics,
          overallMetrics,
          recommendations,
          nodeNameMap
        } = parseEgoGraphForFlow(egoData, analysisData);

        console.log('Loaded nodes:', parsedNodes);
        console.log('Loaded edges:', parsedEdges);

        // Store analysis data for the panel and sidebar
        // Keep both the raw analysis data and the parsed metrics
        setAnalysisData({
          metrics: analysisData?.metrics, // Raw metrics from analysis file
          clusterMetrics,
          overallMetrics,
          recommendations,
          nodeNameMap
        });

        setEdges(parsedEdges);

        // Start animated D3 force-directed layout
        simulationRef.current = createForceSimulation(
          parsedNodes,
          parsedEdges,
          (updatedNodes) => {
            setNodes(updatedNodes);
            // Update edge handles based on new node positions
            const updatedEdges = updateEdgeHandles(parsedEdges, updatedNodes);
            setEdges(updatedEdges);
          }
        );

        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    }

    loadGraph();

    // Cleanup: stop simulation when component unmounts
    return () => {
      if (simulationRef.current) {
        simulationRef.current.stop();
      }
    };
  }, [graphName, setNodes, setEdges]);

  // Handle node click to open sidebar
  const handleNodeClick = useCallback((event, node) => {
    const personData = node.data.person;
    setSelectedPerson(personData);
  }, []);

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
    <div style={{ width: '100vw', height: '100vh', display: 'flex', flexDirection: 'column', position: 'relative' }}>
      <div style={{ flex: 1, width: '100%', position: 'relative' }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={handleNodeClick}
          nodeTypes={nodeTypes}
          elevateEdgesOnSelect={true}
          fitView
          fitViewOptions={{ padding: 0.2, minZoom: 0.1, maxZoom: 2 }}
          minZoom={0.1}
          maxZoom={2}
        >
          <Background />
          {/* <Controls /> */}
        </ReactFlow>
      </div>

      {/* Person Detail Sidebar (left side) */}
      {selectedPerson && egoGraphData && (
        <PersonDetailSidebar
          person={selectedPerson}
          egoGraphData={egoGraphData}
          analysisData={analysisData}
          onClose={() => setSelectedPerson(null)}
        />
      )}
    </div>
  );
}
