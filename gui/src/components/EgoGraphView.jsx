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
import { AnalysisPanel } from './AnalysisPanel';
import { PersonDetailSidebar } from './PersonDetailSidebar';
import { DiffusionEdge } from './DiffusionEdge';
import { loadEgoGraph, loadLatestAnalysis, parseEgoGraphForFlow } from '../lib/egoGraphLoader';
import { createForceSimulation } from '../lib/d3Layout';
import { updateEdgeHandles } from '../lib/edgeUtils';

const nodeTypes = {
  personNode: PersonNode,
};

const edgeTypes = {
  diffusion: DiffusionEdge,
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
  const [showDiffusion, setShowDiffusion] = useState(false);
  const [diffusionTimeStep, setDiffusionTimeStep] = useState('t2');
  const [originalEdges, setOriginalEdges] = useState([]);
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
        setOriginalEdges(parsedEdges); // Store original edges for toggling

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

  // Generate diffusion edges when toggled or time step changes
  useEffect(() => {
    if (!showDiffusion || !analysisData?.metrics?.kernel_neighborhoods?.diffusion_heatmap) {
      // Restore original edges
      if (originalEdges.length > 0) {
        setEdges(originalEdges);
      }
      return;
    }

    const diffusionData = analysisData.metrics.kernel_neighborhoods.diffusion_heatmap;
    const { node_order, matrices } = diffusionData;
    const matrix = matrices[diffusionTimeStep];

    // Collect all probabilities to find thresholds
    const allProbs = [];
    matrix.forEach((row, i) => {
      row.forEach((val, j) => {
        if (i !== j && val > 0.001) {
          allProbs.push(val);
        }
      });
    });

    // Sort and find the top 30% threshold (to reduce clutter)
    allProbs.sort((a, b) => b - a);
    const threshold = allProbs[Math.floor(allProbs.length * 0.3)] || 0.01;
    const maxProb = allProbs[0] || 1;

    console.log(`Diffusion: showing top 30% of flows (threshold=${threshold.toFixed(4)}, max=${maxProb.toFixed(4)})`);

    // Generate diffusion edges (only significant flows)
    const diffusionEdges = [];
    node_order.forEach((sourceId, i) => {
      node_order.forEach((targetId, j) => {
        if (i === j) return; // Skip self-loops
        const prob = matrix[i][j];
        if (prob < threshold) return; // Only show top 30% of flows

        diffusionEdges.push({
          id: `diff-${sourceId}-${targetId}`,
          source: sourceId,
          target: targetId,
          type: 'diffusion',
          data: {
            probability: prob,
            maxProbability: maxProb,
            sourceId: sourceId,
            targetId: targetId,
          },
        });
      });
    });

    console.log(`Generated ${diffusionEdges.length} diffusion edges`);

    setEdges(diffusionEdges);
  }, [showDiffusion, diffusionTimeStep, analysisData, originalEdges, setEdges]);

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

  const diffusionData = analysisData?.metrics?.kernel_neighborhoods?.diffusion_heatmap;

  return (
    <div style={{ width: '100vw', height: '100vh', display: 'flex', flexDirection: 'column', position: 'relative' }}>
      {/* Diffusion controls */}
      <div style={{
        position: 'absolute',
        top: '1rem',
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 10,
        backgroundColor: 'white',
        borderRadius: '0.5rem',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        padding: '0.5rem',
        display: 'flex',
        gap: '0.75rem',
        alignItems: 'center'
      }}>
        {/* <button
          onClick={() => setShowDiffusion(!showDiffusion)}
          disabled={!diffusionData}
          style={{
            padding: '0.5rem 1rem',
            borderRadius: '0.375rem',
            border: 'none',
            cursor: diffusionData ? 'pointer' : 'not-allowed',
            backgroundColor: showDiffusion ? '#3b82f6' : '#f3f4f6',
            color: showDiffusion ? 'white' : '#374151',
            fontWeight: '500',
            fontSize: '0.875rem',
            opacity: diffusionData ? 1 : 0.5
          }}
        >
          {showDiffusion ? '✓ ' : ''}Show Diffusion Flow
        </button> */}

        {showDiffusion && (
          <>
            <div style={{ width: '1px', height: '1.5rem', backgroundColor: '#e5e7eb' }} />
            <div style={{ display: 'flex', gap: '0.25rem', alignItems: 'center' }}>
              <span style={{ fontSize: '0.875rem', color: '#6b7280', marginRight: '0.25rem' }}>
                Steps:
              </span>
              {['t1', 't2', 't3'].map(t => (
                <button
                  key={t}
                  onClick={() => setDiffusionTimeStep(t)}
                  style={{
                    padding: '0.25rem 0.5rem',
                    borderRadius: '0.25rem',
                    border: 'none',
                    cursor: 'pointer',
                    backgroundColor: diffusionTimeStep === t ? '#3b82f6' : '#f3f4f6',
                    color: diffusionTimeStep === t ? 'white' : '#374151',
                    fontWeight: '500',
                    fontSize: '0.75rem'
                  }}
                >
                  {t === 't1' ? '1' : t === 't2' ? '2' : '3'}
                </button>
              ))}
            </div>
          </>
        )}
      </div>

      <div style={{ flex: 1, width: '100%', position: 'relative' }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={handleNodeClick}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
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

      {/* Analysis Panel (right side) */}
      {analysisData && (
        <AnalysisPanel
          clusterMetrics={analysisData.clusterMetrics}
          overallMetrics={analysisData.overallMetrics}
          recommendations={analysisData.recommendations}
          nodeNameMap={analysisData.nodeNameMap}
        />
      )}
    </div>
  );
}
