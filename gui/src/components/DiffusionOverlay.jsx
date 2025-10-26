import { useState } from 'react';

const ROLE_COLORS = {
  core: '#ef4444',
  bridge: '#f59e0b',
  isolated: '#6b7280',
  even: '#3b82f6',
};

export function DiffusionOverlay({ diffusionData, visible, nodes }) {
  const [timeStep, setTimeStep] = useState('t2');
  const [hoveredEdge, setHoveredEdge] = useState(null);

  if (!visible || !diffusionData) return null;

  const { node_order, matrices, node_metrics } = diffusionData;
  const matrix = matrices[timeStep];

  // Build map of node_id -> position from ReactFlow nodes
  const nodePositions = {};
  nodes.forEach(node => {
    nodePositions[node.id] = { x: node.position.x, y: node.position.y };
  });

  // Find max probability for color scaling (excluding diagonal)
  let maxProb = 0;
  matrix.forEach((row, i) => {
    row.forEach((val, j) => {
      if (i !== j && val > maxProb) maxProb = val;
    });
  });

  // Color for diffusion edges (gradient from blue to red)
  const getDiffusionColor = (prob) => {
    if (prob < 0.001) return 'rgba(200, 200, 200, 0.1)';

    const normalized = Math.min(prob / (maxProb * 0.5), 1); // Scale to make colors more visible
    const r = Math.round(255 * normalized);
    const b = Math.round(255 * (1 - normalized));
    const alpha = 0.3 + (normalized * 0.5); // More visible for higher probabilities

    return `rgba(${r}, ${b/2}, ${b}, ${alpha})`;
  };

  const getStrokeWidth = (prob) => {
    if (prob < 0.001) return 0.5;
    const normalized = Math.min(prob / (maxProb * 0.5), 1);
    return 1 + (normalized * 8); // 1-9px range
  };

  return (
    <div style={{
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      pointerEvents: 'none',
      zIndex: 5
    }}>
      {/* Time step controls */}
      <div style={{
        position: 'absolute',
        top: '4rem',
        left: '50%',
        transform: 'translateX(-50%)',
        backgroundColor: 'white',
        borderRadius: '0.5rem',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        padding: '0.5rem',
        display: 'flex',
        gap: '0.5rem',
        alignItems: 'center',
        pointerEvents: 'auto',
        zIndex: 20
      }}>
        <span style={{ fontSize: '0.875rem', fontWeight: '500', color: '#374151' }}>
          Diffusion steps:
        </span>
        {['t1', 't2', 't3'].map(t => (
          <button
            key={t}
            onClick={() => setTimeStep(t)}
            style={{
              padding: '0.375rem 0.75rem',
              borderRadius: '0.375rem',
              border: 'none',
              cursor: 'pointer',
              backgroundColor: timeStep === t ? '#3b82f6' : '#f3f4f6',
              color: timeStep === t ? 'white' : '#374151',
              fontWeight: '500',
              fontSize: '0.875rem'
            }}
          >
            {t === 't1' ? '1' : t === 't2' ? '2' : '3'}
          </button>
        ))}

        <div style={{
          marginLeft: '0.5rem',
          paddingLeft: '0.5rem',
          borderLeft: '1px solid #e5e7eb',
          fontSize: '0.75rem',
          color: '#6b7280'
        }}>
          Color intensity = flow probability
        </div>
      </div>

      {/* SVG overlay for diffusion edges */}
      <svg style={{ width: '100%', height: '100%', position: 'absolute' }}>
        <defs>
          {/* Arrow marker for directed flow */}
          <marker
            id="diffusion-arrow"
            markerWidth="10"
            markerHeight="10"
            refX="9"
            refY="3"
            orient="auto"
            markerUnits="strokeWidth"
          >
            <path d="M0,0 L0,6 L9,3 z" fill="rgba(100,100,100,0.4)" />
          </marker>
        </defs>

        {/* Draw all diffusion edges */}
        {node_order.map((sourceId, i) => {
          const sourcePos = nodePositions[sourceId];
          if (!sourcePos) return null;

          return node_order.map((targetId, j) => {
            if (i === j) return null; // Skip diagonal

            const targetPos = nodePositions[targetId];
            if (!targetPos) return null;

            const prob = matrix[i][j];
            if (prob < 0.001) return null; // Skip very low probabilities

            const isHovered = hoveredEdge?.i === i && hoveredEdge?.j === j;

            // Offset positions to node center (assuming 40x40 nodes)
            const x1 = sourcePos.x + 20;
            const y1 = sourcePos.y + 20;
            const x2 = targetPos.x + 20;
            const y2 = targetPos.y + 20;

            // Create curved path for better visibility
            const dx = x2 - x1;
            const dy = y2 - y1;
            const dr = Math.sqrt(dx * dx + dy * dy);
            const curve = dr * 0.3; // Curve amount

            return (
              <g key={`${i}-${j}`}>
                <path
                  d={`M ${x1},${y1} Q ${x1 + dx/2 + dy/4},${y1 + dy/2 - dx/4} ${x2},${y2}`}
                  stroke={getDiffusionColor(prob)}
                  strokeWidth={isHovered ? getStrokeWidth(prob) * 1.5 : getStrokeWidth(prob)}
                  fill="none"
                  markerEnd="url(#diffusion-arrow)"
                  style={{
                    pointerEvents: 'stroke',
                    cursor: 'pointer',
                    transition: 'all 0.2s'
                  }}
                  onMouseEnter={(e) => {
                    e.target.style.pointerEvents = 'auto';
                    setHoveredEdge({ i, j, sourceId, targetId, prob });
                  }}
                  onMouseLeave={() => setHoveredEdge(null)}
                />
              </g>
            );
          });
        })}
      </svg>

      {/* Hover tooltip */}
      {hoveredEdge && (
        <div style={{
          position: 'absolute',
          bottom: '1rem',
          left: '50%',
          transform: 'translateX(-50%)',
          backgroundColor: 'white',
          padding: '0.75rem',
          borderRadius: '0.5rem',
          boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
          pointerEvents: 'none',
          zIndex: 30,
          minWidth: '200px'
        }}>
          <div style={{ fontWeight: '600', marginBottom: '0.25rem', fontSize: '0.875rem' }}>
            {hoveredEdge.sourceId} → {hoveredEdge.targetId}
          </div>
          <div style={{ color: '#6b7280', fontSize: '0.875rem' }}>
            Flow probability: {(hoveredEdge.prob * 100).toFixed(2)}%
          </div>
          <div style={{ color: '#9ca3af', fontSize: '0.75rem', marginTop: '0.25rem' }}>
            After {timeStep === 't1' ? '1' : timeStep === 't2' ? '2' : '3'} diffusion steps
          </div>
        </div>
      )}
    </div>
  );
}
