import { BaseEdge, getBezierPath, EdgeLabelRenderer } from '@xyflow/react';
import { useState } from 'react';

export function DiffusionEdge({ id, sourceX, sourceY, targetX, targetY, data }) {
  const { probability, sourceId, targetId } = data;
  const [hovered, setHovered] = useState(false);

  // Color gradient based on absolute probability (not relative to max)
  const normalized = Math.min(probability * 5, 1); // Scale: 20% prob = fully saturated
  const r = Math.round(255 * normalized);
  const b = Math.round(255 * (1 - normalized));
  const alpha = hovered ? 0.8 : (0.3 + (normalized * 0.5));

  const strokeColor = `rgba(${r}, ${Math.round(b / 2)}, ${b}, ${alpha})`;
  // Square root keeps thin lines thin, square/cube makes thick lines much thicker
  const sqrtProb = Math.sqrt(probability);
  const baseWidth = Math.pow(sqrtProb, 2) * 10;
  const strokeWidth = hovered ? baseWidth * 1.5 : baseWidth;

  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
  });

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        style={{
          stroke: strokeColor,
          strokeWidth: strokeWidth,
          transition: 'all 0.2s',
        }}
        markerEnd={{
          type: 'arrowclosed',
          color: strokeColor,
          width: 12,
          height: 12,
        }}
      />
      {/* Invisible wider path for easier hovering */}
      <path
        d={edgePath}
        fill="none"
        stroke="transparent"
        strokeWidth={20}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        style={{ cursor: 'pointer' }}
      />

      {/* Hover label */}
      {hovered && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
              pointerEvents: 'none',
              backgroundColor: 'white',
              padding: '4px 8px',
              borderRadius: '4px',
              fontSize: '12px',
              boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
              border: '1px solid #e5e7eb',
            }}
          >
            <div style={{ fontWeight: 600, marginBottom: '2px' }}>
              {sourceId} → {targetId}
            </div>
            <div style={{ color: '#6b7280' }}>
              Flow: {(probability * 100).toFixed(2)}%
            </div>
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}
