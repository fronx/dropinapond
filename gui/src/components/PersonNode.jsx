import { Handle, Position } from '@xyflow/react';

export function PersonNode({ data, selected }) {
  const { person, isSelf } = data;
  const latestAvailability = person.availability?.[0];

  const nodeStyle = {
    position: 'relative',
    padding: '8px 16px',
    borderRadius: '6px',
    boxShadow: selected
      ? '0 0 0 3px #c084fc, 0 4px 6px -1px rgba(0, 0, 0, 0.2)'
      : '0 2px 4px rgba(0, 0, 0, 0.1)',
    transition: 'all 0.2s',
    border: isSelf ? '2px solid #3b82f6' : '2px solid #d1d5db',
    backgroundColor: isSelf ? '#eff6ff' : 'white',
    cursor: 'pointer',
    fontSize: '0.875rem',
    fontWeight: '600',
    color: isSelf ? '#1e40af' : '#1f2937',
  };

  return (
    <div style={nodeStyle}>
      {/* Handles for edge connections */}
      <Handle type="target" position={Position.Top} id="target-top" style={{ opacity: 0 }} />
      <Handle type="source" position={Position.Bottom} id="source-bottom" style={{ opacity: 0 }} />
      <Handle type="target" position={Position.Left} id="target-left" style={{ opacity: 0 }} />
      <Handle type="source" position={Position.Right} id="source-right" style={{ opacity: 0 }} />
      <Handle type="source" position={Position.Top} id="source-top" style={{ opacity: 0 }} />
      <Handle type="target" position={Position.Bottom} id="target-bottom" style={{ opacity: 0 }} />
      <Handle type="source" position={Position.Left} id="source-left" style={{ opacity: 0 }} />
      <Handle type="target" position={Position.Right} id="target-right" style={{ opacity: 0 }} />

      {/* Availability indicator dot */}
      {latestAvailability && (
        <div style={{
          position: 'absolute',
          top: '6px',
          right: '6px',
          width: '8px',
          height: '8px',
          borderRadius: '50%',
          backgroundColor:
            latestAvailability.score > 0.7 ? '#22c55e' :
            latestAvailability.score > 0.4 ? '#eab308' :
            '#ef4444',
        }} />
      )}

      {person.name}
    </div>
  );
}
