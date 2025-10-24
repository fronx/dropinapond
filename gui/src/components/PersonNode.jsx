import { Handle, Position } from '@xyflow/react';

export function PersonNode({ data, selected }) {
  const { person, isSelf } = data;
  const topPhrases = person.phrases
    ?.slice(0, 3)
    .map(p => p.text || p)
    .join(', ') || '';

  const capabilities = person.capabilities?.slice(0, 3) || [];
  const latestAvailability = person.availability?.[0];

  const nodeStyle = {
    position: 'relative',
    padding: '12px 16px',
    borderRadius: '8px',
    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
    minWidth: '220px',
    maxWidth: '280px',
    transition: 'all 0.3s',
    border: isSelf ? '2px solid #3b82f6' : '2px solid #d1d5db',
    backgroundColor: isSelf ? '#eff6ff' : 'white',
    ...(isSelf && { boxShadow: '0 0 0 2px #93c5fd' }),
    ...(selected && {
      boxShadow: '0 0 0 4px #c084fc',
      transform: 'scale(1.05)',
    }),
  };

  return (
    <div style={nodeStyle}>
      {/* Handles for edge connections */}
      <Handle type="target" position={Position.Top} />
      <Handle type="source" position={Position.Bottom} />
      <Handle type="target" position={Position.Left} />
      <Handle type="source" position={Position.Right} />

      {/* Name */}
      <div style={{
        fontWeight: 'bold',
        fontSize: '1.125rem',
        marginBottom: '8px',
        color: isSelf ? '#1e40af' : '#1f2937',
      }}>
        {person.name}
        {isSelf && (
          <span style={{
            fontSize: '0.75rem',
            marginLeft: '8px',
            color: '#2563eb',
          }}>
            (You)
          </span>
        )}
      </div>

      {/* Top phrases */}
      {topPhrases && (
        <div style={{
          fontSize: '0.75rem',
          color: '#4b5563',
          marginBottom: '8px',
          fontStyle: 'italic',
        }}>
          {topPhrases.length > 60 ? `${topPhrases.substring(0, 60)}...` : topPhrases}
        </div>
      )}

      {/* Capabilities */}
      {capabilities.length > 0 && (
        <div style={{
          fontSize: '0.75rem',
          color: '#15803d',
          marginBottom: '4px',
        }}>
          <span style={{ fontWeight: '600' }}>Skills:</span> {capabilities.join(', ')}
        </div>
      )}

      {/* Availability indicator */}
      {latestAvailability && (
        <div style={{
          fontSize: '0.75rem',
          marginTop: '8px',
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
        }}>
          <div style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            backgroundColor:
              latestAvailability.score > 0.7 ? '#22c55e' :
              latestAvailability.score > 0.4 ? '#eab308' :
              '#ef4444',
          }} />
          <span style={{ color: '#4b5563' }}>
            {latestAvailability.content?.substring(0, 40)}
          </span>
        </div>
      )}

      {/* Phrase count */}
      <div style={{
        fontSize: '0.75rem',
        color: '#9ca3af',
        marginTop: '8px',
      }}>
        {person.phrases?.length || 0} phrases
      </div>
    </div>
  );
}
