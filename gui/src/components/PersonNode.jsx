import { Handle, Position } from '@xyflow/react';
import { useState, useEffect } from 'react';
import chroma from 'chroma-js';

export function PersonNode({ data, selected }) {
  const { person, isSelf, connectionStrength = 0.5, clusterColor } = data;
  const latestAvailability = person.availability?.[0];

  // Detect dark mode
  const [isDarkMode, setIsDarkMode] = useState(
    window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
  );

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = (e) => setIsDarkMode(e.matches);
    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, []);

  // Scale node size based on connection strength (smaller overall)
  const scale = isSelf ? 0.7 : 0.35 + (connectionStrength * 0.35);
  const fontSize = 0.875 * scale;
  const padding = `${8 * scale}px ${16 * scale}px`;

  // Use solid cluster color for background, adjust brightness based on theme
  let backgroundColor = isDarkMode ? '#2a2a2a' : 'white';
  let textColor = isDarkMode ? '#e5e5e5' : '#000000';
  let borderColor = isDarkMode ? '#555' : '#d1d5db';

  if (clusterColor) {
    borderColor = clusterColor;

    if (isDarkMode) {
      // In dark mode, darken the color
      const bgColor = chroma(clusterColor).darken(0.7);
      backgroundColor = bgColor.hex();
      // Calculate text color based on background luminance
      const luminance = bgColor.luminance();
      textColor = luminance > 0.5 ? '#000000' : '#ffffff';
    } else {
      // In light mode, lighten the color significantly to ensure light background
      const bgColor = chroma(clusterColor).brighten(1.5);
      backgroundColor = bgColor.hex();
      // Always use black text in light mode for consistency
      textColor = '#000000';
    }
  } else if (isSelf) {
    if (isDarkMode) {
      backgroundColor = '#1e3a5f';
      borderColor = '#3b82f6';
      textColor = '#93c5fd';
    } else {
      backgroundColor = '#eff6ff';
      borderColor = '#3b82f6';
      textColor = '#1e40af';
    }
  }

  const nodeStyle = {
    position: 'relative',
    padding,
    borderRadius: '6px',
    boxShadow: selected
      ? '0 0 0 3px #c084fc, 0 4px 6px -1px rgba(0, 0, 0, 0.2)'
      : '0 2px 4px rgba(0, 0, 0, 0.1)',
    transition: 'all 0.2s',
    border: `2px solid ${borderColor}`,
    backgroundColor,
    cursor: 'pointer',
    fontSize: `${fontSize}rem`,
    fontWeight: '600',
    color: textColor,
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
          top: `${4 * scale}px`,
          right: `${4 * scale}px`,
          width: `${6 * scale}px`,
          height: `${6 * scale}px`,
          borderRadius: '50%',
          backgroundColor:
            latestAvailability.score > 0.7 ? '#22c55e' :
            latestAvailability.score > 0.4 ? '#eab308' :
            '#ef4444',
          boxShadow: '0 1px 2px rgba(0, 0, 0, 0.2)',
        }} />
      )}

      {person.name}
    </div>
  );
}
