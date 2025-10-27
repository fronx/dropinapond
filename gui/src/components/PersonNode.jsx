import { Handle, Position } from '@xyflow/react';
import { useState, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import chroma from 'chroma-js';
import { getMetricLabel } from '../lib/metricLabels';

export function PersonNode({ data, selected }) {
  const { person, isSelf, connectionStrength = 0.5, clusterColor, analysisMetrics } = data;
  const latestAvailability = person.availability?.[0];
  const [showTooltip, setShowTooltip] = useState(false);
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });
  const nodeRef = useRef(null);

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
  const scale = isSelf ? 0.7 : 0.3 + (connectionStrength * 0.5);
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

  const handleMouseEnter = () => {
    if (nodeRef.current) {
      const rect = nodeRef.current.getBoundingClientRect();
      setTooltipPosition({
        x: rect.left + rect.width / 2,
        y: rect.bottom + 8
      });
    }
    setShowTooltip(true);
  };

  return (
    <div
      ref={nodeRef}
      style={nodeStyle}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={() => setShowTooltip(false)}
    >
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

      {/* Analysis metrics tooltip - rendered via portal */}
      {showTooltip && analysisMetrics && !isSelf && (() => {
        const orientationLabel = analysisMetrics.orientationScore !== undefined
          ? getMetricLabel('orientation_score', analysisMetrics.orientationScore)
          : null;
        const readabilityLabel = analysisMetrics.readability !== undefined
          ? getMetricLabel('readability', analysisMetrics.readability)
          : null;
        const overlapLabel = analysisMetrics.overlap !== undefined
          ? getMetricLabel('overlap', analysisMetrics.overlap)
          : null;

        const tooltipContent = (
          <div style={{
            position: 'fixed',
            left: `${tooltipPosition.x}px`,
            top: `${tooltipPosition.y}px`,
            transform: 'translateX(-50%)',
            backgroundColor: isDarkMode ? '#1f2937' : 'white',
            border: `1px solid ${isDarkMode ? '#374151' : '#d1d5db'}`,
            borderRadius: '8px',
            padding: '12px 16px',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
            fontSize: '0.875rem',
            whiteSpace: 'nowrap',
            zIndex: 9999,
            pointerEvents: 'none',
          }}>
            {orientationLabel && (
              <div style={{ marginBottom: '6px', color: isDarkMode ? '#93c5fd' : '#1e40af', fontWeight: '600', fontSize: '0.9rem' }}>
                {orientationLabel.label}
              </div>
            )}
            {readabilityLabel && (
              <div style={{ marginBottom: '4px', color: isDarkMode ? '#d1d5db' : '#374151' }}>
                {readabilityLabel.label}
              </div>
            )}
            {overlapLabel && (
              <div style={{ color: isDarkMode ? '#d1d5db' : '#374151' }}>
                {overlapLabel.label}
              </div>
            )}
            {orientationLabel && (
              <div style={{
                marginTop: '8px',
                paddingTop: '8px',
                borderTop: `1px solid ${isDarkMode ? '#374151' : '#e5e7eb'}`,
                color: isDarkMode ? '#9ca3af' : '#6b7280',
                fontSize: '0.8rem'
              }}>
                {orientationLabel.description}
              </div>
            )}
          </div>
        );

        return createPortal(tooltipContent, document.body);
      })()}
    </div>
  );
}
