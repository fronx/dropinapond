import { useState, useEffect } from 'react';

export function SimplifiedMetricsLegend() {
  const [isDarkMode, setIsDarkMode] = useState(
    window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
  );
  const [isExpanded, setIsExpanded] = useState(true);

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = (e) => setIsDarkMode(e.matches);
    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, []);

  const panelStyle = {
    position: 'absolute',
    top: '16px',
    left: '16px',
    backgroundColor: isDarkMode ? 'rgba(31, 41, 55, 0.95)' : 'rgba(255, 255, 255, 0.95)',
    border: `1px solid ${isDarkMode ? '#374151' : '#d1d5db'}`,
    borderRadius: '8px',
    padding: '16px',
    maxWidth: '340px',
    maxHeight: '80vh',
    overflowY: 'auto',
    boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    fontSize: '0.875rem',
    color: isDarkMode ? '#e5e5e5' : '#1f2937',
    zIndex: 10,
  };

  const headingStyle = {
    fontSize: '1rem',
    fontWeight: '700',
    marginBottom: '4px',
    color: isDarkMode ? '#93c5fd' : '#1e40af',
  };

  const subtitleStyle = {
    fontSize: '0.75rem',
    color: isDarkMode ? '#9ca3af' : '#6b7280',
    marginBottom: '12px',
  };

  const metricStyle = {
    marginBottom: '16px',
    paddingBottom: '16px',
    borderBottom: `1px solid ${isDarkMode ? '#374151' : '#e5e7eb'}`,
  };

  const metricNameStyle = {
    fontWeight: '600',
    color: isDarkMode ? '#d1d5db' : '#374151',
    marginBottom: '6px',
    fontSize: '0.875rem',
  };

  const descStyle = {
    color: isDarkMode ? '#9ca3af' : '#6b7280',
    lineHeight: '1.5',
    marginBottom: '8px',
    fontSize: '0.8125rem',
  };

  const rangeBoxStyle = {
    backgroundColor: isDarkMode ? 'rgba(59, 130, 246, 0.1)' : 'rgba(59, 130, 246, 0.05)',
    borderLeft: `3px solid ${isDarkMode ? '#3b82f6' : '#60a5fa'}`,
    padding: '8px',
    borderRadius: '4px',
    fontSize: '0.75rem',
    color: isDarkMode ? '#93c5fd' : '#1e40af',
  };

  const collapseButtonStyle = {
    background: 'none',
    border: 'none',
    cursor: 'pointer',
    fontSize: '1.25rem',
    color: isDarkMode ? '#9ca3af' : '#6b7280',
    padding: '0',
  };

  return (
    <div style={panelStyle}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <div>
          <h2 style={headingStyle}>What do these numbers mean?</h2>
          <div style={subtitleStyle}>Metrics guide</div>
        </div>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          style={collapseButtonStyle}
        >
          {isExpanded ? '−' : '+'}
        </button>
      </div>

      {isExpanded && (
        <>
          <div style={metricStyle}>
            <div style={metricNameStyle}>Orientation Score</div>
            <div style={descStyle}>
              Who should I talk to next? Higher scores = better strategic fit for your goals.
            </div>
            <div style={rangeBoxStyle}>
              <strong>1.4-1.7:</strong> Safe, familiar connections<br/>
              <strong>1.8-2.1:</strong> Good opportunities<br/>
              <strong>2.2+:</strong> High-value connections (explore these!)
            </div>
          </div>

          <div style={metricStyle}>
            <div style={metricNameStyle}>Readability</div>
            <div style={descStyle}>
              Can this one person predict all my interests? Low values are normal and healthy.
            </div>
            <div style={rangeBoxStyle}>
              <strong>0.0-0.2:</strong> You have unique interests they don't share (good!)<br/>
              <strong>0.2-0.4:</strong> Some overlap, but you're distinct<br/>
              <strong>0.4+:</strong> Strong alignment (rare with one person)
            </div>
          </div>

          <div style={metricStyle}>
            <div style={metricNameStyle}>Overlap</div>
            <div style={descStyle}>
              How much do our social circles overlap? Low = they connect you to new people.
            </div>
            <div style={rangeBoxStyle}>
              <strong>0.0-0.1:</strong> Opens new social territory (exploration)<br/>
              <strong>0.1-0.3:</strong> Some shared contacts<br/>
              <strong>0.3+:</strong> You're in the same bubble
            </div>
          </div>

          <div style={metricStyle}>
            <div style={metricNameStyle}>Public Legibility (cluster)</div>
            <div style={descStyle}>
              Does this group "get" me? Can they understand what I care about?
            </div>
            <div style={rangeBoxStyle}>
              <strong>0.0-0.3:</strong> You're mysterious to them<br/>
              <strong>0.3-0.6:</strong> Partial understanding<br/>
              <strong>0.6+:</strong> They really understand you
            </div>
          </div>

          <div style={metricStyle}>
            <div style={metricNameStyle}>Subjective Attunement (cluster)</div>
            <div style={descStyle}>
              Do I understand them? Low + high legibility = learning opportunity!
            </div>
            <div style={rangeBoxStyle}>
              <strong>0.0-0.2:</strong> They have perspectives you haven't grasped<br/>
              <strong>0.2-0.5:</strong> Getting there, more to learn<br/>
              <strong>0.5+:</strong> You understand them well
            </div>
          </div>

          <div style={metricStyle}>
            <div style={metricNameStyle}>Heat-Residual Novelty</div>
            <div style={descStyle}>
              How structurally distant are they in the network? Higher = more novel.
            </div>
            <div style={rangeBoxStyle}>
              <strong>0.04-0.05:</strong> Well-connected through others<br/>
              <strong>0.05-0.06:</strong> Moderately distant<br/>
              <strong>0.06+:</strong> Structurally far, hard to reach
            </div>
          </div>

          <div style={{
            backgroundColor: isDarkMode ? 'rgba(34, 197, 94, 0.1)' : 'rgba(34, 197, 94, 0.05)',
            border: `1px solid ${isDarkMode ? '#22c55e' : '#86efac'}`,
            borderRadius: '6px',
            padding: '12px',
            fontSize: '0.8125rem',
            color: isDarkMode ? '#86efac' : '#15803d',
            lineHeight: '1.5',
          }}>
            <strong>💡 Key insight:</strong> Low readability and low overlap are often GOOD signs.
            They mean exploration, diversity, and learning opportunities. The system recommends
            people who offer new perspectives while maintaining enough mutual understanding.
          </div>
        </>
      )}
    </div>
  );
}
