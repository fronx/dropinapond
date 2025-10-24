import { useState, useEffect } from 'react';

export function MetricsLegend({ isDarkMode }) {
  const [isExpanded, setIsExpanded] = useState(true);

  const containerStyle = {
    marginTop: '12px',
    paddingTop: '12px',
    borderTop: `1px solid ${isDarkMode ? '#374151' : '#e5e7eb'}`,
  };

  const headerStyle = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    cursor: 'pointer',
    padding: '8px 0',
  };

  const titleStyle = {
    fontSize: '0.875rem',
    fontWeight: '600',
    color: isDarkMode ? '#93c5fd' : '#1e40af',
  };

  const metricStyle = {
    marginBottom: '12px',
    fontSize: '0.8125rem',
  };

  const metricNameStyle = {
    fontWeight: '600',
    color: isDarkMode ? '#d1d5db' : '#374151',
    marginBottom: '4px',
  };

  const metricDescStyle = {
    color: isDarkMode ? '#9ca3af' : '#6b7280',
    lineHeight: '1.4',
    marginBottom: '4px',
  };

  const rangeStyle = {
    fontSize: '0.75rem',
    color: isDarkMode ? '#6b7280' : '#9ca3af',
    fontStyle: 'italic',
  };

  return (
    <div style={containerStyle}>
      <div style={headerStyle} onClick={() => setIsExpanded(!isExpanded)}>
        <span style={titleStyle}>Metrics Guide</span>
        <span style={{
          color: isDarkMode ? '#9ca3af' : '#6b7280',
          fontSize: '1rem',
        }}>
          {isExpanded ? '▼' : '▶'}
        </span>
      </div>

      {isExpanded && (
        <div style={{ paddingTop: '8px' }}>
          <div style={metricStyle}>
            <div style={metricNameStyle}>Orientation Score</div>
            <div style={metricDescStyle}>
              <strong>Who should you talk to next?</strong> This is your main navigation compass.
              Higher scores mean more strategic value for expanding your network and learning new things.
            </div>
            <div style={rangeStyle}>
              Strong recommendations: Above 2.0 | Typical range: 1.4-2.7
            </div>
          </div>

          <div style={metricStyle}>
            <div style={metricNameStyle}>Readability</div>
            <div style={metricDescStyle}>
              <strong>How well does this one person "get" your interests?</strong>
              Low values (0.05-0.20) are healthy - they mean you're multifaceted and have interests this person doesn't share.
              High values (0.50+) mean strong alignment with what this specific person cares about.
            </div>
            <div style={rangeStyle}>
              Low ({'<'} 0.20): Different interests | Medium (0.20-0.50): Some overlap | High (0.50+): Strong alignment
            </div>
          </div>

          <div style={metricStyle}>
            <div style={metricNameStyle}>Overlap</div>
            <div style={metricDescStyle}>
              <strong>Do you know the same people?</strong>
              Low overlap means this person opens doors to new circles you don't know yet.
              High overlap means you're already in the same social cluster.
            </div>
            <div style={rangeStyle}>
              Low ({'<'} 0.20): New circles | Medium (0.20-0.50): Some shared connections | High (0.50+): Same bubble
            </div>
          </div>

          <div style={metricStyle}>
            <div style={metricNameStyle}>Public Legibility (Cluster)</div>
            <div style={metricDescStyle}>
              <strong>Does this whole group understand you?</strong>
              When an entire cluster scores high (0.60+), they really "get" what you're about.
              Low scores (below 0.30) mean you might surprise them or seem mysterious.
            </div>
            <div style={rangeStyle}>
              Low ({'<'} 0.30): You're mysterious here | Medium (0.30-0.60): Partial understanding | High (0.60+): They get you
            </div>
          </div>

          <div style={metricStyle}>
            <div style={metricNameStyle}>Subjective Attunement (Cluster)</div>
            <div style={metricDescStyle}>
              <strong>How well do you understand this cluster?</strong>
              Low scores where they understand you = prime learning opportunity.
              Low scores when they don't understand you either = harder to connect.
            </div>
            <div style={rangeStyle}>
              Low ({'<'} 0.30): Room to learn | Medium (0.30-0.60): Decent grasp | High (0.60+): You understand them well
            </div>
          </div>

          <div style={metricStyle}>
            <div style={metricNameStyle}>Heat-Residual Novelty</div>
            <div style={metricDescStyle}>
              <strong>How far away is this cluster in your network?</strong>
              Higher values mean fewer social paths connect you - they're structurally distant.
              Lower values mean you're well-connected through mutual friends.
            </div>
            <div style={rangeStyle}>
              Typical range: 0.04-0.07 | Higher = More structurally distant
            </div>
          </div>

          <div style={metricStyle}>
            <div style={metricNameStyle}>Attention Entropy</div>
            <div style={metricDescStyle}>
              <strong>How spread out is your attention?</strong>
              High values mean you're exploring broadly across different groups.
              Low values mean you're focused intensely on one cluster.
            </div>
            <div style={rangeStyle}>
              Low: Focused mode | High: Exploration mode
            </div>
          </div>

          <div style={{
            marginTop: '12px',
            padding: '8px',
            backgroundColor: isDarkMode ? 'rgba(59, 130, 246, 0.1)' : 'rgba(59, 130, 246, 0.05)',
            borderRadius: '4px',
            fontSize: '0.75rem',
            color: isDarkMode ? '#93c5fd' : '#1e40af',
          }}>
            <strong>Remember:</strong> Low readability and overlap are often good signs - they mean potential for exploration,
            fresh perspectives, and access to new parts of the network. The orientation score balances all these factors
            to help you choose your next move strategically.
          </div>
        </div>
      )}
    </div>
  );
}
