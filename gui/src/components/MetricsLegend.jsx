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
              <strong>Main navigation signal.</strong> Composite metric recommending next interactions.
              Higher = better strategic value for your goals.
            </div>
            <div style={metricDescStyle}>
              Combines: exploration value (1-overlap) + mutual understanding (readability) +
              learning opportunity (low attunement where you're legible) + semantic alignment.
            </div>
            <div style={rangeStyle}>
              Typical range: 1.4-2.7. Top recommendations: {'>'}2.0
            </div>
          </div>

          <div style={metricStyle}>
            <div style={metricNameStyle}>Readability (R²_in)</div>
            <div style={metricDescStyle}>
              <strong>How well can this ONE person reconstruct your semantic field?</strong>
              Low values are normal and healthy - they mean you have interests beyond what this person alone represents.
            </div>
            <div style={metricDescStyle}>
              Implementation: Tries to predict your entire field as a scalar multiple of theirs (beta * their_field ≈ your_field).
            </div>
            <div style={rangeStyle}>
              Range: 0.0-1.0. Typical: 0.05-0.50. High ({'>'} 0.5) means strong overlap, low ({'<'} 0.2) means you have orthogonal interests.
            </div>
          </div>

          <div style={metricStyle}>
            <div style={metricNameStyle}>Overlap (Jaccard)</div>
            <div style={metricDescStyle}>
              <strong>Network structure overlap, NOT semantic similarity.</strong>
              Measures: |shared_neighbors| / |all_neighbors_combined|
            </div>
            <div style={metricDescStyle}>
              Low overlap (0.0-0.2) means this person connects you to NEW parts of the network (exploration value).
              High overlap ({'>'} 0.5) means you're in the same social cluster (reinforcement).
            </div>
            <div style={rangeStyle}>
              Range: 0.0-1.0. Low ({'<'} 0.2) is valuable for exploration.
            </div>
          </div>

          <div style={metricStyle}>
            <div style={metricNameStyle}>Public Legibility (per cluster)</div>
            <div style={metricDescStyle}>
              <strong>How well can this entire cluster reconstruct your field?</strong>
              Uses all cluster members together (not just one person).
            </div>
            <div style={metricDescStyle}>
              High ({'>'} 0.6): This cluster "gets you" - you're predictable/understandable to them.
              Low ({'<'} 0.3): You're mysterious to this cluster, potential for surprise.
            </div>
            <div style={rangeStyle}>
              Range: 0.0-1.0. Higher = more mutual understanding.
            </div>
          </div>

          <div style={metricStyle}>
            <div style={metricNameStyle}>Subjective Attunement (R²_out)</div>
            <div style={metricDescStyle}>
              <strong>How well can YOU reconstruct this cluster's fields from yours?</strong>
              Low values mean learning opportunities (if legibility is sufficient).
            </div>
            <div style={metricDescStyle}>
              Low attunement + high legibility = ideal for learning (they understand you, but you don't fully grasp them yet).
              Low attunement + low legibility = misalignment (hard to bridge).
            </div>
            <div style={rangeStyle}>
              Range: 0.0-1.0. Low ({'<'} 0.3) with legibility {'>'} 0.3 = learning opportunity.
            </div>
          </div>

          <div style={metricStyle}>
            <div style={metricNameStyle}>Heat-Residual Novelty</div>
            <div style={metricDescStyle}>
              <strong>Topological distance in the interaction graph.</strong>
              Uses diffusion geometry - how far does "heat" need to spread to reach this cluster?
            </div>
            <div style={metricDescStyle}>
              High novelty = structurally distant, not well-connected through intermediate paths.
              Low novelty = reachable through multiple social paths.
            </div>
            <div style={rangeStyle}>
              Range: ~0.04-0.07 (small values). Higher = more topologically novel.
            </div>
          </div>

          <div style={metricStyle}>
            <div style={metricNameStyle}>Attention Entropy</div>
            <div style={metricDescStyle}>
              <strong>How evenly distributed is your attention across clusters?</strong>
              Shannon entropy of interaction weights.
            </div>
            <div style={metricDescStyle}>
              High entropy = exploration mode (attention spread across the landscape).
              Low entropy = exploitation mode (focused on one region).
            </div>
            <div style={rangeStyle}>
              Range: 0.0-log₂(n_clusters). Higher = more diverse attention.
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
            <strong>Key insight:</strong> Low readability and overlap are often GOOD - they indicate exploration value
            and diverse perspectives. The orientation score combines all metrics to find strategically valuable connections.
          </div>
        </div>
      )}
    </div>
  );
}
