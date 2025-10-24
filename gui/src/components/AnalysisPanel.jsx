import { useState, useEffect } from 'react';

export function AnalysisPanel({ clusterMetrics, overallMetrics, recommendations, nodeNameMap }) {
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

  if (!overallMetrics && !clusterMetrics && !recommendations) {
    return null;
  }

  const panelStyle = {
    position: 'absolute',
    top: '16px',
    right: '16px',
    backgroundColor: isDarkMode ? 'rgba(31, 41, 55, 0.95)' : 'rgba(255, 255, 255, 0.95)',
    border: `1px solid ${isDarkMode ? '#374151' : '#d1d5db'}`,
    borderRadius: '8px',
    padding: '16px',
    maxWidth: '320px',
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
    marginBottom: '12px',
    color: isDarkMode ? '#93c5fd' : '#1e40af',
  };

  const sectionStyle = {
    marginBottom: '16px',
    paddingBottom: '16px',
    borderBottom: `1px solid ${isDarkMode ? '#374151' : '#e5e7eb'}`,
  };

  const sectionTitleStyle = {
    fontSize: '0.875rem',
    fontWeight: '600',
    marginBottom: '8px',
    color: isDarkMode ? '#d1d5db' : '#374151',
  };

  const metricRowStyle = {
    display: 'flex',
    justifyContent: 'space-between',
    marginBottom: '4px',
    fontSize: '0.8125rem',
  };

  const labelStyle = {
    color: isDarkMode ? '#9ca3af' : '#6b7280',
  };

  const valueStyle = {
    fontWeight: '600',
    color: isDarkMode ? '#e5e5e5' : '#1f2937',
  };

  return (
    <div style={panelStyle}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <h2 style={headingStyle}>Network Analysis</h2>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          style={{
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            fontSize: '1.25rem',
            color: isDarkMode ? '#9ca3af' : '#6b7280',
            padding: '0',
          }}
        >
          {isExpanded ? '−' : '+'}
        </button>
      </div>

      {isExpanded && (
        <>
          {overallMetrics && (
            <div style={sectionStyle}>
              <div style={sectionTitleStyle}>Overall Metrics</div>
              <div style={metricRowStyle}>
                <span style={labelStyle}>Public Legibility</span>
                <span style={valueStyle}>{overallMetrics.publicLegibilityOverall?.toFixed(3) || 'N/A'}</span>
              </div>
              <div style={metricRowStyle}>
                <span style={labelStyle}>Attention Entropy</span>
                <span style={valueStyle}>{overallMetrics.attentionEntropy?.toFixed(3) || 'N/A'}</span>
              </div>
            </div>
          )}

          {recommendations && recommendations.length > 0 && (
            <div style={sectionStyle}>
              <div style={sectionTitleStyle}>Top Recommendations</div>
              {recommendations.slice(0, 5).map((rec, idx) => {
                const displayName = nodeNameMap?.[rec.node_id] || rec.node_id;
                return (
                  <div key={rec.node_id} style={{
                    ...metricRowStyle,
                    padding: '4px 0',
                    fontWeight: idx === 0 ? '600' : '400',
                  }}>
                    <span style={labelStyle}>
                      {idx + 1}. {displayName}
                    </span>
                    <span style={{
                      ...valueStyle,
                      color: idx === 0 ? (isDarkMode ? '#93c5fd' : '#1e40af') : valueStyle.color,
                    }}>
                      {rec.score.toFixed(2)}
                    </span>
                  </div>
                );
              })}
            </div>
          )}

          {clusterMetrics && clusterMetrics.publicLegibilityPerCluster && (
            <div style={sectionStyle}>
              <div style={sectionTitleStyle}>Cluster Metrics</div>
              {Object.entries(clusterMetrics.publicLegibilityPerCluster).map(([clusterName, value]) => {
                const attunement = clusterMetrics.subjectiveAttunementPerCluster?.[clusterName];
                const novelty = clusterMetrics.heatResidualNovelty?.[clusterName];

                return (
                  <div key={clusterName} style={{ marginBottom: '12px' }}>
                    <div style={{
                      fontSize: '0.75rem',
                      fontWeight: '600',
                      marginBottom: '4px',
                      color: isDarkMode ? '#d1d5db' : '#4b5563',
                    }}>
                      {clusterName}
                    </div>
                    <div style={{ paddingLeft: '8px' }}>
                      <div style={{ ...metricRowStyle, fontSize: '0.75rem' }}>
                        <span style={labelStyle}>Legibility</span>
                        <span style={valueStyle}>{value?.toFixed(3)}</span>
                      </div>
                      {attunement !== undefined && (
                        <div style={{ ...metricRowStyle, fontSize: '0.75rem' }}>
                          <span style={labelStyle}>Attunement</span>
                          <span style={valueStyle}>{attunement.toFixed(3)}</span>
                        </div>
                      )}
                      {novelty !== undefined && (
                        <div style={{ ...metricRowStyle, fontSize: '0.75rem' }}>
                          <span style={labelStyle}>Novelty</span>
                          <span style={valueStyle}>{novelty.toFixed(3)}</span>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </>
      )}
    </div>
  );
}
