import { useState, useEffect } from 'react';
import { getMetricLabel } from '../lib/metricLabels';

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
              <div style={{ marginBottom: '12px' }}>
                <div style={metricRowStyle}>
                  <span style={labelStyle}>Public Legibility</span>
                  <span style={valueStyle}>
                    {overallMetrics.publicLegibilityOverall
                      ? getMetricLabel('public_legibility', overallMetrics.publicLegibilityOverall).label
                      : 'N/A'}
                  </span>
                </div>
                {overallMetrics.publicLegibilityOverall && (
                  <div style={{
                    fontSize: '0.75rem',
                    color: isDarkMode ? '#6b7280' : '#9ca3af',
                    marginTop: '2px',
                    fontStyle: 'italic'
                  }}>
                    {getMetricLabel('public_legibility', overallMetrics.publicLegibilityOverall).description}
                  </div>
                )}
              </div>
              <div>
                <div style={metricRowStyle}>
                  <span style={labelStyle}>Attention Spread</span>
                  <span style={valueStyle}>
                    {overallMetrics.attentionEntropy
                      ? getMetricLabel('attention_entropy', overallMetrics.attentionEntropy).label
                      : 'N/A'}
                  </span>
                </div>
                {overallMetrics.attentionEntropy && (
                  <div style={{
                    fontSize: '0.75rem',
                    color: isDarkMode ? '#6b7280' : '#9ca3af',
                    marginTop: '2px',
                    fontStyle: 'italic'
                  }}>
                    {getMetricLabel('attention_entropy', overallMetrics.attentionEntropy).description}
                  </div>
                )}
              </div>
            </div>
          )}

          {recommendations && recommendations.length > 0 && (
            <div style={sectionStyle}>
              <div style={sectionTitleStyle}>Top Recommendations</div>
              {recommendations.slice(0, 5).map((rec, idx) => {
                const displayName = nodeNameMap?.[rec.node_id] || rec.node_id;
                const scoreLabel = getMetricLabel('orientation_score', rec.score).label;
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
                      {scoreLabel}
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
                      <div style={{ marginBottom: '8px' }}>
                        <div style={{ ...metricRowStyle, fontSize: '0.75rem' }}>
                          <span style={labelStyle}>Legibility</span>
                          <span style={valueStyle}>
                            {getMetricLabel('public_legibility', value).label}
                          </span>
                        </div>
                        <div style={{
                          fontSize: '0.7rem',
                          color: isDarkMode ? '#6b7280' : '#9ca3af',
                          marginTop: '2px',
                          fontStyle: 'italic'
                        }}>
                          {getMetricLabel('public_legibility', value).description}
                        </div>
                      </div>
                      {attunement !== undefined && (
                        <div style={{ marginBottom: '8px' }}>
                          <div style={{ ...metricRowStyle, fontSize: '0.75rem' }}>
                            <span style={labelStyle}>Attunement</span>
                            <span style={valueStyle}>
                              {getMetricLabel('subjective_attunement', attunement).label}
                            </span>
                          </div>
                          <div style={{
                            fontSize: '0.7rem',
                            color: isDarkMode ? '#6b7280' : '#9ca3af',
                            marginTop: '2px',
                            fontStyle: 'italic'
                          }}>
                            {getMetricLabel('subjective_attunement', attunement).description}
                          </div>
                        </div>
                      )}
                      {novelty !== undefined && (
                        <div>
                          <div style={{ ...metricRowStyle, fontSize: '0.75rem' }}>
                            <span style={labelStyle}>Novelty</span>
                            <span style={valueStyle}>
                              {getMetricLabel('heat_residual_novelty', novelty).label}
                            </span>
                          </div>
                          <div style={{
                            fontSize: '0.7rem',
                            color: isDarkMode ? '#6b7280' : '#9ca3af',
                            marginTop: '2px',
                            fontStyle: 'italic'
                          }}>
                            {getMetricLabel('heat_residual_novelty', novelty).description}
                          </div>
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
