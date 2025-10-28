import { useState, useEffect } from 'react';
import {
  getMetricLabel,
  getSemanticFlowLabel,
  extractSemanticFlowMetrics,
  computeAllPercentiles,
  getStructuralDescription,
  getSemanticDistanceDescription,
  getPercentileLabel
} from '../lib/metricLabels';

/**
 * Sidebar that appears when clicking a person node.
 * Shows the metric values and explains what data went into computing them.
 */
export function PersonDetailSidebar({ person, egoGraphData, analysisData, onClose }) {
  const [isDarkMode, setIsDarkMode] = useState(
    window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
  );

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = (e) => setIsDarkMode(e.matches);
    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, []);

  if (!person || !egoGraphData) return null;

  const isFocalNode = person.id === egoGraphData.self.id;

  // Get the full person data from egoGraphData (includes phrases)
  const fullPersonData = isFocalNode
    ? egoGraphData.self
    : egoGraphData.connections.find(c => c.id === person.id) || person;

  // Detect which analysis type we have (semantic_flow vs ego_ops)
  const isSemanticFlow = analysisData?.metrics?.layers !== undefined;

  // Extract metrics for this person from analysis data
  const metrics = isSemanticFlow
    ? extractSemanticFlowMetrics(analysisData, egoGraphData.self.id, person.id)
    : {
        // Legacy ego_ops metrics
        readability: analysisData?.metrics?.per_neighbor_readability?.[person.id],
        overlap: analysisData?.metrics?.overlaps?.[person.id],
        orientationScore: analysisData?.metrics?.orientation_scores?.[person.id],
        orientationBreakdown: analysisData?.metrics?.orientation_score_breakdowns?.[person.id],
      };

  // Find which cluster this person belongs to
  let personCluster = null;
  let clusterIndex = null;
  if (analysisData?.metrics?.clusters) {
    analysisData.metrics.clusters.forEach((cluster, idx) => {
      if (cluster.includes(person.id)) {
        personCluster = cluster;
        clusterIndex = idx;
      }
    });
  }

  // Compute percentiles for relative metrics from the actual network data
  const percentiles = isSemanticFlow
    ? computeAllPercentiles(analysisData, egoGraphData.self.id)
    : null;

  // Get cluster metrics if available
  let clusterMetrics = null;
  if (clusterIndex !== null && analysisData?.metrics) {
    if (isSemanticFlow) {
      // Semantic flow cluster metrics
      clusterMetrics = analysisData.metrics.coherence?.regions?.[clusterIndex];
    } else {
      // Legacy ego_ops cluster metrics
      const clusterNames = Object.keys(analysisData.metrics.public_legibility_per_cluster || {});
      const clusterName = clusterNames[clusterIndex];

      if (clusterName) {
        clusterMetrics = {
          publicLegibility: analysisData.metrics.public_legibility_per_cluster[clusterName],
          subjectiveAttunement: analysisData.metrics.subjective_attunement_per_cluster?.[clusterName],
          heatResidualNovelty: analysisData.metrics.heat_residual_novelty?.[clusterName],
        };
      }
    }
  }

  // Extract relevant data snippets from ego graph
  const selfPhrases = egoGraphData.self.phrases || [];
  const personPhrases = fullPersonData.phrases || [];

  // Get pre-computed phrase similarities from backend analysis (embedding-based, not word overlap!)
  const similarPhrases = !isFocalNode && analysisData?.metrics?.phrase_similarities?.[person.id]
    ? analysisData.metrics.phrase_similarities[person.id]
    : [];

  // Find unique phrases (those not in similar pairs)
  const getUniquePhrases = (phrases, similarPairs, isFromSelf) => {
    const matchedPhraseTexts = new Set(
      similarPairs.map(sp => isFromSelf ? sp.focal_phrase : sp.neighbor_phrase)
    );
    return phrases.filter(p => !matchedPhraseTexts.has(p.text));
  };

  const uniqueSelfPhrases = !isFocalNode ? getUniquePhrases(selfPhrases, similarPhrases, true) : [];
  const uniquePersonPhrases = !isFocalNode ? getUniquePhrases(personPhrases, similarPhrases, false) : [];

  // Find edges involving this person
  const edgesWithPerson = egoGraphData.edges.filter(
    edge => edge.source === person.id || edge.target === person.id
  );
  const edgeToFocal = egoGraphData.edges.find(
    edge =>
      (edge.source === egoGraphData.self.id && edge.target === person.id) ||
      (edge.target === egoGraphData.self.id && edge.source === person.id)
  );

  // Find neighbors of focal node
  const focalNeighbors = new Set();
  egoGraphData.edges.forEach(edge => {
    if (edge.source === egoGraphData.self.id) focalNeighbors.add(edge.target);
    if (edge.target === egoGraphData.self.id) focalNeighbors.add(edge.source);
  });

  // Find neighbors of this person
  const personNeighbors = new Set();
  edgesWithPerson.forEach(edge => {
    const neighbor = edge.source === person.id ? edge.target : edge.source;
    personNeighbors.add(neighbor);
  });

  // Calculate shared neighbors for overlap explanation
  const sharedNeighbors = [...focalNeighbors].filter(n => personNeighbors.has(n));
  const allNeighbors = new Set([...focalNeighbors, ...personNeighbors]);

  const sidebarStyle = {
    position: 'fixed',
    top: 0,
    left: 0,
    width: '480px',
    height: '100vh',
    backgroundColor: isDarkMode ? 'rgba(17, 24, 39, 0.98)' : 'rgba(255, 255, 255, 0.98)',
    borderRight: `2px solid ${isDarkMode ? '#374151' : '#d1d5db'}`,
    boxShadow: '4px 0 12px rgba(0, 0, 0, 0.15)',
    overflowY: 'auto',
    zIndex: 1000,
    padding: '24px',
    color: isDarkMode ? '#e5e5e5' : '#1f2937',
  };

  const headerStyle = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: '24px',
    paddingBottom: '16px',
    borderBottom: `2px solid ${isDarkMode ? '#374151' : '#e5e7eb'}`,
  };

  const sectionStyle = {
    marginBottom: '24px',
    paddingBottom: '16px',
    borderBottom: `1px solid ${isDarkMode ? '#374151' : '#e5e7eb'}`,
  };

  const sectionTitleStyle = {
    fontSize: '1rem',
    fontWeight: '700',
    marginBottom: '12px',
    color: isDarkMode ? '#93c5fd' : '#1e40af',
  };

  const subSectionTitleStyle = {
    fontSize: '0.875rem',
    fontWeight: '600',
    marginBottom: '8px',
    marginTop: '12px',
    color: isDarkMode ? '#d1d5db' : '#4b5563',
  };

  const codeBlockStyle = {
    backgroundColor: isDarkMode ? '#1f2937' : '#f3f4f6',
    border: `1px solid ${isDarkMode ? '#374151' : '#d1d5db'}`,
    borderRadius: '6px',
    padding: '12px',
    fontSize: '0.8125rem',
    fontFamily: 'monospace',
    marginTop: '8px',
    maxHeight: '200px',
    overflowY: 'auto',
  };

  const phraseListStyle = {
    ...codeBlockStyle,
    fontFamily: 'inherit',
  };

  const labelStyle = {
    fontSize: '0.75rem',
    fontWeight: '600',
    color: isDarkMode ? '#9ca3af' : '#6b7280',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    marginBottom: '4px',
  };

  const explanationStyle = {
    fontSize: '0.8125rem',
    color: isDarkMode ? '#9ca3af' : '#6b7280',
    fontStyle: 'italic',
    marginTop: '6px',
    lineHeight: '1.4',
  };

  return (
    <div style={sidebarStyle}>
      {/* Header */}
      <div style={headerStyle}>
        <div>
          <h2 style={{
            fontSize: '1.5rem',
            fontWeight: '700',
            color: isDarkMode ? '#e5e5e5' : '#1f2937',
            marginBottom: '4px',
          }}>
            {fullPersonData.name}
          </h2>
          {isFocalNode && (
            <div style={{
              fontSize: '0.875rem',
              color: isDarkMode ? '#93c5fd' : '#1e40af',
              fontWeight: '600',
            }}>
              (You)
            </div>
          )}
        </div>
        <button
          onClick={onClose}
          style={{
            background: 'none',
            border: 'none',
            fontSize: '1.5rem',
            cursor: 'pointer',
            color: isDarkMode ? '#9ca3af' : '#6b7280',
            padding: '0',
            lineHeight: '1',
          }}
        >
          ×
        </button>
      </div>

      {/* For focal node, show different content */}
      {isFocalNode ? (
        <>
          <div style={sectionStyle}>
            <div style={sectionTitleStyle}>Your Semantic Field</div>
            <div style={labelStyle}>Phrases (top 10)</div>
            <div style={phraseListStyle}>
              {selfPhrases.slice(0, 10).map((phrase, idx) => (
                <div key={idx} style={{ marginBottom: '4px' }}>
                  • {phrase.text} {phrase.weight && `(weight: ${phrase.weight.toFixed(2)})`}
                </div>
              ))}
              {selfPhrases.length > 10 && (
                <div style={{ marginTop: '8px', color: isDarkMode ? '#6b7280' : '#9ca3af' }}>
                  ...and {selfPhrases.length - 10} more
                </div>
              )}
            </div>
          </div>

          <div style={sectionStyle}>
            <div style={sectionTitleStyle}>Network Overview</div>
            <div style={{ fontSize: '0.875rem' }}>
              <div style={{ marginBottom: '8px' }}>
                <strong>Connections:</strong> {focalNeighbors.size} people
              </div>
              <div style={{ marginBottom: '8px' }}>
                <strong>Clusters:</strong> {analysisData?.metrics?.clusters?.length || 0} groups
              </div>
            </div>
          </div>
        </>
      ) : isSemanticFlow ? (
        <>
          {/* Semantic Flow Metrics - Humanized */}

          {/* Relationship Summary */}
          {(metrics.structuralEdge !== undefined || metrics.semanticAffinity !== undefined) && (
            <div style={sectionStyle}>
              <div style={sectionTitleStyle}>Your Connection</div>
              <div style={{ fontSize: '0.9375rem', lineHeight: '1.6', marginBottom: '0' }}>
                {metrics.structuralEdge !== undefined && metrics.semanticAffinity !== undefined ? (
                  <>
                    <span style={{ textTransform: 'capitalize' }}>
                      {getSemanticFlowLabel('structuralEdge', metrics.structuralEdge).label}
                    </span>
                    {' contact with '}
                    <span>
                      {getPercentileLabel('semanticAffinity', metrics.semanticAffinity, percentiles?.semanticAffinity)?.label ||
                       `${(metrics.semanticAffinity * 100).toFixed(0)}% affinity`}
                    </span>
                    {' interests'}
                    {metrics.effectiveEdge !== undefined && (
                      <>
                        {' — '}
                        <strong>
                          {getPercentileLabel('effectiveEdge', metrics.effectiveEdge, percentiles?.effectiveEdge)?.label ||
                           'connection'}
                        </strong>
                        {' overall'}
                      </>
                    )}
                    .
                  </>
                ) : (
                  metrics.structuralEdge !== undefined ? (
                    <span style={{ textTransform: 'capitalize' }}>
                      {getSemanticFlowLabel('structuralEdge', metrics.structuralEdge).label} contact.
                    </span>
                  ) : (
                    <span style={{ textTransform: 'capitalize' }}>
                      {getPercentileLabel('semanticAffinity', metrics.semanticAffinity, percentiles?.semanticAffinity)?.label ||
                       `${(metrics.semanticAffinity * 100).toFixed(0)}% affinity`} interests.
                    </span>
                  )
                )}
              </div>
            </div>
          )}

          {/* Mutual Understanding */}
          {(metrics.predictabilityRaw !== undefined || metrics.distanceRaw !== undefined) && (
            <div style={sectionStyle}>
              <div style={sectionTitleStyle}>Mutual Understanding</div>
              <div style={{ fontSize: '0.875rem', lineHeight: '1.6' }}>
                {metrics.predictabilityRaw !== undefined && (
                  <div style={{ marginBottom: metrics.distanceRaw !== undefined ? '8px' : '0' }}>
                    <strong style={{ textTransform: 'capitalize' }}>
                      {getSemanticFlowLabel('predictabilityRaw', metrics.predictabilityRaw).label}
                    </strong>
                    {' — '}
                    {getSemanticFlowLabel('predictabilityRaw', metrics.predictabilityRaw).interpretation}
                  </div>
                )}
                {metrics.distanceRaw !== undefined && (
                  <div>
                    <strong style={{ textTransform: 'capitalize' }}>
                      {getSemanticFlowLabel('distanceRaw', metrics.distanceRaw).label}
                    </strong>
                    {' — '}
                    {getSemanticFlowLabel('distanceRaw', metrics.distanceRaw).interpretation}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* In Your Network - neutral descriptions */}
          {metrics.structuralEdge !== undefined && (
            <div style={sectionStyle}>
              <div style={sectionTitleStyle}>In Your Network</div>
              <div style={{ fontSize: '0.875rem', lineHeight: '1.6' }}>
                {/* Structural dimension */}
                {getStructuralDescription(metrics.structuralEdge) && (
                  <div style={{ marginBottom: '8px' }}>
                    <strong>Structurally:</strong> {getStructuralDescription(metrics.structuralEdge)}
                  </div>
                )}

                {/* Semantic dimension - use percentiles */}
                {getSemanticDistanceDescription(metrics.distanceRaw, percentiles?.distanceRaw) && (
                  <div>
                    <strong>Semantically:</strong> {getSemanticDistanceDescription(metrics.distanceRaw, percentiles?.distanceRaw)}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Node Coherence - Simplified */}
          {metrics.coherenceNode && (
            <div style={sectionStyle}>
              <div style={sectionTitleStyle}>Role in Network</div>
              <div style={{ fontSize: '0.875rem', lineHeight: '1.6' }}>
                <div>
                  {fullPersonData.name} is a <strong>{getSemanticFlowLabel('fitRatio', metrics.coherenceNode.fit_ratio).label}</strong> in Cluster {metrics.coherenceNode.region_index}.
                  {' '}
                  {getSemanticFlowLabel('fitRatio', metrics.coherenceNode.fit_ratio).interpretation}
                </div>
              </div>
            </div>
          )}
        </>
      ) : (
        <>
          {/* Legacy Ego Ops Metrics */}

          {/* Metric: Readability */}
          {metrics.readability !== undefined && (
            <div style={sectionStyle}>
              <div style={sectionTitleStyle}>
                {getMetricLabel('readability', metrics.readability).label}
              </div>
              <div style={explanationStyle}>
                {getMetricLabel('readability', metrics.readability).description}
              </div>

              <div style={subSectionTitleStyle}>What went into this:</div>
              <div style={{ fontSize: '0.8125rem', marginBottom: '8px' }}>
                This measures how well <strong>{fullPersonData.name}</strong> could predict your interests
                based on their own semantic field. The computation uses the <strong>mean embeddings</strong> (weighted
                average of all phrase vectors) for you and {fullPersonData.name}, then applies ridge regression.
              </div>
              <div style={{ fontSize: '0.75rem', color: isDarkMode ? '#6b7280' : '#9ca3af', marginBottom: '12px', fontStyle: 'italic' }}>
                The phrase overlaps below help explain why the means are similar, but the actual R² is computed
                on the aggregated embeddings, not individual phrases.
              </div>

              {similarPhrases.length > 0 && (
                <>
                  <div style={labelStyle}>Overlapping interests ({similarPhrases.length})</div>
                  <div style={phraseListStyle}>
                    {similarPhrases.slice(0, 8).map((match, idx) => (
                      <div key={idx} style={{ marginBottom: '8px', paddingBottom: '8px', borderBottom: idx < Math.min(7, similarPhrases.length - 1) ? `1px solid ${isDarkMode ? '#374151' : '#e5e7eb'}` : 'none' }}>
                        <div style={{ color: isDarkMode ? '#93c5fd' : '#1e40af', fontWeight: '600', fontSize: '0.75rem' }}>
                          You: {match.focal_phrase}
                        </div>
                        <div style={{ color: isDarkMode ? '#d1d5db' : '#4b5563', fontSize: '0.75rem' }}>
                          Them: {match.neighbor_phrase}
                        </div>
                        <div style={{ color: isDarkMode ? '#6b7280' : '#9ca3af', fontSize: '0.7rem', marginTop: '2px' }}>
                          {Math.round(match.similarity * 100)}% semantic similarity
                        </div>
                      </div>
                    ))}
                    {similarPhrases.length > 8 && (
                      <div style={{ marginTop: '8px', color: isDarkMode ? '#6b7280' : '#9ca3af', fontSize: '0.75rem' }}>
                        ...and {similarPhrases.length - 8} more overlaps
                      </div>
                    )}
                  </div>
                </>
              )}

              {uniqueSelfPhrases.length > 0 && (
                <>
                  <div style={labelStyle}>Your unique interests (top 5)</div>
                  <div style={phraseListStyle}>
                    {uniqueSelfPhrases.slice(0, 5).map((phrase, idx) => (
                      <div key={idx} style={{ fontSize: '0.8125rem' }}>
                        • {phrase.text}
                        {phrase.weight && (
                          <span style={{ color: isDarkMode ? '#6b7280' : '#9ca3af', marginLeft: '4px' }}>
                            ({phrase.weight.toFixed(2)})
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                </>
              )}

              {uniquePersonPhrases.length > 0 && (
                <>
                  <div style={labelStyle}>{fullPersonData.name}'s unique interests (top 5)</div>
                  <div style={phraseListStyle}>
                    {uniquePersonPhrases.slice(0, 5).map((phrase, idx) => (
                      <div key={idx} style={{ fontSize: '0.8125rem' }}>
                        • {phrase.text}
                        {phrase.weight && (
                          <span style={{ color: isDarkMode ? '#6b7280' : '#9ca3af', marginLeft: '4px' }}>
                            ({phrase.weight.toFixed(2)})
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                </>
              )}

              {similarPhrases.length === 0 && personPhrases.length > 0 && (
                <>
                  <div style={labelStyle}>No obvious phrase overlaps detected</div>
                  <div style={{ fontSize: '0.8125rem', color: isDarkMode ? '#9ca3af' : '#6b7280', marginBottom: '12px' }}>
                    Despite low text overlap, {fullPersonData.name} may still understand you through semantic similarity in embeddings.
                  </div>

                  <div style={labelStyle}>Your top phrases (5)</div>
                  <div style={phraseListStyle}>
                    {selfPhrases.slice(0, 5).map((phrase, idx) => (
                      <div key={idx} style={{ fontSize: '0.8125rem' }}>• {phrase.text}</div>
                    ))}
                  </div>

                  <div style={labelStyle}>{fullPersonData.name}'s top phrases (5)</div>
                  <div style={phraseListStyle}>
                    {personPhrases.slice(0, 5).map((phrase, idx) => (
                      <div key={idx} style={{ fontSize: '0.8125rem' }}>• {phrase.text}</div>
                    ))}
                  </div>
                </>
              )}

              <div style={explanationStyle}>
                Formula: R² from ridge regression z_j → z_F (see [ego_ops.py:318-334](src/ego_ops.py#L318-L334))
              </div>
            </div>
          )}

          {/* Metric: Overlap */}
          {metrics.overlap !== undefined && (
            <div style={sectionStyle}>
              <div style={sectionTitleStyle}>
                {getMetricLabel('overlap', metrics.overlap).label}
              </div>
              <div style={explanationStyle}>
                {getMetricLabel('overlap', metrics.overlap).description}
              </div>

              <div style={subSectionTitleStyle}>What went into this:</div>
              <div style={{ fontSize: '0.8125rem', marginBottom: '8px' }}>
                Measures how much of your networks overlap using Jaccard index:
                (shared neighbors) / (all combined neighbors)
              </div>

              <div style={labelStyle}>Shared neighbors ({sharedNeighbors.length})</div>
              <div style={phraseListStyle}>
                {sharedNeighbors.length > 0 ? (
                  sharedNeighbors.map((neighborId) => {
                    const neighbor = egoGraphData.connections.find(c => c.id === neighborId);
                    return <div key={neighborId}>• {neighbor?.name || neighborId}</div>;
                  })
                ) : (
                  <div style={{ color: isDarkMode ? '#6b7280' : '#9ca3af' }}>
                    No mutual connections
                  </div>
                )}
              </div>

              <div style={labelStyle}>All combined neighbors ({allNeighbors.size})</div>
              <div style={explanationStyle}>
                Calculation: {sharedNeighbors.length} / {allNeighbors.size} = {metrics.overlap.toFixed(3)}
              </div>
              <div style={explanationStyle}>
                Formula: Jaccard index (see [ego_ops.py:247-255](src/ego_ops.py#L247-L255))
              </div>
            </div>
          )}

          {/* Metric: Orientation Score */}
          {metrics.orientationScore !== undefined && (
            <div style={sectionStyle}>
              <div style={sectionTitleStyle}>
                {getMetricLabel('orientation_score', metrics.orientationScore).label}
              </div>
              <div style={explanationStyle}>
                {getMetricLabel('orientation_score', metrics.orientationScore).description}
              </div>

              {metrics.orientationBreakdown ? (
                <>
                  <div style={subSectionTitleStyle}>Component Breakdown:</div>
                  <div style={{ fontSize: '0.8125rem', marginBottom: '12px' }}>
                    Total Score: <strong>{metrics.orientationBreakdown.total_score.toFixed(3)}</strong>
                  </div>

                  {/* Component details with actual values from backend */}
                  <div style={codeBlockStyle}>
                    {Object.entries(metrics.orientationBreakdown.components).map(([componentName, component]) => {
                      const isNegative = component.weighted_contribution < 0;
                      const weight = metrics.orientationBreakdown.weights[`lambda${
                        componentName === 'exploration' ? '1' :
                        componentName === 'readability' ? '2' :
                        componentName === 'attunement' ? '3' :
                        componentName === 'relevance' ? '4' : '5'
                      }_${componentName}`];

                      return (
                        <div key={componentName} style={{
                          marginBottom: '12px',
                          paddingBottom: '12px',
                          borderBottom: `1px solid ${isDarkMode ? '#374151' : '#e5e7eb'}`
                        }}>
                          <div style={{
                            fontWeight: '600',
                            color: isDarkMode ? '#93c5fd' : '#1e40af',
                            marginBottom: '4px',
                            textTransform: 'capitalize'
                          }}>
                            {componentName}
                          </div>
                          <div style={{ fontSize: '0.75rem', lineHeight: '1.4' }}>
                            <div style={{ marginBottom: '2px' }}>
                              Raw value: <strong>{component.raw_value.toFixed(3)}</strong>
                            </div>
                            <div style={{ marginBottom: '2px' }}>
                              Weight (λ): <strong>{weight.toFixed(1)}</strong>
                            </div>
                            <div style={{ marginBottom: '4px' }}>
                              Contribution: <strong style={{
                                color: isNegative
                                  ? (isDarkMode ? '#fca5a5' : '#dc2626')
                                  : (isDarkMode ? '#86efac' : '#16a34a')
                              }}>
                                {isNegative ? '' : '+'}{component.weighted_contribution.toFixed(3)}
                              </strong>
                            </div>
                            {component.metadata && (
                              <div style={{
                                fontSize: '0.7rem',
                                color: isDarkMode ? '#9ca3af' : '#6b7280',
                                fontStyle: 'italic'
                              }}>
                                {component.metadata.description}
                                {component.metadata.target_cluster_name && (
                                  <div style={{ marginTop: '2px' }}>
                                    Target cluster: {component.metadata.target_cluster_name}
                                  </div>
                                )}
                                {component.metadata.translation_vector_magnitude !== undefined &&
                                 component.metadata.translation_vector_magnitude > 0 && (
                                  <div style={{ marginTop: '2px' }}>
                                    Translation distance: {component.metadata.translation_vector_magnitude.toFixed(3)}
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </>
              ) : (
                // Fallback if breakdown not available
                <>
                  <div style={subSectionTitleStyle}>What went into this:</div>
                  <div style={{ fontSize: '0.8125rem', marginBottom: '8px' }}>
                    Composite metric combining:
                  </div>
                  <ul style={{
                    fontSize: '0.8125rem',
                    paddingLeft: '20px',
                    marginBottom: '12px',
                    lineHeight: '1.6',
                  }}>
                    <li><strong>Network exploration</strong>: {metrics.overlap !== undefined ? `1 - ${metrics.overlap.toFixed(2)} = ${(1 - metrics.overlap).toFixed(2)}` : 'N/A'}</li>
                    <li><strong>They get you</strong>: R²_in = {metrics.readability !== undefined ? metrics.readability.toFixed(3) : 'N/A'}</li>
                    <li><strong>You get their cluster</strong>: R²_out = {clusterMetrics?.subjectiveAttunement?.toFixed(3) || 'N/A'}</li>
                    <li><strong>Semantic relevance</strong>: Cosine similarity after translation</li>
                    <li><strong>Stability</strong>: No instability penalty in this data</li>
                  </ul>
                </>
              )}

              {edgeToFocal && (
                <>
                  <div style={labelStyle}>Connection strength</div>
                  <div style={codeBlockStyle}>
                    Actual: {edgeToFocal.actual !== undefined ? edgeToFocal.actual.toFixed(2) : 'N/A'}
                    {edgeToFocal.channels && edgeToFocal.channels.length > 0 && (
                      <div>Channels: {edgeToFocal.channels.join(', ')}</div>
                    )}
                  </div>
                </>
              )}

              <div style={explanationStyle}>
                Formula: λ₁·(1-overlap) + λ₂·R²_in + λ₃·R²_out + λ₄·cos(q_k, z_j) - λ₅·instability
              </div>
              <div style={explanationStyle}>
                (see [ego_ops.py:632-781](src/ego_ops.py#L632-L781))
              </div>
            </div>
          )}

          {/* Cluster Information */}
          {personCluster && (
            <div style={sectionStyle}>
              <div style={sectionTitleStyle}>Cluster Membership</div>
              <div style={{ fontSize: '0.875rem', marginBottom: '12px' }}>
                {fullPersonData.name} is in Cluster {clusterIndex} with:
              </div>
              <div style={phraseListStyle}>
                {personCluster.map(memberId => {
                  const member = egoGraphData.connections.find(c => c.id === memberId) ||
                                 (memberId === egoGraphData.self.id ? egoGraphData.self : null);
                  return (
                    <div key={memberId} style={{
                      fontWeight: memberId === person.id ? '700' : '400',
                    }}>
                      • {member?.name || memberId} {memberId === person.id && '(this person)'}
                    </div>
                  );
                })}
              </div>

              {clusterMetrics && isSemanticFlow && (
                <>
                  <div style={{ ...subSectionTitleStyle, marginTop: '16px' }}>Cluster Cohesion</div>
                  <div style={{ fontSize: '0.8125rem', lineHeight: '1.6' }}>
                    {clusterMetrics.coherence_MB !== undefined && (
                      <div>
                        This cluster holds together with <strong>{(clusterMetrics.coherence_MB * 100).toFixed(0)}%</strong> internal coherence.
                        {clusterMetrics.conductance_sem !== undefined && clusterMetrics.conductance_sem < 0.3 && (
                          <span> Clear boundaries.</span>
                        )}
                        {clusterMetrics.silhouette_D !== undefined && clusterMetrics.silhouette_D > 0.3 && (
                          <span> Well-separated from other groups.</span>
                        )}
                      </div>
                    )}
                  </div>
                </>
              )}

              {clusterMetrics && !isSemanticFlow && (
                <>
                  <div style={subSectionTitleStyle}>
                    Cluster Metrics
                  </div>
                  <div style={{ fontSize: '0.8125rem', lineHeight: '1.6' }}>
                    <div style={{ marginBottom: '8px' }}>
                      <strong>Public Legibility:</strong>{' '}
                      {getMetricLabel('public_legibility', clusterMetrics.publicLegibility).label}
                      <div style={explanationStyle}>
                        How well this cluster collectively understands you
                      </div>
                    </div>
                    {clusterMetrics.subjectiveAttunement !== undefined && (
                      <div style={{ marginBottom: '8px' }}>
                        <strong>Subjective Attunement:</strong>{' '}
                        {getMetricLabel('subjective_attunement', clusterMetrics.subjectiveAttunement).label}
                        <div style={explanationStyle}>
                          How well you understand this cluster's interests
                        </div>
                      </div>
                    )}
                    {clusterMetrics.heatResidualNovelty !== undefined && (
                      <div>
                        <strong>Novelty:</strong>{' '}
                        {getMetricLabel('heat_residual_novelty', clusterMetrics.heatResidualNovelty).label}
                        <div style={explanationStyle}>
                          Semantic distance - how new this cluster's territory is
                        </div>
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          )}

          {/* Person's Availability */}
          {fullPersonData.availability && fullPersonData.availability.length > 0 && (
            <div style={sectionStyle}>
              <div style={sectionTitleStyle}>Availability</div>
              {fullPersonData.availability.map((avail, idx) => (
                <div key={idx} style={{ fontSize: '0.8125rem', marginBottom: '8px' }}>
                  <div style={{ fontWeight: '600' }}>
                    {avail.date && `${avail.date}: `}
                    Score {avail.score.toFixed(1)}/1.0
                  </div>
                  {avail.content && (
                    <div style={explanationStyle}>{avail.content}</div>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Person's Notes */}
          {fullPersonData.notes && fullPersonData.notes.length > 0 && (
            <div style={sectionStyle}>
              <div style={sectionTitleStyle}>Notes</div>
              {fullPersonData.notes.map((note, idx) => (
                <div key={idx} style={{ fontSize: '0.8125rem', marginBottom: '12px' }}>
                  {note.date && (
                    <div style={{ ...labelStyle, marginBottom: '4px' }}>{note.date}</div>
                  )}
                  <div>{note.content}</div>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
