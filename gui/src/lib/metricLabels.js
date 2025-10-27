// Import the metric labels JSON
import metricLabelsData from '../../../src/metric_labels.json';

/**
 * Get the label for a metric value
 * @param {string} metricName - 'readability', 'overlap', or 'orientation_score'
 * @param {number} value - The metric value
 * @returns {{label: string, description: string}} The label and description
 */
export function getMetricLabel(metricName, value) {
  if (value === undefined || value === null) {
    return { label: 'N/A', description: 'No data available' };
  }

  const metric = metricLabelsData.metrics[metricName];

  if (!metric) {
    return { label: value.toFixed(2), description: '' };
  }

  // Find the matching range
  const range = metric.labels.find(r => value >= r.min && value < r.max);

  // Handle edge case where value equals the max of the last range
  const lastRange = metric.labels[metric.labels.length - 1];
  if (!range && value === lastRange.max) {
    return { label: lastRange.label, description: lastRange.description };
  }

  if (!range) {
    return { label: value.toFixed(2), description: 'Out of range' };
  }

  return { label: range.label, description: range.description };
}

/**
 * Get human-readable interpretation of semantic flow metrics
 * @param {string} metricName - Name of the semantic flow metric
 * @param {number} value - The metric value
 * @returns {{label: string, description: string, interpretation: string}} Metric information
 */
export function getSemanticFlowLabel(metricName, value) {
  if (value === undefined || value === null) {
    return { label: 'N/A', description: 'No data available', interpretation: '' };
  }

  const interpretations = {
    structuralEdge: [
      { max: 0.2, label: 'minimal contact', description: 'Very little actual interaction', interpretation: 'You rarely interact with this person in practice.' },
      { max: 0.4, label: 'occasional', description: 'Some interaction but infrequent', interpretation: 'You interact occasionally, but not regularly.' },
      { max: 0.6, label: 'regular', description: 'Consistent ongoing interaction', interpretation: 'You have regular, ongoing contact with this person.' },
      { max: 0.8, label: 'frequent', description: 'Strong and frequent interaction', interpretation: 'You interact frequently and consistently.' },
      { max: 1.0, label: 'primary', description: 'Very high interaction level', interpretation: 'This is one of your primary relationships.' },
    ],
    semanticAffinity: [
      { max: 0.2, label: 'distinct topics', description: 'Very different conceptual focus', interpretation: 'Your topics and theirs have little conceptual overlap.' },
      { max: 0.4, label: 'somewhat aligned', description: 'Some semantic resonance', interpretation: 'You share some conceptual territory, but still quite different.' },
      { max: 0.6, label: 'good alignment', description: 'Significant thematic overlap', interpretation: 'Your interests and theirs align well on key themes.' },
      { max: 0.8, label: 'strong resonance', description: 'High semantic overlap', interpretation: 'Strong conceptual resonance - you think about similar things.' },
      { max: 1.0, label: 'deep alignment', description: 'Very high semantic affinity', interpretation: 'Deep thematic alignment - almost identical conceptual focus.' },
    ],
    effectiveEdge: [
      { max: 0.2, label: 'weak link', description: 'Low combined weight', interpretation: 'Both structure and meaning suggest a weak connection.' },
      { max: 0.4, label: 'moderate link', description: 'Some combined strength', interpretation: 'A moderate connection when structure and meaning combine.' },
      { max: 0.6, label: 'solid link', description: 'Good combined strength', interpretation: 'A solid connection - structure and meaning reinforce each other.' },
      { max: 0.8, label: 'strong link', description: 'High blended weight', interpretation: 'A strong link where actual interaction and semantic fit align well.' },
      { max: 1.0, label: 'primary link', description: 'Very high combined strength', interpretation: 'One of your strongest links by both practice and meaning.' },
    ],
    predictabilityRaw: [
      { max: 0.2, label: 'one-sided pull', description: 'Very asymmetric affinity', interpretation: 'The semantic pull is mostly one-way - little mutual resonance.' },
      { max: 0.4, label: 'moderate reciprocity', description: 'Some mutual pull', interpretation: 'There\'s some mutual semantic pull, but not strongly balanced.' },
      { max: 0.6, label: 'balanced pull', description: 'Good mutual predictability', interpretation: 'Good reciprocity - you both pull toward each other conceptually.' },
      { max: 0.8, label: 'strong reciprocity', description: 'High mutual affinity', interpretation: 'Strong mutual pull - collaboration feels natural and balanced.' },
      { max: 1.0, label: 'deep reciprocity', description: 'Very high mutual pull', interpretation: 'Deep reciprocal resonance - you can easily anticipate each other.' },
    ],
    distanceRaw: [
      { max: 0.2, label: 'very close', description: 'Centers of gravity nearly aligned', interpretation: 'Your semantic centers are very close - you occupy similar conceptual space.' },
      { max: 0.4, label: 'nearby', description: 'Close but distinct centers', interpretation: 'Close enough to share territory but with distinct focuses.' },
      { max: 0.6, label: 'moderate gap', description: 'Significant separation', interpretation: 'There\'s a moderate gap between your conceptual centers.' },
      { max: 0.8, label: 'far apart', description: 'Large semantic distance', interpretation: 'Your conceptual centers are quite far apart.' },
      { max: 1.0, label: 'very distant', description: 'Very different semantic fields', interpretation: 'You occupy very different parts of semantic space.' },
    ],
    predictabilityBlanket: [
      { max: 0.04, label: 'lost in noise', description: 'Very low relative coupling', interpretation: 'Among all your options, this connection barely registers.' },
      { max: 0.07, label: 'one of many', description: 'Low relative prominence', interpretation: 'This is just one of many possible attention targets for both of you.' },
      { max: 0.10, label: 'noticeable', description: 'Moderate relative coupling', interpretation: 'This relationship stands out somewhat from your other options.' },
      { max: 0.15, label: 'prominent', description: 'High relative coupling', interpretation: 'Given all your options, this connection is quite prominent.' },
      { max: 1.0, label: 'magnetic', description: 'Very high relative coupling', interpretation: 'This relationship strongly stands out - you pull toward each other despite many options.' },
    ],
    explorationPotential: [
      { max: 0.02, label: 'high friction', description: 'Low-opportunity terrain', interpretation: 'A next step here would require significant effort or feel awkward.' },
      { max: 0.035, label: 'some friction', description: 'Moderate barriers to next step', interpretation: 'There\'s some opportunity here but with noticeable friction.' },
      { max: 0.05, label: 'smooth path', description: 'Good opportunity surface', interpretation: 'A next step would feel relatively natural and low-effort.' },
      { max: 0.08, label: 'easy opening', description: 'High opportunity potential', interpretation: 'Easy, low-friction opportunity - a natural next step is obvious.' },
      { max: 1.0, label: 'ripe opportunity', description: 'Optimal exploration potential', interpretation: 'Prime opportunity - strong coupling and close distance make this very easy.' },
    ],
    fitRatio: [
      { max: 0.5, label: 'strong bridge', description: 'Couples more outside than inside', interpretation: 'This person is a bridge or ambassador - they connect to other clusters more than their own.' },
      { max: 1.0, label: 'boundary node', description: 'Balanced coupling', interpretation: 'This person sits on a cluster boundary, connecting equally inside and outside.' },
      { max: 2.0, label: 'good fit', description: 'Couples more inside than outside', interpretation: 'This person fits well in their cluster, though not exclusively.' },
      { max: 5.0, label: 'strong anchor', description: 'Strongly coupled to cluster', interpretation: 'This person is strongly anchored in their cluster.' },
      { max: Infinity, label: 'core member', description: 'Very high cluster coupling', interpretation: 'This person is a core member of their cluster with very few outside ties.' },
    ],
  };

  const ranges = interpretations[metricName];
  if (!ranges) {
    return {
      label: value.toFixed(3),
      description: '',
      interpretation: `Metric value: ${value.toFixed(3)}`
    };
  }

  const match = ranges.find(r => value <= r.max);
  if (match) {
    return {
      label: match.label,
      description: match.description,
      interpretation: match.interpretation
    };
  }

  // Fallback for values outside ranges
  return {
    label: value.toFixed(3),
    description: 'Out of range',
    interpretation: `Metric value: ${value.toFixed(3)}`
  };
}

/**
 * Get a formatted display string for multiple metrics
 * @param {Object} metrics - Object with metric values
 * @returns {string} Formatted string like "well understood · different worlds · high potential"
 */
export function formatMetrics(metrics) {
  const parts = [];

  if (metrics.readability !== undefined) {
    parts.push(getMetricLabel('readability', metrics.readability).label);
  }

  if (metrics.overlap !== undefined) {
    parts.push(getMetricLabel('overlap', metrics.overlap).label);
  }

  if (metrics.orientationScore !== undefined) {
    parts.push(getMetricLabel('orientation_score', metrics.orientationScore).label);
  }

  return parts.join(' · ');
}

// ============================================================================
// Percentile-based metric interpretation (relative to user's network)
// ============================================================================

/**
 * Compute percentiles for a specific metric across the ego's network.
 * @param {string} metricName - Name of the metric
 * @param {Object} analysisData - The full analysis data
 * @param {string} egoId - The ego node's ID
 * @returns {Object|null} Percentiles {p20, p40, p60, p80, median}
 */
export function computeMetricPercentiles(metricName, analysisData, egoId) {
  if (!analysisData?.metrics) return null;

  const values = [];

  if (metricName === 'semanticAffinity' && analysisData.metrics.layers?.semantic_affinity?.[egoId]) {
    Object.values(analysisData.metrics.layers.semantic_affinity[egoId]).forEach(v => values.push(v));
  } else if (metricName === 'predictabilityRaw' && analysisData.metrics.fields?.edge_fields?.[egoId]) {
    Object.values(analysisData.metrics.fields.edge_fields[egoId]).forEach(obj => values.push(obj.predictability_raw));
  } else if (metricName === 'distanceRaw' && analysisData.metrics.fields?.edge_fields?.[egoId]) {
    Object.values(analysisData.metrics.fields.edge_fields[egoId]).forEach(obj => values.push(obj.distance_raw));
  } else if (metricName === 'predictabilityBlanket' && analysisData.metrics.fields?.edge_fields_blanket?.[egoId]) {
    Object.values(analysisData.metrics.fields.edge_fields_blanket[egoId]).forEach(obj => values.push(obj.predictability_blanket));
  } else if (metricName === 'explorationPotential' && analysisData.metrics.fields?.edge_fields_blanket?.[egoId]) {
    Object.values(analysisData.metrics.fields.edge_fields_blanket[egoId]).forEach(obj => values.push(obj.exploration_potential));
  }

  if (values.length === 0) return null;

  values.sort((a, b) => a - b);
  return {
    p20: values[Math.floor(values.length * 0.2)],
    p40: values[Math.floor(values.length * 0.4)],
    p60: values[Math.floor(values.length * 0.6)],
    p80: values[Math.floor(values.length * 0.8)],
    median: values[Math.floor(values.length * 0.5)],
  };
}

/**
 * Compute all percentiles for semantic flow metrics.
 * @param {Object} analysisData - The full analysis data
 * @param {string} egoId - The ego node's ID
 * @returns {Object} Map of metric names to percentiles
 */
export function computeAllPercentiles(analysisData, egoId) {
  if (!analysisData?.metrics?.layers) return null;

  return {
    semanticAffinity: computeMetricPercentiles('semanticAffinity', analysisData, egoId),
    predictabilityRaw: computeMetricPercentiles('predictabilityRaw', analysisData, egoId),
    distanceRaw: computeMetricPercentiles('distanceRaw', analysisData, egoId),
    predictabilityBlanket: computeMetricPercentiles('predictabilityBlanket', analysisData, egoId),
    explorationPotential: computeMetricPercentiles('explorationPotential', analysisData, egoId),
  };
}

/**
 * Get a human-readable description of structural interaction level.
 * Uses absolute thresholds since structural edges have absolute meaning (0-1 scale).
 * @param {number} structuralEdge - Structural edge weight (0-1)
 * @returns {string} Description
 */
export function getStructuralDescription(structuralEdge) {
  if (structuralEdge === undefined || structuralEdge === null) return null;

  if (structuralEdge >= 0.8) return 'One of your most frequent contacts.';
  if (structuralEdge >= 0.5) return 'Regular ongoing interaction.';
  if (structuralEdge >= 0.3) return 'Occasional contact.';
  return 'Infrequent interaction.';
}

/**
 * Get a human-readable description of semantic distance relative to network.
 * Uses percentiles since semantic distance is relative to your network.
 * @param {number} distance - Semantic distance (0-1)
 * @param {Object} percentiles - Percentiles for distanceRaw metric
 * @returns {string|null} Description or null if not enough data
 */
export function getSemanticDistanceDescription(distance, percentiles) {
  if (distance === undefined || distance === null || !percentiles) return null;

  if (distance <= percentiles.p20) return 'Among your closest in conceptual space.';
  if (distance <= percentiles.p40) return 'Closer than most in your network.';
  if (distance <= percentiles.p60) return 'Moderate distance in conceptual space.';
  if (distance <= percentiles.p80) return 'Further than most in your network.';
  return 'Among your most distant in conceptual space.';
}

/**
 * Extract metrics for a person from semantic flow analysis data.
 * @param {Object} analysisData - The full analysis data
 * @param {string} egoId - The ego node's ID
 * @param {string} personId - The person's ID
 * @returns {Object} Extracted metrics
 */
export function extractSemanticFlowMetrics(analysisData, egoId, personId) {
  if (!analysisData?.metrics?.layers) return {};

  return {
    structuralEdge: analysisData.metrics.layers.structural_edges?.[egoId]?.[personId],
    semanticAffinity: analysisData.metrics.layers.semantic_affinity?.[egoId]?.[personId],
    effectiveEdge: analysisData.metrics.layers.effective_edges?.[egoId]?.[personId],
    predictabilityRaw: analysisData.metrics.fields?.edge_fields?.[egoId]?.[personId]?.predictability_raw,
    distanceRaw: analysisData.metrics.fields?.edge_fields?.[egoId]?.[personId]?.distance_raw,
    predictabilityBlanket: analysisData.metrics.fields?.edge_fields_blanket?.[egoId]?.[personId]?.predictability_blanket,
    explorationPotential: analysisData.metrics.fields?.edge_fields_blanket?.[egoId]?.[personId]?.exploration_potential,
    coherenceNode: analysisData.metrics.coherence?.nodes?.[personId],
  };
}
