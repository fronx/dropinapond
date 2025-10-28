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
 * Get human-readable interpretation of semantic flow metrics with absolute thresholds.
 * Only use for metrics with clear absolute meaning or good natural spread.
 * For narrow-range metrics, use getPercentileLabel() instead.
 *
 * @param {string} metricName - Name of the semantic flow metric
 * @param {number} value - The metric value
 * @returns {{label: string, description: string, interpretation: string}} Metric information
 */
export function getSemanticFlowLabel(metricName, value) {
  if (value === undefined || value === null) {
    return { label: 'N/A', description: 'No data available', interpretation: '' };
  }

  // Only include metrics with absolute meaning or good natural spread
  const interpretations = {
    structuralEdge: [
      { max: 0.2, label: 'minimal contact', description: 'Very little actual interaction', interpretation: 'You rarely interact with this person in practice.' },
      { max: 0.4, label: 'occasional', description: 'Some interaction but infrequent', interpretation: 'You interact occasionally, but not regularly.' },
      { max: 0.6, label: 'regular', description: 'Consistent ongoing interaction', interpretation: 'You have regular, ongoing contact with this person.' },
      { max: 0.8, label: 'frequent', description: 'Strong and frequent interaction', interpretation: 'You interact frequently and consistently.' },
      { max: 1.0, label: 'primary', description: 'Very high interaction level', interpretation: 'This is one of your primary relationships.' },
    ],
    distanceRaw: [
      { max: 0.2, label: 'very close', description: 'Centers of gravity nearly aligned', interpretation: 'Your semantic centers are very close - you occupy similar conceptual space.' },
      { max: 0.4, label: 'nearby', description: 'Close but distinct centers', interpretation: 'Close enough to share territory but with distinct focuses.' },
      { max: 0.6, label: 'moderate gap', description: 'Significant separation', interpretation: 'There\'s a moderate gap between your conceptual centers.' },
      { max: 0.8, label: 'far apart', description: 'Large semantic distance', interpretation: 'Your conceptual centers are quite far apart.' },
      { max: 1.0, label: 'very distant', description: 'Very different semantic fields', interpretation: 'You occupy very different parts of semantic space.' },
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
  } else if (metricName === 'effectiveEdge' && analysisData.metrics.layers?.effective_edges?.[egoId]) {
    Object.values(analysisData.metrics.layers.effective_edges[egoId]).forEach(v => values.push(v));
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
    effectiveEdge: computeMetricPercentiles('effectiveEdge', analysisData, egoId),
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
 * Get percentile-based labels for metrics with narrow natural ranges.
 * Use this for metrics that cluster tightly and need relative interpretation.
 *
 * @param {string} metricName - Name of the metric
 * @param {number} value - The metric value
 * @param {Object} percentiles - Percentiles for this metric {p20, p40, p60, p80}
 * @returns {{label: string, interpretation: string}|null} Label and interpretation or null if not enough data
 */
export function getPercentileLabel(metricName, value, percentiles) {
  if (value === undefined || value === null || !percentiles) return null;

  const labelDefinitions = {
    semanticAffinity: [
      { label: 'quite different topics', interpretation: 'Your conceptual focuses are further apart than most of your connections.' },
      { label: 'somewhat aligned', interpretation: 'You share some conceptual territory, though not as much as your closer connections.' },
      { label: 'good alignment', interpretation: 'Your interests align well - typical for your network.' },
      { label: 'strong resonance', interpretation: 'Stronger conceptual overlap than most of your connections.' },
      { label: 'exceptional alignment', interpretation: 'Among your closest semantic matches - you think about very similar things.' },
    ],
    predictabilityRaw: [
      { label: 'lower mutual pull', interpretation: 'The semantic pull between you is weaker than most of your connections.' },
      { label: 'moderate reciprocity', interpretation: 'There\'s mutual semantic pull, though not as strong as your top connections.' },
      { label: 'balanced pull', interpretation: 'Good reciprocity - you both pull toward each other conceptually.' },
      { label: 'strong reciprocity', interpretation: 'Strong mutual pull - collaboration feels natural and balanced.' },
      { label: 'exceptional reciprocity', interpretation: 'Among your strongest mutual connections - you can easily anticipate each other.' },
    ],
    effectiveEdge: [
      { label: 'weaker link', interpretation: 'Structure and meaning combine to make this a less prominent connection.' },
      { label: 'moderate link', interpretation: 'A moderate connection when structure and meaning combine.' },
      { label: 'solid link', interpretation: 'A solid connection where structure and meaning reinforce each other.' },
      { label: 'strong link', interpretation: 'Among your stronger links by both practice and semantic fit.' },
      { label: 'primary link', interpretation: 'One of your strongest links by both practice and meaning.' },
    ],
    predictabilityBlanket: [
      { label: 'quieter signal', interpretation: 'Given all your options, this connection is less prominent than most.' },
      { label: 'one of many', interpretation: 'This is one of several comparable attention targets.' },
      { label: 'noticeable', interpretation: 'This relationship stands out somewhat from your other options.' },
      { label: 'prominent', interpretation: 'This connection is quite prominent among all your options.' },
      { label: 'magnetic', interpretation: 'This relationship strongly stands out - you pull toward each other despite many options.' },
    ],
    explorationPotential: [
      { label: 'higher friction', interpretation: 'A next step here has more friction than most of your options.' },
      { label: 'some friction', interpretation: 'There\'s opportunity here but with moderate friction.' },
      { label: 'smooth path', interpretation: 'A next step would feel relatively natural.' },
      { label: 'easy opening', interpretation: 'Among your easier opportunities - a natural next step is clear.' },
      { label: 'ripe opportunity', interpretation: 'One of your ripest opportunities - very low friction and high potential.' },
    ],
  };

  const labels = labelDefinitions[metricName];
  if (!labels) return null;

  // Find which quintile bucket the value falls into
  const thresholds = [percentiles.p20, percentiles.p40, percentiles.p60, percentiles.p80];
  const bucketIndex = thresholds.findIndex(threshold => value <= threshold);
  return labels[bucketIndex === -1 ? 4 : bucketIndex];
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
