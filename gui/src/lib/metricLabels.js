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
      { max: 0.05, label: 'lost in noise', description: 'Very low relative coupling', interpretation: 'Among all your options, this connection barely registers.' },
      { max: 0.15, label: 'one of many', description: 'Low relative prominence', interpretation: 'This is just one of many possible attention targets for both of you.' },
      { max: 0.30, label: 'noticeable', description: 'Moderate relative coupling', interpretation: 'This relationship stands out somewhat from your other options.' },
      { max: 0.50, label: 'prominent', description: 'High relative coupling', interpretation: 'Given all your options, this connection is quite prominent.' },
      { max: 1.0, label: 'magnetic', description: 'Very high relative coupling', interpretation: 'This relationship strongly stands out - you pull toward each other despite many options.' },
    ],
    explorationPotential: [
      { max: 0.05, label: 'high friction', description: 'Low-opportunity terrain', interpretation: 'A next step here would require significant effort or feel awkward.' },
      { max: 0.15, label: 'some friction', description: 'Moderate barriers to next step', interpretation: 'There\'s some opportunity here but with noticeable friction.' },
      { max: 0.30, label: 'smooth path', description: 'Good opportunity surface', interpretation: 'A next step would feel relatively natural and low-effort.' },
      { max: 0.50, label: 'easy opening', description: 'High opportunity potential', interpretation: 'Easy, low-friction opportunity - a natural next step is obvious.' },
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
