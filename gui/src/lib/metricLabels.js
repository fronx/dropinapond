// Import the metric labels JSON
import metricLabelsData from '../../../src/metric_labels.json';

/**
 * Get the label for a metric value
 * @param {string} metricName - 'readability', 'overlap', or 'orientation_score'
 * @param {number} value - The metric value
 * @returns {{label: string, description: string}} The label and description
 */
export function getMetricLabel(metricName, value) {
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
