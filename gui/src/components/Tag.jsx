/**
 * Reusable tag bubble component with consistent styling
 */
export default function Tag({
  children,
  variant = 'shared',
  similarity = 1.0,
  isDarkMode,
  onMouseEnter,
  onMouseLeave,
  style = {}
}) {
  const baseRadius = '4px';

  const getVariantStyle = () => {
    if (variant === 'unique') {
      // Unique phrases: neutral gray, low opacity
      return {
        border: `1px solid ${isDarkMode ? '#4b5563' : '#d1d5db'}`,
        backgroundColor: isDarkMode ? 'rgba(55, 65, 81, 0.2)' : 'rgba(243, 244, 246, 0.5)',
        color: isDarkMode ? '#9ca3af' : '#6b7280',
        opacity: 0.7,
        cursor: 'default'
      };
    }

    // Shared phrases: blue colors with left bar indicator
    const isExactMatch = similarity >= 0.99;
    const opacity = Math.max(0.7, similarity * 0.3 + 0.7);

    // Left bar indicates match strength
    const barWidth = isExactMatch ? '4px' : '2px';
    const barColor = isDarkMode ? '#60a5fa' : '#3b82f6';
    const bgColor = isDarkMode ? 'rgba(37, 99, 235, 0.15)' : 'rgba(96, 165, 250, 0.15)';

    // Create a gradient that makes a vertical bar on the left
    const backgroundImage = `linear-gradient(to right, ${barColor} 0%, ${barColor} ${barWidth}, ${bgColor} ${barWidth}, ${bgColor} 100%)`;

    return {
      border: `1px solid ${isDarkMode ? '#3b82f6' : '#60a5fa'}`,
      backgroundImage: backgroundImage,
      color: isDarkMode ? '#e5e7eb' : '#1e40af',
      opacity: opacity,
      cursor: isExactMatch ? 'default' : 'help'
    };
  };

  return (
    <div
      style={{
        display: 'inline-block',
        padding: '6px 12px',
        borderRadius: baseRadius,
        fontSize: '0.8125rem',
        position: 'relative',
        transition: 'all 0.15s ease',
        ...getVariantStyle(),
        ...style
      }}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      {children}
    </div>
  );
}
