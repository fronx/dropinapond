import { useState } from 'react';
import Tag from './Tag';

/**
 * Displays phrase-level semantic overlap as flowing tags.
 * Visual indicators:
 * - Left border thickness = how similar (thick = exact match, thin = partial)
 * - Opacity = semantic similarity strength
 * - Hover = shows your matching phrase if different
 */
export default function SemanticOverlapDetails({
  similarPhrases,
  uniquePersonPhrases,
  isDarkMode
}) {
  const [hoveredTag, setHoveredTag] = useState(null);

  const containerStyle = {
    marginTop: '12px'
  };

  const sectionStyle = {
    marginBottom: '16px'
  };

  const tagContainerStyle = {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '8px',
    marginTop: '8px'
  };

  const tooltipStyle = {
    position: 'absolute',
    bottom: '100%',
    left: '50%',
    transform: 'translateX(-50%)',
    marginBottom: '8px',
    padding: '6px 10px',
    backgroundColor: isDarkMode ? '#111827' : '#1f2937',
    color: '#e5e7eb',
    fontSize: '0.75rem',
    borderRadius: '6px',
    whiteSpace: 'nowrap',
    boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
    zIndex: 1000,
    pointerEvents: 'none'
  };

  const labelStyle = {
    fontSize: '0.75rem',
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    color: isDarkMode ? '#9ca3af' : '#6b7280',
    marginBottom: '4px'
  };

  return (
    <div style={containerStyle}>
      {/* Shared interests */}
      {similarPhrases.length > 0 && (
        <div style={sectionStyle}>
          <div style={labelStyle}>
            Shared interests ({similarPhrases.length})
          </div>
          <div style={tagContainerStyle}>
            {similarPhrases.map((match, idx) => {
              const isExactMatch = match.similarity >= 0.99;
              const isDifferent = match.focal_phrase !== match.neighbor_phrase;
              const showTooltip = hoveredTag === `shared-${idx}` && !isExactMatch && isDifferent;

              return (
                <Tag
                  key={idx}
                  variant="shared"
                  similarity={match.similarity}
                  isDarkMode={isDarkMode}
                  onMouseEnter={() => setHoveredTag(`shared-${idx}`)}
                  onMouseLeave={() => setHoveredTag(null)}
                >
                  {match.neighbor_phrase}
                  {showTooltip && (
                    <div style={tooltipStyle}>
                      Your phrase: {match.focal_phrase}
                    </div>
                  )}
                </Tag>
              );
            })}
          </div>
        </div>
      )}

      {/* Unique to them */}
      {uniquePersonPhrases.length > 0 && (
        <div style={sectionStyle}>
          <div style={labelStyle}>
            Unique to them (top {Math.min(10, uniquePersonPhrases.length)})
          </div>
          <div style={tagContainerStyle}>
            {uniquePersonPhrases.slice(0, 10).map((phrase, idx) => (
              <Tag
                key={idx}
                variant="unique"
                isDarkMode={isDarkMode}
              >
                {phrase.text}
              </Tag>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
