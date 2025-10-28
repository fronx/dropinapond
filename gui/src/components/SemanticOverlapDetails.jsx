import React from 'react';

/**
 * Displays the phrase-level semantic overlap between focal node and selected person.
 * Shows what connects them (similar phrases) and what's unique to each.
 */
export default function SemanticOverlapDetails({
  similarPhrases,
  uniquePersonPhrases,
  uniqueSelfPhrases,
  isDarkMode
}) {
  const labelStyle = {
    fontSize: '0.75rem',
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    color: isDarkMode ? '#9ca3af' : '#6b7280',
    marginBottom: '8px'
  };

  const phraseListStyle = {
    marginTop: '8px',
    padding: '8px',
    backgroundColor: isDarkMode ? '#1f2937' : '#f9fafb',
    borderRadius: '4px',
    fontSize: '0.8125rem',
    color: isDarkMode ? '#d1d5db' : '#374151'
  };

  return (
    <>
      {/* What connects you */}
      {similarPhrases.length > 0 && (
        <>
          <div style={{ ...labelStyle, marginTop: '12px' }}>
            What connects you ({similarPhrases.length})
          </div>
          <div style={phraseListStyle}>
            {similarPhrases.slice(0, 8).map((match, idx) => (
              <div
                key={idx}
                style={{
                  marginBottom: '8px',
                  paddingBottom: '8px',
                  borderBottom: idx < Math.min(7, similarPhrases.length - 1)
                    ? `1px solid ${isDarkMode ? '#374151' : '#e5e7eb'}`
                    : 'none'
                }}
              >
                <div style={{
                  color: isDarkMode ? '#93c5fd' : '#1e40af',
                  fontWeight: '600',
                  fontSize: '0.75rem'
                }}>
                  You: {match.focal_phrase}
                </div>
                <div style={{
                  color: isDarkMode ? '#d1d5db' : '#4b5563',
                  fontSize: '0.75rem'
                }}>
                  Them: {match.neighbor_phrase}
                </div>
                <div style={{
                  color: isDarkMode ? '#6b7280' : '#9ca3af',
                  fontSize: '0.7rem',
                  marginTop: '2px'
                }}>
                  {Math.round(match.similarity * 100)}% semantic similarity
                </div>
              </div>
            ))}
            {similarPhrases.length > 8 && (
              <div style={{
                marginTop: '8px',
                color: isDarkMode ? '#6b7280' : '#9ca3af',
                fontSize: '0.75rem'
              }}>
                ...and {similarPhrases.length - 8} more overlaps
              </div>
            )}
          </div>
        </>
      )}

      {/* What's unique to them */}
      {uniquePersonPhrases.length > 0 && (
        <>
          <div style={{ ...labelStyle, marginTop: '12px' }}>
            What's unique to them (top 5)
          </div>
          <div style={phraseListStyle}>
            {uniquePersonPhrases.slice(0, 5).map((phrase, idx) => (
              <div key={idx}>
                • {phrase.text}
                {phrase.weight && (
                  <span style={{
                    color: isDarkMode ? '#6b7280' : '#9ca3af',
                    marginLeft: '4px'
                  }}>
                    ({phrase.weight.toFixed(2)})
                  </span>
                )}
              </div>
            ))}
          </div>
        </>
      )}

      {/* What's unique to you */}
      {uniqueSelfPhrases.length > 0 && (
        <>
          <div style={{ ...labelStyle, marginTop: '12px' }}>
            What's unique to you (top 5)
          </div>
          <div style={phraseListStyle}>
            {uniqueSelfPhrases.slice(0, 5).map((phrase, idx) => (
              <div key={idx}>
                • {phrase.text}
                {phrase.weight && (
                  <span style={{
                    color: isDarkMode ? '#6b7280' : '#9ca3af',
                    marginLeft: '4px'
                  }}>
                    ({phrase.weight.toFixed(2)})
                  </span>
                )}
              </div>
            ))}
          </div>
        </>
      )}
    </>
  );
}
