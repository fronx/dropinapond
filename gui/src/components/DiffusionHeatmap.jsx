import { useState, useMemo } from 'react';

const ROLE_COLORS = {
  core: '#ef4444',      // red
  bridge: '#f59e0b',    // amber
  isolated: '#6b7280',  // gray
  even: '#3b82f6',      // blue
};

export function DiffusionHeatmap({ diffusionData }) {
  const [timeStep, setTimeStep] = useState('t2'); // Start with 2-step diffusion
  const [hoveredCell, setHoveredCell] = useState(null);

  if (!diffusionData) {
    return (
      <div className="p-4 text-gray-500">
        No diffusion data available
      </div>
    );
  }

  const {
    node_order,
    node_names,
    roles,
    matrices,
    node_metrics
  } = diffusionData;

  const matrix = matrices[timeStep];
  const n = node_order.length;

  // Compute color scale (log scale for better contrast)
  const getHeatColor = (value) => {
    if (value < 0.001) return 'rgb(255, 255, 255)'; // white for near-zero

    // Log scale: map [0.001, 1] to [0, 1] on log scale
    const logMin = Math.log(0.001);
    const logMax = Math.log(1);
    const logVal = Math.log(Math.max(value, 0.001));
    const intensity = (logVal - logMin) / (logMax - logMin);

    // Blue-to-red gradient
    const blue = Math.round(255 * (1 - intensity));
    const red = Math.round(255 * intensity);
    return `rgb(${red}, ${blue/2}, ${blue})`;
  };

  // Get role color for row/column headers
  const getRoleColor = (nodeId) => {
    const role = roles[nodeId] || 'even';
    return ROLE_COLORS[role];
  };

  const cellSize = Math.min(30, Math.max(15, 600 / n)); // Adaptive cell size

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', backgroundColor: 'white' }}>
      {/* Header */}
      <div style={{ padding: '1rem', borderBottom: '1px solid #e5e7eb' }}>
        <h2 className="text-lg font-semibold mb-2">Diffusion Heatmap</h2>
        <p className="text-sm text-gray-600 mb-3">
          Shows how probability diffuses through the network. Warmer colors = higher probability flow.
        </p>

        {/* Time step selector */}
        <div className="flex gap-2 items-center">
          <span className="text-sm font-medium">Time steps:</span>
          {['t1', 't2', 't3'].map(t => (
            <button
              key={t}
              onClick={() => setTimeStep(t)}
              className={`px-3 py-1 rounded text-sm ${
                timeStep === t
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {t === 't1' ? '1' : t === 't2' ? '2' : '3'}
            </button>
          ))}
        </div>

        {/* Legend */}
        <div className="mt-3 flex flex-wrap gap-3 text-xs">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: ROLE_COLORS.core }} />
            <span>Core</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: ROLE_COLORS.bridge }} />
            <span>Bridge</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: ROLE_COLORS.isolated }} />
            <span>Isolated</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: ROLE_COLORS.even }} />
            <span>Even</span>
          </div>
        </div>
      </div>

      {/* Heatmap */}
      <div className="flex-1 overflow-auto p-4">
        <div className="inline-block">
          {/* Column headers (vertical names) */}
          <div className="flex" style={{ marginLeft: `${cellSize + 5}px` }}>
            {node_order.map((nodeId, j) => (
              <div
                key={j}
                className="flex items-end justify-center"
                style={{
                  width: `${cellSize}px`,
                  height: '100px',
                  transform: 'rotate(-45deg)',
                  transformOrigin: 'left bottom',
                }}
              >
                <span
                  className="text-xs whitespace-nowrap font-medium"
                  style={{ color: getRoleColor(nodeId) }}
                >
                  {node_names[j]}
                </span>
              </div>
            ))}
          </div>

          {/* Matrix rows */}
          {matrix.map((row, i) => (
            <div key={i} className="flex items-center">
              {/* Row header */}
              <div
                className="text-right pr-2 text-xs font-medium whitespace-nowrap"
                style={{
                  width: `${cellSize}px`,
                  color: getRoleColor(node_order[i])
                }}
              >
                {node_names[i]}
              </div>

              {/* Row cells */}
              {row.map((value, j) => {
                const isDiagonal = i === j;
                return (
                  <div
                    key={j}
                    className="border border-gray-200 cursor-pointer relative"
                    style={{
                      width: `${cellSize}px`,
                      height: `${cellSize}px`,
                      backgroundColor: getHeatColor(value),
                      opacity: isDiagonal ? 0.7 : 1,
                    }}
                    onMouseEnter={() => setHoveredCell({ i, j, value })}
                    onMouseLeave={() => setHoveredCell(null)}
                  >
                    {isDiagonal && (
                      <div className="absolute inset-0 flex items-center justify-center text-xs font-bold opacity-50">
                        ·
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          ))}
        </div>

        {/* Hover tooltip */}
        {hoveredCell && (
          <div className="mt-4 p-3 bg-gray-50 rounded border border-gray-200 text-sm">
            <div className="font-semibold mb-1">
              {node_names[hoveredCell.i]} → {node_names[hoveredCell.j]}
            </div>
            <div className="text-gray-600">
              Probability: {hoveredCell.value.toFixed(4)}
            </div>
            {hoveredCell.i === hoveredCell.j && (
              <div className="text-gray-500 text-xs mt-1">
                (Return probability after {timeStep === 't1' ? '1' : timeStep === 't2' ? '2' : '3'} steps)
              </div>
            )}
          </div>
        )}
      </div>

      {/* Color scale legend */}
      <div className="p-4 border-t border-gray-200">
        <div className="text-xs text-gray-600 mb-1">Probability scale (log)</div>
        <div className="flex h-4 rounded overflow-hidden">
          {Array.from({ length: 100 }, (_, i) => {
            const val = 0.001 * Math.exp((i / 99) * Math.log(1000));
            return (
              <div
                key={i}
                style={{
                  flex: 1,
                  backgroundColor: getHeatColor(val)
                }}
              />
            );
          })}
        </div>
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>0.001</span>
          <span>0.01</span>
          <span>0.1</span>
          <span>1.0</span>
        </div>
      </div>
    </div>
  );
}
