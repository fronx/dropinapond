# Drop in a Pond - GUI

React-based visualization interface for exploring ego graph networks.

## Features

- **Force-directed graph layout** using D3.js for natural positioning based on connection strength
- **Interactive node visualization** with xyflow/react
- **Dynamic routing** - load any ego graph by URL path (e.g., `/fronx`)
- **Rich node display** showing:
  - Person's name and top phrases
  - Capabilities/skills
  - Availability status
  - Notes and metadata
- **Edge annotations** showing connection strength as thickness and opacity

## Quick Start

```bash
# Install dependencies
npm install

# Start dev server
npm run dev

# Build for production
npm run build
```

Then open your browser to `http://localhost:5173` and enter a graph name (e.g., "fronx").

Or navigate directly to `http://localhost:5173/fronx` to view the fronx ego graph.

## Architecture

- **EgoGraphView**: Main visualization component that loads data and orchestrates the display
- **PersonNode**: Custom xyflow node component for displaying person information
- **egoGraphLoader**: Utilities for loading and parsing ego graph JSON files
- **d3Layout**: D3 force-directed layout algorithm that positions nodes based on connection strength

## Data Loading

The app loads ego graph JSON files from `/data/ego_graphs/[name].json` via a symlink in the `public/` directory.

The symlink (`public/data -> ../../data`) allows Vite to serve the parent `data/` directory during development.
