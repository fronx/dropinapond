# Drop in a Pond - GUI

React-based visualization interface for exploring your ego graph network.

## Features

- **Force-directed graph layout** using D3.js for natural positioning based on connection strength
- **Interactive node visualization** with xyflow/react
- **Single-graph model** - visualizes your personal ego graph
- **Rich node display** showing:
  - Person's name and semantic clusters
  - Analysis metrics (coherence, orientation scores)
  - Connection strength visualization
  - Cluster-based coloring
- **Edge annotations** showing connection strength as thickness and opacity
- **Backend integration** - loads data from FastAPI backend with auto-detection of data source

## Quick Start

```bash
# Terminal 1: Start the backend server (from project root)
cd ..
uv run uvicorn server.main:app --reload --port 3002

# Terminal 2: Start the GUI
npm install
npm run dev
```

Then open your browser to `http://localhost:5173`.

**The GUI always connects to the backend API**. The backend auto-detects your data source:
- With Neo4j env vars (`.env` in project root): loads from Neo4j
- Without Neo4j env vars: loads from files in `data/ego_graph/`

## Configuration

Create `gui/.env` to configure the backend URL:

```bash
# Backend API URL (update if running on different port)
VITE_API_URL=http://localhost:3002
```

See `.env.example` for template.

**To switch data sources**, configure environment variables in the project root `.env` file:
- Add Neo4j credentials to use Neo4j
- Remove Neo4j credentials to use file storage

The GUI doesn't need to know which data source is being used - the backend handles it transparently.

## Architecture

- **EgoGraphView**: Main visualization component that loads data and orchestrates the display
- **PersonNode**: Custom xyflow node component for displaying person information
- **egoGraphLoader**: Utilities for loading and parsing ego graph data from backend API
- **apiClient**: Backend API communication module
- **d3Layout**: D3 force-directed layout algorithm that positions nodes based on connection strength

## Data Flow

```
Backend (auto-detects):
  Neo4j OR Files → FastAPI endpoints
                        ↓
Frontend:
  GUI → API Client → Backend API
```

Single decision point: backend's environment variables determine data source.
