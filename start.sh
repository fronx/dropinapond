#!/bin/bash
# Start both backend and frontend for Drop in a Pond

set -e

echo "Starting Drop in a Pond..."
echo ""

# Check if Neo4j needs to be resumed and start backend
echo "1. Starting backend API server..."
uv run python scripts/start_server.py &
BACKEND_PID=$!

# Give backend a moment to start
sleep 2

# Start frontend
echo ""
echo "2. Starting frontend..."
cd gui
npm run dev &
FRONTEND_PID=$!
cd ..

# Wait a moment for frontend to start
sleep 3

echo ""
echo "================================================================"
echo "  Drop in a Pond is running!"
echo "================================================================"
echo ""
echo "  Open in your browser:"
echo "  → http://localhost:5173/"
echo ""
echo "  Backend API: http://localhost:3002"
echo ""
echo "  Press Ctrl+C to stop both servers"
echo "================================================================"
echo ""

# Wait for either process to exit
wait -n

# Kill both processes when one exits
kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
