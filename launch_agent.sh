#!/bin/bash
cd "/Users/cj/Dropbox/Windsurf/CascadeProjects/Sports Props Betting/backend"
# Use the local python env instead of poetry since it was not found
source .venv/bin/activate 2>/dev/null || true
python3 -m uvicorn app.main:app --reload --port 8000 &
PID=$!
echo "Agent API started with PID $PID"
sleep 5

echo "Starting autonomous trading loop (LIVE MONEY)..."
curl -X POST http://localhost:8000/api/kalshi/agent/start \
  -H "Content-Type: application/json" \
  -d '{"interval_seconds": 60, "paper_mode": false}'

echo ""
echo "Agent loop started. Keep this terminal open."
wait $PID
