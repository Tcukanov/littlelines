#!/bin/bash
# Start PNG to DST — backend (port 8000) + frontend (port 3000)
set -e
cd "$(dirname "$0")"

echo "Starting digitizing backend on http://localhost:8000 ..."
(cd backend && .venv/bin/uvicorn main:app --port 8000) &
BACKEND_PID=$!

cleanup() { kill $BACKEND_PID 2>/dev/null; }
trap cleanup EXIT

echo "Starting web app on http://localhost:3000 ..."
npm run dev --prefix frontend
