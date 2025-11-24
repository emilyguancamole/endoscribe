#!/bin/bash
# Build and run the EndoScribe web app

set -e

echo "Building React frontend..."
cd web_app/frontend
npm run build
cd ../..

echo ""
echo "Starting EndoScribe server..."
echo "Frontend: React (built)"
echo "Backend: http://localhost:8001"
echo ""

uvicorn web_app.server:app --host localhost --port 8001 --reload
