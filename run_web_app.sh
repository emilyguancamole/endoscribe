#!/bin/bash

# EndoScribe Web Application Startup Script

echo "Starting EndoScribe Web Application..."
echo "======================================="
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found!"
    echo "Please create a .env file with your API keys:"
    echo "  ANTHROPIC_API_KEY=your_key_here"
    echo "  OPENAI_API_KEY=your_key_here"
    echo "  HF_TOKEN=your_token_here"
    echo ""
fi

# Start the server
echo "Starting server on http://localhost:8000"
echo ""
python web_app/server.py

# Alternative using uvicorn directly:
# uvicorn web_app.server:app --host 0.0.0.0 --port 8000 --reload
