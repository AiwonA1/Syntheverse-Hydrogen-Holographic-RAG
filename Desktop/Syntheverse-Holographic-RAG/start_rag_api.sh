#!/bin/bash

# Start RAG API Server
# This script starts the RAG API server with the web UI

echo "=========================================="
echo "Starting Syntheverse RAG API Server"
echo "=========================================="
echo ""
echo "API will be available at:"
echo "  - API: http://localhost:8000"
echo "  - UI:  http://localhost:8000"
echo "  - Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

python rag_api.py

