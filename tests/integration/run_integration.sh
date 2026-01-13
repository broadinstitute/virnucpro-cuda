#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Building Docker image..."
cd "$PROJECT_ROOT"
docker build -t virnucpro:test .

echo ""
echo "Running integration tests..."
cd "$SCRIPT_DIR"
pytest -v test_integration.py

echo ""
echo "Integration tests completed successfully!"
