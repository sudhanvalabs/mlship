#!/bin/bash
# Run all tests for ShipML

set -e

echo "🧪 Running ShipML Integration Tests"
echo "===================================="
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not activated. Activating .venv..."
    source .venv/bin/activate 2>/dev/null || {
        echo "❌ No virtual environment found at .venv/"
        echo "   Run: uv venv && source .venv/bin/activate"
        exit 1
    }
fi

# Install test dependencies if needed
echo "📦 Installing dependencies..."
uv pip install -e ".[dev,all]" --quiet

echo ""
echo "🔬 Running tests..."
echo ""

# Run all tests with coverage
pytest tests/ -v --cov=shipml --cov-report=term-missing

echo ""
echo "✅ All tests completed!"
