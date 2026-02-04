#!/bin/bash

echo "🏥 Hospital Operations Analytics Dashboard - Setup"
echo "============================================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install fastapi uvicorn sqlalchemy psycopg2-binary pandas numpy python-multipart jinja2

# Try Prophet (optional)
echo "Attempting Prophet installation (optional)..."
pip install prophet || echo "⚠️ Prophet not installed, using simple forecasting"

echo ""
echo "============================================================"
echo "✅ Setup complete!"
echo ""
echo "🚀 To run the application:"
echo "   1. Ensure virtual environment is active: source venv/bin/activate"
echo "   2. Run: python3 generate_sample_data.py"
echo "   3. Run: python3 main.py"
echo "   4. Open: http://localhost:8000"
