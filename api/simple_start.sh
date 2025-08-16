#!/bin/bash

echo "Simple Python detection..."

# Just try the most common commands
if command -v python3 &> /dev/null; then
    echo "Using python3"
    python3 simple_app.py
elif command -v python &> /dev/null; then
    echo "Using python"
    python simple_app.py
else
    echo "No Python found. Trying to install..."
    # This might trigger Railway to install Python
    exit 1
fi 