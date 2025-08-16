#!/bin/bash

echo "Starting Python detection..."
echo "Current directory: $(pwd)"
echo "Available Python commands:"

# Try to find Python in various locations
PYTHON_CMD=""

# Method 1: Check PATH
if command -v python3.9 &> /dev/null; then
    PYTHON_CMD="python3.9"
    echo "Found python3.9 in PATH"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "Found python3 in PATH"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    echo "Found python in PATH"
fi

# Method 2: Check common locations
if [ -z "$PYTHON_CMD" ]; then
    echo "Checking common Python locations..."
    for py_path in "/usr/local/bin/python3" "/opt/python/bin/python3" "/usr/bin/python3" "/usr/local/bin/python" "/opt/python/bin/python"; do
        if [ -f "$py_path" ]; then
            PYTHON_CMD="$py_path"
            echo "Found Python at: $py_path"
            break
        fi
    done
fi

# Method 3: Use which command
if [ -z "$PYTHON_CMD" ]; then
    echo "Using 'which' to find Python..."
    PYTHON_CMD=$(which python3 2>/dev/null || which python 2>/dev/null)
    if [ -n "$PYTHON_CMD" ]; then
        echo "Found Python via 'which': $PYTHON_CMD"
    fi
fi

# Method 4: Check if we're in a virtual environment
if [ -z "$PYTHON_CMD" ] && [ -n "$VIRTUAL_ENV" ]; then
    echo "Virtual environment detected: $VIRTUAL_ENV"
    PYTHON_CMD="$VIRTUAL_ENV/bin/python"
    echo "Using virtual env Python: $PYTHON_CMD"
fi

# Final check and execution
if [ -n "$PYTHON_CMD" ]; then
    echo "Starting app with: $PYTHON_CMD"
    echo "Python version: $($PYTHON_CMD --version 2>&1)"
    $PYTHON_CMD simple_app.py
else
    echo "ERROR: No Python found!"
    echo "Available commands in PATH:"
    echo $PATH | tr ':' '\n'
    echo "Current environment:"
    env | grep -i python
    exit 1
fi 