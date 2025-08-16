#!/bin/bash

# Try different Python commands
if command -v python3.9 &> /dev/null; then
    echo "Using python3.9"
    python3.9 simple_app.py
elif command -v python3 &> /dev/null; then
    echo "Using python3"
    python3 simple_app.py
elif command -v python &> /dev/null; then
    echo "Using python"
    python simple_app.py
else
    echo "Python not found. Available commands:"
    ls /usr/bin/python*
    exit 1
fi 