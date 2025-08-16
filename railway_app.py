#!/usr/bin/env python3
"""
Railway-specific entry point for the Water Content App
"""

import os
import sys

# Add the api directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))

# Import and run the Flask app
from simple_app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 