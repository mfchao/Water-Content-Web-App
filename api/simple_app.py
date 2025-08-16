from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({'message': 'Water Content API is running!'})

@app.route('/test')
def test():
    return jsonify({'message': 'Test endpoint working!'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mock prediction for testing
        return jsonify({
            'prediction': 12.5, 
            'message': 'Mock prediction - full model not loaded yet'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 