from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from flask_cors import CORS
from tensorflow import keras
from keras_preprocessing.image import load_img, img_to_array

import os

app = Flask(__name__)
CORS(app)
# Get the current directory
current_dir = os.path.dirname(__file__)

# Construct the full path to the model file
model_path = os.path.join(current_dir, './model.h5')
# Check if the model file exists
if not os.path.exists(model_path):
    print(f"Model file not found at: {model_path}")
else:
    # Load the model
    model = keras.models.load_model(model_path)

# Define a preprocessing function
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Resize the image to 224x224
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add a batch dimension
    img = img / 255.0  # Normalize the pixel values
    return img

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        image_file = request.files['image']

        # Save the image temporarily
        temp_image_path = 'temp_image.jpg'
        image_file.save(temp_image_path)

        # Preprocess the image
        image = preprocess_image(temp_image_path)

        # Make a prediction using the model
        output = model.predict(image)

        # Remove the temporary image
        os.remove(temp_image_path)

        # Return the prediction as a JSON response
        return jsonify({'prediction': output[0][0].tolist()})
    except Exception as e:
        print(f"Error: {str(e)}")  # Print the error message to the console
        return jsonify({'error': str(e)}), 500

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.errorhandler(404)
def not_found(e):
    return app.send_static_file('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)