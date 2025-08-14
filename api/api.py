from flask import Flask, request, jsonify
import torch
from PIL import Image
from api.regression import RegressionModel
from flask_cors import CORS
from flask_cors import cross_origin
from torchvision import transforms



app = Flask(__name__)

# Load the model
device = torch.device('cpu')  # Use CPU for inference
model = RegressionModel()  # Create a new instance of the model
model.load_state_dict(torch.load('./model.pth', map_location=device))  # Load the state dictionary
model.eval()  # Set the model to evaluation mode



# Define a preprocessing transform
transform = transforms.Compose([
    transforms.Resize(256),  # Resize the image to 256x256
    transforms.CenterCrop(224),  # Crop the image to 224x224
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the pixel values
])



@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        image_file = request.files['image']

        # Open the image
        image = Image.open(image_file)

        # Remove alpha channel
        image = image.convert('RGB')

        # Preprocess the image
        image = transform(image)

        # Add a batch dimension
        image = image.unsqueeze(0)

        # Move the image to the device (GPU or CPU)
        image = image.to(device)

        # Make a prediction using the model
        output = model(image)

        # Return the prediction as a JSON response
        return jsonify({'prediction': output.item()})
    except Exception as e:
        print(f"Error: {str(e)}")  # Print the error message to the console
        return jsonify({'error': str(e)}), 500


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    app.run(debug=True)