import json
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import base64
import io
from http.server import BaseHTTPRequestHandler
import os

class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.feature_extractor = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.feature_extractor.fc = nn.Linear(512, 128)
        self.regression_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.regression_head(x)
        return x

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

def predict_water_content(image_data):
    try:
        # Decode base64 image data
        image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
        
        # Open the image
        image = Image.open(io.BytesIO(image_bytes))

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

        return {'prediction': float(output.item())}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {'error': str(e)}

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Set CORS headers
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

        try:
            # Get the content length
            content_length = int(self.headers['Content-Length'])
            
            # Read the request body
            post_data = self.rfile.read(content_length)
            
            # Parse the JSON data
            request_data = json.loads(post_data.decode('utf-8'))
            
            # Extract image data (assuming it's base64 encoded)
            image_data = request_data.get('image')
            
            if not image_data:
                response = {'error': 'No image data provided'}
            else:
                # Make prediction
                response = predict_water_content(image_data)
                
        except Exception as e:
            response = {'error': f'Request processing error: {str(e)}'}
        
        # Send the response
        self.wfile.write(json.dumps(response).encode())
    
    def do_OPTIONS(self):
        # Handle preflight CORS request
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(b'')

if __name__ == '__main__':
    # This is for local testing only
    from http.server import HTTPServer
    server = HTTPServer(('localhost', 8000), handler)
    print('Server running on localhost:8000')
    server.serve_forever() 