from flask import Flask, request, jsonify
import torch
import requests
from flask import Flask, request, jsonify, send_file
import numpy as np
import cv2
from segment_anything_hq import sam_model_registry, SamPredictor
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS 
import json
# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['RESULT_FOLDER'] = 'results/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load model
sam_checkpoint = "./pretrained_checkpoint/sam_hq_vit_tiny.pth"
model_type = "vit_tiny"
device = 'cpu'
# device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
sam.eval()
predictor = SamPredictor(sam)

# Define the predict function (simplified version of the given script)
def predict(image_path, input_box=None, hq_token_only=False):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box,
        multimask_output=False,
        hq_token_only=hq_token_only, 
    )
    return masks, scores

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def upload_file():
    print('file_reached')
    print(request.files)
    # Check if a file is provided
    if 'croppedImage' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['croppedImage']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Here you can process other inputs like `input_box` from the request

    # For demonstration, we call predict without boxes or points
    masks, scores = predict(file_path, input_box=None, hq_token_only=False)
    
    # Process masks and scores, and save/send results
    result_filename = os.path.join(app.config['RESULT_FOLDER'], 'result.png')
    # Here you can call the `show_res` or `show_res_multi` function from the script to save the result image
    # For example:
    # show_res(masks, scores, None, None, None, result_filename, image)
    
    # For simplicity, we just return the scores
    return jsonify({'masks': masks.tolist(),
                    'scores': scores.tolist()})

@app.route('/test')
def hello_world():
    return 'Hello World'

if __name__ == '__main__':
    app.run(debug=True)
