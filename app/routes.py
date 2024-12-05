from flask import Flask, render_template, request, jsonify, Blueprint
from PIL import Image
import io
import torch
import os
import json
from datetime import datetime
import numpy as np
from transformers import ViTFeatureExtractor, ViTForImageClassification
import cv2

app_routes = Blueprint('app_routes', __name__)

resize_dim = (224, 224)
threshold = 100
model_trained = os.path.join(app_routes.root_path, 'static', 'vit_trained_wheel.pth')
class_names = ["full_tire", "flat_tire", "no_tire"]
HISTORY_FILE = os.path.join(app_routes.root_path, 'static','history.json')

# load Model
def load_model(model_trained):
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=3,
        ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(model_trained))
    model.eval()
    return model

# Blur Detection
def is_blurry(image_path, threshold, resize_dim):    
    resized = cv2.resize(image_path, resize_dim)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    variance = laplacian.var()
    
    return variance, variance < threshold

def preprocess_image(image_input, resize_dim):
    image = Image.open(image_input).convert("RGB")
    image = image.resize(resize_dim)
    return image

def predict_image(saved_model, image_input, class_names, resize_dim):
    model = load_model(saved_model)
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    inputs = feature_extractor(images=image_input, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class_idx = probabilities.argmax().item()
    
    predicted_class_name = class_names[predicted_class_idx] if class_names else str(predicted_class_idx)
    return predicted_class_name, probabilities.squeeze().tolist()

@app_routes.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
        image_cv = np.array(image)

        variance, is_blurry_image = is_blurry(image_cv, threshold, resize_dim)

        if is_blurry_image:
            return jsonify({"error": "Image is blurry", "variance": variance}), 400
        else:
            image_preprocessed = preprocess_image(file.stream, resize_dim)
            predicted_class_name, probabilities = predict_image(model_trained, image_preprocessed, class_names, resize_dim)
            response = {
                "predicted_class": predicted_class_name,
                "probabilities": probabilities,
                "variance": variance
            }
            return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app_routes.route('/')
def index():
    return render_template('index.html')

@app_routes.route('/upload')
def upload():
    return render_template('upload.html')

@app_routes.route('/history')
def history():
    return render_template('history.html')


