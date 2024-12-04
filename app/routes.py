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
# model_trained = "static/vit_trained_wheel.pth"
model_trained = os.path.join(app_routes.root_path, 'static', 'vit_trained_wheel.pth')
class_names = ["full_tire", "flat_tire", "no_tire"]
HISTORY_FILE = os.path.join(app_routes.root_path, 'static','history.json')
# UPLOAD_FOLDER = os.path.join(app_routes.root_path, 'static', 'uploads')
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
    # image = cv2.imread(image_path)
    #     raise FileNotFoundError(f"Image at path '{image_path}' not found.")
    
    resized = cv2.resize(image_path, resize_dim)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    variance = laplacian.var()
    
    return variance, variance < threshold

def preprocess_image(image_input, resize_dim):
    # feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    image = Image.open(image_input).convert("RGB")
    image = image.resize(resize_dim)
    # inputs = feature_extractor(images=image, return_tensors="pt")
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

# Load History
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    else:
        return []

# Save history
def save_history(history_data):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history_data, f)

@app_routes.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # image = Image.open(file.stream).convert("RGB")
        
        # image_cv = image.resize(resize_dim)
        # image_cv = preprocess_image(file.stream, resize_dim)

        image = Image.open(file.stream).convert("RGB")
        image_cv = np.array(image)

        variance, is_blurry_image = is_blurry(image_cv, threshold, resize_dim)

        if is_blurry_image:
            return jsonify({"error": "Image is blurry", "variance": variance}), 400
        else:
            image_preprocessed = preprocess_image(file.stream, resize_dim)
            predicted_class_name, probabilities = predict_image(model_trained, image_preprocessed, class_names, resize_dim)

            history_data = load_history()

            hasil_prediction = {
                "image_path": file.filename,
                "predicted_class": predicted_class_name,
                "probabilities": f"{round(max(probabilities) * 100, 2)}%",
                "variance": round(variance, 2),
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            history_data.append(hasil_prediction)
            save_history(history_data)

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

@app_routes.route('/hasilanalisis')
def hasilanalisis():
    return render_template('hasilanalisis.html')

@app_routes.route('/history')
def history():
    history_data = load_history()
    hasil_terbaru = history_data[-3:][::-1]
    hidden_count = len(history_data) - len(hasil_terbaru)
    return render_template('history.html', history=hasil_terbaru, hidden_count=hidden_count)

@app_routes.route('/tes')
def tesaja():
    return render_template('tesupload.html')

