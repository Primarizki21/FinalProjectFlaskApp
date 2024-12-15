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
from google.cloud import storage
import traceback
import hashlib
import datetime

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

def upload_to_gcs(file, bucket_name, destination_blob_name, service_account, make_public):
    client = storage.Client.from_service_account_json(service_account)
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    expiration = datetime.datetime.utcnow() + datetime.timedelta(days=7)
    blob.upload_from_file(file, content_type="image/jpeg")
    public_url = blob.generate_signed_url(expiration=expiration, version="v4")
    return public_url
    # public_url = f"https://storage.googleapis.com/{bucket_name}/{destination_blob_name}"
    # if make_public:
    #     blob.make_public()

    

@app_routes.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
            temp_image = io.BytesIO(file.read())
            file.seek(0)
            file_hash = hashlib.md5(file.read()).hexdigest()
            file.seek(0)
            bucket_name = "finalproject-imagestorage"
            service_account = "F:/kuliah/Semester 3/Algoritma Pemrograman II/Final Project OOOOOOO/Code/alpro-2-2024-13e3d5c82a81.json"
            destination_blob_name = f"images/{file_hash}.jpg"
            image_url = upload_to_gcs(temp_image, bucket_name, destination_blob_name, service_account, make_public=True)
            
            try:
                temp_image.seek(0)
                image = Image.open(temp_image).convert("RGB")
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
                        "variance": variance,
                        "image_url":image_url
                    }
                    return jsonify(response)
                
            except Exception as e:
                return jsonify({"error": str(e)}), 500
    except Exception as e:
        print("Server Error:", e)  # Log to terminal
        print(traceback.format_exc())  # Full traceback log
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app_routes.route('/')
def index():
    return render_template('index.html')

@app_routes.route('/upload')
def upload():
    return render_template('upload.html')

@app_routes.route('/history')
def history():
    return render_template('history.html')

@app_routes.route('/feedback')
def feedbacktouser():
    return render_template('feedbacktouser.html')
