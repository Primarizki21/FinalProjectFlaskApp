from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import torch
import os
import numpy as np
from transformers import ViTFeatureExtractor, ViTForImageClassification
from flask_cors import CORS
import cv2

app_routes = Blueprint('app_routes', __name__)

resize_dim = (224, 224)
threshold = 100
# model_trained = "static/vit_trained_wheel.pth"
model_trained = os.path.join(app.root_path, 'static', 'vit_trained_wheel.pth')
class_names = ["full_tire", "flat_tire", "no_tire"]

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
    # # Read the image
    # image = cv2.imread(image_path)
    # if image is None:
    #     raise FileNotFoundError(f"Image at path '{image_path}' not found.")
    
    # Convert to grayscale
    resized = cv2.resize(image_path, resize_dim)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Apply the Laplacian operator to detect edges
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Calculate the variance of the Laplacian
    variance = laplacian.var()
    
    # Determine if the image is blurry
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
            # inputs = preprocess_image(image_cv, resize_dim)

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

@app_routes.route("/")
def home():
    return '<h1>THIS API WORKS!!!!</h1>'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

    