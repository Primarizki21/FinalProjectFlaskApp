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
import base64

app_routes = Blueprint('app_routes', __name__)

# Inisiasi Variabel
resize_dim = (224, 224)
threshold = 100
# model_trained = os.path.join(app_routes.root_path, 'static', 'vit_trained_wheel.pth')
model_trained = '/app/static/vit_trained_wheel.pth'
# model_trained = 'vit_trained_wheel.pth'
class_names = ["full_tire", "flat_tire", "no_tire"]
# HISTORY_FILE = os.path.join(app_routes.root_path, 'static','history.json')

# load Model
def load_model(model_trained):
    """
    Load model dengan arsitektu pre-trained ViT Model

    Args: model_trained adalah nama ViT model yang akan kita gunakan

    Return: Mengembalikan model yang sudah diload
    """
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=3,
        ignore_mismatched_sizes=True) # Inisiasi Model
    model.load_state_dict(torch.load(model_trained)) # Load model dari model_trained
    model.eval() # Set model dalam evaluation mode untuk prediksi
    return model

# Blur Detection
def is_blurry(image_path, threshold, resize_dim):
    """
    Mengecek apakah gambar tersebut blur dengan menghitung variance dari Laplacian
    Laplacian adalah operator matematika menghitung seberapa cepat nilai intensitas pada suatu piksel
    berubah dibandingkan dengan piksel di sekitarnya

    Args: image_path adalah path dari gambar dalam bentuk array
    threshold adalah batas nilai blur, set 100 untuk sekarang
    resize_dim adalah ukuran resize gambar yaitu 224x224

    Return: mengembalikan nilai variance dan boolean
    """  
    resized = cv2.resize(image_path, resize_dim) # Resize gambar
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) # Mengubah ke grayscale (0 sampai 255 untuk 8-bit)
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F) # Aplikasikan Laplacian
    
    variance = laplacian.var() # Menghitung variasi nilai Laplacian
    
    return variance, variance < threshold 

def preprocess_image(image_input, resize_dim):
    """
    Preprocess gambar dengan resizing dan konversi ke RGB format 

    Args: image_input adalah input imagenya
    resize_dim adalah ukuran resize gambar yaitu 224x224

    Return: gambar hasil preprocess
    """
    image = Image.open(image_input).convert("RGB") # Mengubah ke RGB format
    image = image.resize(resize_dim) # Resize gambar
    return image

def predict_image(saved_model, image_input, class_names, resize_dim):
    """
    Proses prediksi gambar 

    Args: image_input adalah input imagenya
    resize_dim adalah ukuran resize gambar yaitu 224x224

    Return: mengembalikan class prediksi dan probabilitasnya (kepercayaan model terhadap gambar)
    """
    model = load_model(saved_model) # Load model dengan fungsi load_model
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224") # Load feature extractor
    inputs = feature_extractor(images=image_input, return_tensors="pt") # Ekstrak feature pada gambar untuk input model
    with torch.no_grad(): # Nonaktifkan metode Gradient Computation
        outputs = model(**inputs) # Mendapatkan hasil output model

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1) # Menerapkan softmax pada nilai logits untuk mendapatkan probabilitasnya
    predicted_class_idx = probabilities.argmax().item() # Mendapatkan probabilitas tertinggi
    
    predicted_class_name = class_names[predicted_class_idx] if class_names else str(predicted_class_idx) # Mengubah index menjadi class
    return predicted_class_name, probabilities.squeeze().tolist()

def gettingServiceAccount():
    """ 
    Mendapatkan Service Account untuk akses GCS
    perlu untuk set GCS_SERVICE_ACCOUNT='path/to/key.json'
    """
    service_account = os.getenv("GCS_SERVICE_ACCOUNT") # Mengambil key dari environment
    if not service_account:
        raise Exception("GCS_SERVICE_ACCOUNT environment variable not set")
    service_account_info = json.loads(base64.b64decode(service_account)) # Decode key base64
    return service_account_info

# def upload_to_gcs(file, bucket_name, destination_blob_name, service_account, make_public):
def upload_to_gcs(file, bucket_name, destination_blob_name, make_public):
    """ 
    Mengupload file gambar ke bucket gcs
    perlu untuk set GOOGLE_APPLICATION_CREDENTIALS='path/to/key.json'

    Args: file adalah file yang ingin diupload
    bucket_name adalah nama bucket pada GCS
    destination_blob_name adalah format nama file yang akan disimpan di GCS
    make_public adalah boolean untuk apakah url gambar public atau tidak

    Return: mengembalikan url gambar
    """
    # client = storage.Client.from_service_account_json(service_account)
    # client = storage.Client.from_service_account_info(service_account)
    # Membuat client GCS dan bucketnya
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name) # Format nama filenya
    expiration = datetime.datetime.utcnow() + datetime.timedelta(days=7) # Link yang muncul akan expired dalam 7 hari
    blob.upload_from_file(file, content_type="image/jpeg") # Upload gambar
    public_url = blob.generate_signed_url(expiration=expiration, version="v4") # Membuat url
    return public_url

@app_routes.route('/predict', methods=['POST'])
def predict():
    """ 
    Proses menerima gambar dan membuat prediksi

    Return: mengembalikan hasil prediksi dalam bentuk JSON
    """
    try:
        if 'image' not in request.files: # Memeriksa apakah file ada pada request
            return jsonify({"error": "No file part"}), 400

        file = request.files['image'] # Mengambil file dari request
        
        if file.filename == '': # Memeriksa apakah gambarnya ada atau tidak
            return jsonify({"error": "No selected file"}), 400

        # Memeriksa apakah file png, jpg, atau jpeg
        if file and file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
            temp_image = io.BytesIO(file.read()) # Membaca file sebagai byte stream
            file.seek(0)
            file_hash = hashlib.md5(file.read()).hexdigest() # Membuat hash file
            file.seek(0)
            bucket_name = "finalproject-imagestorage"
            # service_account = "F:/kuliah/Semester 3/Algoritma Pemrograman II/Final Project OOOOOOO/Code/alpro-2-2024-13e3d5c82a81.json"
            service_account = gettingServiceAccount()
            destination_blob_name = f"images/{file_hash}.jpg"
            # image_url = upload_to_gcs(temp_image, bucket_name, destination_blob_name, service_account, make_public=True)
            image_url = upload_to_gcs(temp_image, bucket_name, destination_blob_name, make_public=True)
            
            try:
                temp_image.seek(0)
                image = Image.open(temp_image).convert("RGB") # Konversi ke RGB
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

@app_routes.route('/feedbacktous')
def feedback():
    return render_template('feedback.html')