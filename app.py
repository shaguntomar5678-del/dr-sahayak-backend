import os
import io
import requests
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)  # allows your website (a different domain) to call this API

MODEL_PATH = "dr_grading_resnet50.keras"

# IMPORTANT: replace YOUR-USERNAME and YOUR-MODEL-REPO below with your own,
# after you upload the model to Hugging Face in Stage 2.
MODEL_URL = "https://huggingface.co/shagun5678/dr-sahayak-model/resolve/main/dr_grading_resnet50.keras"

LABELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
IMG_SIZE = 224


def download_model():
    """Downloads the model file once, the first time the server starts."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading model (only happens once)...")
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Model downloaded.")


download_model()
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded and ready.")


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "DR-Sahayak API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Send it as form field 'image'."}), 400

    file = request.files["image"]
    img = Image.open(io.BytesIO(file.read())).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32")
    arr = tf.keras.applications.resnet50.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    probs = model.predict(arr)[0]
    level = int(np.argmax(probs))
    referable = level >= 2

    return jsonify({
        "level": level,
        "label": LABELS[level],
        "referable": referable,
        "confidence": float(probs[level]),
        "probabilities": {LABELS[i]: float(probs[i]) for i in range(5)},
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
