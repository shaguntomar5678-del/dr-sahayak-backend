import os
import io
import requests
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

app = Flask(__name__)
CORS(app)  # allows your website (a different domain) to call this API

MODEL_PATH = "dr_grading_resnet50.tflite"

# Your Hugging Face model file link
MODEL_URL = "https://huggingface.co/shagun5678/dr-sahayak-model/resolve/main/dr_grading_resnet50.tflite"

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
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model loaded and ready.")


def preprocess(image):
    """Matches ResNet50's expected preprocessing (RGB->BGR, ImageNet mean subtraction)."""
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32")
    arr = arr[..., ::-1]  # RGB -> BGR
    mean = np.array([103.939, 116.779, 123.68], dtype="float32")
    arr = arr - mean
    return np.expand_dims(arr, axis=0)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "DR-Sahayak API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Send it as form field 'image'."}), 400

    file = request.files["image"]
    img = Image.open(io.BytesIO(file.read()))
    arr = preprocess(img)

    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]['index'])[0]

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

