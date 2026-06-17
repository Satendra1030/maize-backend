"""
Maize Leaf Disease Detection - Flask Backend
Major Project: Maize Leaf Disease Detection Using CNN
Pokhara University, 2026

This server exposes a single REST endpoint, POST /predict, which:
  1. Accepts a multipart image file from the Flutter app
  2. Preprocesses it (resize, normalize) to match MobileNetV2 input
  3. Runs inference using the trained CNN model (model/maize_model.h5)
  4. Looks up the predicted disease in the recommendation knowledge base
  5. Returns a structured JSON response (label, confidence, severity, treatment)
"""

import os
import io
import logging

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import tensorflow as tf

from utils.recommendations import get_recommendation
from utils.preprocessing import preprocess_image, ALLOWED_EXTENSIONS

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "model/maize_model.h5")
IMG_SIZE = (224, 224)  # MobileNetV2 expected input size, per proposal section 3.7

# Class order MUST exactly match the order used during training
# (i.e. the order Keras assigned indices to your training folders /
# train_generator.class_indices). Update this if your training order differs.
CLASS_NAMES = [
    "Healthy",
    "Common Rust",
    "Gray Leaf Spot",
    "Northern Leaf Blight",
    "Southern Leaf Blight",
    "Southern Rust",
    "Banded Leaf and Sheath Blight",
    "Maize Streak Virus",
    "Brown Spot",
    "Downy Mildew",
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# App setup
# --------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)  # allow cross-origin requests from the Flutter app (section 3.11)

# Load the model ONCE at server startup, not per-request, to minimize latency
logger.info("Loading model from %s ...", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
logger.info("Model loaded successfully.")


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def health_check():
    """Simple health check endpoint, also useful for Render's health checks."""
    return jsonify({
        "status": "ok",
        "message": "Maize Leaf Disease Detection API is running.",
        "num_classes": len(CLASS_NAMES),
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a multipart/form-data image upload under the key 'image'
    and returns disease prediction + recommendation as JSON.
    """
    # 1. Validate request
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use form key 'image'."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    if not _allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400

    try:
        # 2. Read and preprocess image
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        processed = preprocess_image(img, target_size=IMG_SIZE)  # shape (1, 224, 224, 3)

        # 3. Run inference
        predictions = model.predict(processed)[0]  # shape (num_classes,)
        predicted_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_index])

        if predicted_index >= len(CLASS_NAMES):
            logger.error("Predicted index %d out of range for CLASS_NAMES", predicted_index)
            return jsonify({"error": "Model output does not match configured class list."}), 500

        disease_label = CLASS_NAMES[predicted_index]

        # 4. Look up recommendation (description, treatment, prevention, severity)
        recommendation = get_recommendation(disease_label)

        # 5. Build structured response
        response = {
            "disease": disease_label,
            "confidence": round(confidence * 100, 2),  # as percentage
            "is_healthy": disease_label == "Healthy",
            "severity": recommendation["severity"],          # "Green" | "Yellow" | "Red"
            "description": recommendation["description"],
            "treatment": recommendation["treatment"],
            "prevention": recommendation["prevention"],
            "all_class_probabilities": {
                CLASS_NAMES[i]: round(float(p) * 100, 2) for i, p in enumerate(predictions)
            },
        }

        return jsonify(response), 200

    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        return jsonify({"error": f"Internal error during prediction: {str(exc)}"}), 500


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # host=0.0.0.0 required for deployment platforms like Render (section 3.11)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
