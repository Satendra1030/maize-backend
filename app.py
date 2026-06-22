"""
Maize Leaf Disease Detection - Flask Backend (Optimized with TFLite)
Major Project: Maize Leaf Disease Detection Using CNN
Pokhara University, 2026

This server exposes a single REST endpoint, POST /predict, which:
  1. Accepts a multipart image file from the Flutter app
  2. Preprocesses it (resize, normalize) to match MobileNetV2 input
  3. Runs inference using the lightweight TFLite model (model/maize_model.tflite)
  4. Looks up the predicted disease in the recommendation knowledge base
  5. Returns a structured JSON response (label, confidence, severity, treatment)
"""

import os
import io
import logging
import importlib

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image

# 1. SETUP LOGGING FIRST SO THE IMPORT FALLBACK CAN USE IT IMMEDIATELY
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. RUN IMPORT FALLBACK ENGINE
# Prefer lightweight runtime options; fall back to TensorFlow's core lite engine if completely unavailable
tflite = None
for module_name in ["ai_edge_litert.interpreter", "tflite_runtime.interpreter"]:
    try:
        tflite = importlib.import_module(module_name)
        logger.info("Successfully loaded TFLite engine via: %s", module_name)
        break
    except ImportError:
        continue

if tflite is None:
    try:
        import tensorflow as tf
        tflite = tf.lite
        logger.info("Runtime standalone wheels not found; falling back to core tensorflow.lite engine.")
    except Exception as e:
        logger.critical("Fatal: No TFLite execution layer found (ai-edge-litert, tflite_runtime, or tensorflow). Error: %s", str(e))
        raise

from utils.recommendations import get_recommendation
from utils.preprocessing import preprocess_image, ALLOWED_EXTENSIONS

# --------------------------------------------------------------------------
# Configuration & Global Variables
# --------------------------------------------------------------------------
# UPDATED: Explicitly pointing to your newly converted optimized file inside the model directory
MODEL_PATH = os.environ.get("MODEL_PATH", "model/final_model.tflite")
IMG_SIZE = (224, 224)  # MobileNetV2 expected input size, per proposal section 3.7

# Roadmap of all 10 target classes outlined for the final project proposal
ALL_PROJECT_CLASSES = [
    "Common Rust",
    "Gray Leaf Spot",
    "Healthy",
    "Northern Leaf Blight",
    "Southern Leaf Blight",
    "Southern Rust",
    "Banded Leaf and Sheath Blight",
    "Maize Streak Virus",
    "Brown Spot",
    "Downy Mildew",
]

# This list will be dynamically assigned at startup based on the actual model file output layers
CLASS_NAMES = []

# --------------------------------------------------------------------------
# App Setup & Dynamic TFLite Model Loading
# --------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the Flutter app (section 3.11)

# Load the TFLite interpreter at startup to minimize API latency
logger.info("Loading TFLite interpreter from %s ...", MODEL_PATH)
try:
    # Initialize the TFLite runtime and allocate tensor memory maps
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    # Extract internal metadata structural layers
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Read output shape array structure dynamically: shape index is usually [1, num_classes]
    num_model_outputs = output_details[0]['shape'][-1]
    logger.info("Detected %d output classes from TFLite structural architecture.", num_model_outputs)
    
    if num_model_outputs == 4:
        logger.info("Configuring for 4-class development dataset.")
        # Re-mapped order to align with Keras internal folder indices
        CLASS_NAMES = ["Gray Leaf Spot", "Common Rust", "Northern Leaf Blight", "Healthy"]
    else:
        logger.info("Configuring workspace for full 10-class dataset deployment.")
        CLASS_NAMES = ALL_PROJECT_CLASSES

except Exception as e:
    logger.critical("Fatal initialization failure: Could not load TFLite model at %s. Error: %s", MODEL_PATH, str(e))
    raise e


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def health_check():
    """Simple health check endpoint, also useful for production monitoring platforms."""
    return jsonify({
        "status": "ok",
        "message": "Maize Leaf Disease Detection API (TFLite Optimized) is running.",
        "active_classes_count": len(CLASS_NAMES),
        "active_classes": CLASS_NAMES
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
        return jsonify({"error": f"Unsupported file type. Allowed extensions: {ALLOWED_EXTENSIONS}"}), 400

    try:
        # 2. Read and preprocess image
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        processed = preprocess_image(img, target_size=IMG_SIZE)  # shape (1, 224, 224, 3)

        # Ensure array data type matches the model expected precision (float32)
        input_data = np.array(processed, dtype=np.float32)

        # 3. Run TFLite inference
        # Inject the image array into the model's input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Fire the execution runtime loop
        interpreter.invoke()
        
        # Pull output results array matrix and flatten it
        predictions = interpreter.get_tensor(output_details[0]['index'])[0] # shape (num_classes,)
        
        predicted_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_index])

        if predicted_index >= len(CLASS_NAMES):
            logger.error("Predicted index %d out of range for current CLASS_NAMES map", predicted_index)
            return jsonify({"error": "Model output dimensions mismatch backend configuration."}), 500

        disease_label = CLASS_NAMES[predicted_index]

        # 4. Look up recommendation metadata (description, treatment, prevention, severity)
        recommendation = get_recommendation(disease_label)

        # 5. Build structured response payload
        response = {
            "disease": disease_label,
            "confidence": round(confidence * 100, 2),  # Convert to percentage float
            "is_healthy": disease_label == "Healthy",
            "severity": recommendation.get("severity", "Unknown"),
            "description": recommendation.get("description", "No information available."),
            "treatment": recommendation.get("treatment", "No treatment guidelines found."),
            "prevention": recommendation.get("prevention", "No prevention metrics found."),
            "all_class_probabilities": {
                CLASS_NAMES[i]: round(float(p) * 100, 2) for i, p in enumerate(predictions)
            },
        }

        return jsonify(response), 200

    except Exception as exc:
        logger.exception("Inference workflow hit an exception loop:")
        return jsonify({"error": f"Internal process error during prediction execution: {str(exc)}"}), 500


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# --------------------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)