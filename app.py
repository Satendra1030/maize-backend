"""
Maize Leaf Disease Detection - Flask Backend (Optimized with Two-Stage TFLite Engine)
Major Project: Maize Leaf Disease Detection Using CNN
Pokhara University, 2026

This server exposes a single REST endpoint, POST /predict, which:
  1. Accepts a multipart image file from the Flutter app
  2. Runs Stage 1 Inference (Gatekeeper Model) to ensure it is a Maize Leaf
  3. If valid, runs Stage 2 Inference (Disease Classifier TFLite Model)
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
# Main disease classifier model
MODEL_PATH = os.environ.get("MODEL_PATH", "model/final_model.tflite")
# Gatekeeper validation binary model path (Maize vs Not-Maize)
GATEKEEPER_PATH = os.environ.get("GATEKEEPER_PATH", "model/gatekeeper_model.tflite")

IMG_SIZE = (224, 224)  # Global resolution layer size expected by both runtimes

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
    "Success"
]

# Dynamically assigned at startup based on main model file output dimensions
CLASS_NAMES = []

# --------------------------------------------------------------------------
# App Setup & Dynamic TFLite Model Loading
# --------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the Flutter app (section 3.11)

# Global holders for interpreters and input/output layer maps
interpreter = None
input_details = None
output_details = None

gate_interpreter = None
gate_input_details = None
gate_output_details = None

logger.info("Initializing multi-stage TFLite validation pipelines...")
try:
    # ---- STAGE 1: LOAD GATEKEEPER MODEL INTERPRETER ----
    logger.info("Loading Stage 1 Gatekeeper model from %s ...", GATEKEEPER_PATH)
    gate_interpreter = tflite.Interpreter(model_path=GATEKEEPER_PATH)
    gate_interpreter.allocate_tensors()
    gate_input_details = gate_interpreter.get_input_details()
    gate_output_details = gate_interpreter.get_output_details()

    # ---- STAGE 2: LOAD MAIN DISEASE CLASSIFIER MODEL ----
    logger.info("Loading Stage 2 Disease model from %s ...", MODEL_PATH)
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Read output shape structure dynamically to automatically scale active array configuration
    num_model_outputs = output_details[0]['shape'][-1]
    logger.info("Detected %d output classes from classification TFLite layers.", num_model_outputs)
    
    if num_model_outputs == 4:
        logger.info("Configuring layout map for 4-class development dataset.")
        CLASS_NAMES = ["Blight", "Common Rust", "Gray_Leaf_Spot", "Healthy"]
    else:
        logger.info("Configuring workspace for full 10-class dataset deployment.")
        CLASS_NAMES = ALL_PROJECT_CLASSES

except Exception as e:
    logger.critical("Fatal initialization failure: Could not map TFLite runtimes at startup. Error: %s", str(e))
    raise e


# --------------------------------------------------------------------------
# Helper Function for Crop Verification
# --------------------------------------------------------------------------
def verify_is_maize(img) -> bool:
    """
    Passes preprocessed image array into the gatekeeper model runtime.
    Returns True if the model classifies the input object as a Maize leaf,
    and False if it matches background elements or non-maize species.
    """
    try:
        # 1. Reuse global image preprocessor module scaled to 224x224
        processed = preprocess_image(img, target_size=IMG_SIZE)
        input_data = np.array(processed, dtype=np.float32)

        # 2. Fire inference execution loop on the Gatekeeper engine
        gate_interpreter.set_tensor(gate_input_details[0]['index'], input_data)
        gate_interpreter.invoke()

        # 3. Pull predictions: Teachable Machine structure saves labels sequentially
        # Index 0 = Maize, Index 1 = Not_Maize
        gate_predictions = gate_interpreter.get_tensor(gate_output_details[0]['index'])[0]
        
        logger.info("!!! GATEKEEPER ANALYSIS RAW PROBABILITIES -> Maize: %.4f, Not-Maize: %.4f", 
                    gate_predictions[0], gate_predictions[1])

        # Return True only if Maize probability is strictly dominant over alternative classes
        return gate_predictions[0] > gate_predictions[1]
    except Exception as err:
        logger.error("Exception tripped inside Gatekeeper pipeline layer: %s", str(err))
        # Fallback security shield to prevent system bypass on runtime processing error
        return False


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def health_check():
    """Simple health check endpoint, also useful for production monitoring platforms."""
    return jsonify({
        "status": "ok",
        "message": "Maize Leaf Disease Detection API (Dual-Stage TFLite Validation Pipeline) is running.",
        "active_classes_count": len(CLASS_NAMES),
        "active_classes": CLASS_NAMES
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a multipart/form-data image upload under the key 'image',
    validates that the sample belongs to a maize plant, and handles prediction.
    """
    # 1. Validate request structure boundary blocks
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image file provided. Use form key 'image'."
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "Empty filename."
        }), 400

    if not _allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Unsupported file type. Allowed extensions: {ALLOWED_EXTENSIONS}"
        }), 400

    try:
        # 2. Read input binary stream into PIL framework
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # ==================================================================
        # STAGE 1 PIPELINE: CRITICAL GATEKEEPER VALIDATION CHECK
        # ==================================================================
        if not verify_is_maize(img):
            logger.warning("Rejected upload payload: Target image is classified as NON-MAIZE.")
            return jsonify({
                "success": False,                       # Fixed to match requested response blueprint
                "status": "error",
                "error_type": "INVALID_LEAF_TYPE",
                "message": "Please upload a maize leaf image only."
            }), 400

        # ==================================================================
        # STAGE 2 PIPELINE: STANDARDIZED MAIZE DISEASE CLASSIFICATION
        # ==================================================================
        processed = preprocess_image(img, target_size=IMG_SIZE)
        input_data = np.array(processed, dtype=np.float32)

        # Inject array matrices and run core diagnostic inference execution loop
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Pull output structural results array matrix
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        logger.info("!!! DIAGNOSTIC MODEL OUTPUT PROBABILITIES: %s", predictions.tolist())

        predicted_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_index])

        if predicted_index >= len(CLASS_NAMES):
            logger.error("Predicted index %d out of range for active CLASS_NAMES structural arrays", predicted_index)
            return jsonify({
                "success": False,
                "error": "Model output dimensions mismatch backend configuration layout maps."
            }), 500

        disease_label = CLASS_NAMES[predicted_index]

        # 3. Look up recommendation metadata metrics mapping arrays
        recommendation = get_recommendation(disease_label)

        # 4. Build complete structured deployment payload
        response = {
            "success": True,                            # Explicit success indicator flag
            "disease": disease_label,
            "confidence": round(confidence * 100, 2),
            "is_healthy": disease_label == "Healthy",
            "severity": recommendation.get("severity", "Unknown"),
            "description": recommendation.get("description", "No detailed information available."),
            "treatment": recommendation.get("treatment", "No application guidelines found."),
            "prevention": recommendation.get("prevention", "No preventive metrics found."),
            "all_class_probabilities": {
                CLASS_NAMES[i]: round(float(p) * 100, 2) for i, p in enumerate(predictions)
            },
        }

        return jsonify(response), 200

    except Exception as exc:
        logger.exception("Inference workflow hit an exception loop:")
        return jsonify({
            "success": False,
            "error": f"Internal process error during prediction execution: {str(exc)}"
        }), 500


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# --------------------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)