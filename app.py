"""
Maize Leaf Disease Detection - Flask Backend (Two-Stage TFLite Engine & Groq Chat Engine)
Major Project: Maize Leaf Disease Detection Using CNN
Pokhara University, 2026

Endpoints:
  1. POST /predict -> Accepts a leaf image, runs validation + classification TFLite layers,
                      and returns a diagnostic recommendation package.
  2. POST /chat    -> Processes natural language conversational questions about maize farming via Groq.
"""

import os
import io
import logging
import importlib
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from local .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================================================
# GROQ API CONFIGURATION
# ==========================================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Import Fallback Engine for TFLite Runtimes
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

# Initialize Groq Client
groq_client = None
if GROQ_API_KEY:
    try:
        groq_module = importlib.import_module("groq")
        Groq = getattr(groq_module, "Groq", None)
        if Groq is None:
            raise ImportError("Groq class not found in groq package")
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq SDK initialized successfully.")
    except ImportError:
        logger.warning("Groq SDK not installed or Groq class missing. Please run `pip install groq`.")
    except Exception as err:
        logger.error("Failed to initialize Groq client: %s", str(err))
else:
    logger.warning("GROQ_API_KEY missing in environment variables. Chat endpoint will be unavailable.")

from utils.recommendations import get_recommendation
from utils.preprocessing import preprocess_image, ALLOWED_EXTENSIONS

# --------------------------------------------------------------------------
# Configuration & Global Variables
# --------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "model/final_model.tflite")
GATEKEEPER_PATH = os.getenv("GATEKEEPER_PATH", "model/gatekeeper_model.tflite")

IMG_SIZE = (224, 224) 

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
    "Downy Mildew"
]

CLASS_NAMES = []

# --------------------------------------------------------------------------
# App Setup & Dynamic TFLite Model Loading
# --------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)  

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
# Helper Functions
# --------------------------------------------------------------------------
def verify_is_maize(img: Image.Image) -> bool:
    """
    Passes preprocessed image array into the gatekeeper model runtime.
    Index 0 = Maize, Index 1 = Not_Maize
    """
    try:
        processed = preprocess_image(img, target_size=IMG_SIZE)
        input_data = np.array(processed, dtype=np.float32)

        gate_interpreter.set_tensor(gate_input_details[0]['index'], input_data)
        gate_interpreter.invoke()

        gate_predictions = gate_interpreter.get_tensor(gate_output_details[0]['index'])[0]
        
        logger.info("GATEKEEPER ANALYSIS -> Maize: %.4f, Not-Maize: %.4f", 
                    gate_predictions[0], gate_predictions[1])

        return bool(gate_predictions[0] > gate_predictions[1])
    except Exception as err:
        logger.error("Exception tripped inside Gatekeeper pipeline layer: %s", str(err))
        return False


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        "status": "ok",
        "message": "Maize Leaf Disease Detection & Chatbot API is running.",
        "active_classes_count": len(CLASS_NAMES),
        "active_classes": CLASS_NAMES
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts an image, verifies plant type via Gatekeeper model, 
    and returns diagnostic predictions with agronomic recommendations.
    """
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error_type": "MISSING_PAYLOAD",
            "message": "No image file provided. Please use form key 'image'."
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "success": False,
            "error_type": "EMPTY_FILENAME",
            "message": "Empty filename detected."
        }), 400

    if not _allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error_type": "UNSUPPORTED_EXTENSION",
            "message": f"Unsupported file type. Allowed extensions: {ALLOWED_EXTENSIONS}"
        }), 400

    try:
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # STAGE 1 PIPELINE: GATEKEEPER VALIDATION CHECK
        if not verify_is_maize(img):
            logger.warning("Rejected upload payload: Target image classified as NON-MAIZE.")
            return jsonify({
                "success": False,
                "error_type": "INVALID_LEAF_TYPE",
                "message": "Invalid crop image detected. Please upload a valid maize leaf image."
            }), 400

        # STAGE 2 PIPELINE: MAIZE DISEASE CLASSIFICATION
        processed = preprocess_image(img, target_size=IMG_SIZE)
        input_data = np.array(processed, dtype=np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        logger.info("DIAGNOSTIC MODEL PROBABILITIES: %s", predictions.tolist())

        predicted_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_index])

        if predicted_index >= len(CLASS_NAMES):
            logger.error("Predicted index %d out of bounds for CLASS_NAMES array", predicted_index)
            return jsonify({
                "success": False,
                "error_type": "DIMENSION_MISMATCH",
                "message": "Model output dimensions mismatch backend layout maps."
            }), 500

        disease_label = CLASS_NAMES[predicted_index]
        recommendation = get_recommendation(disease_label)

        response = {
            "success": True,
            "disease": disease_label,
            "confidence": round(confidence * 100, 2),
            "is_healthy": disease_label.lower() == "healthy",
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
            "error_type": "INTERNAL_SERVER_ERROR",
            "message": f"Internal process error during prediction execution: {str(exc)}"
        }), 500


@app.route("/chat", methods=["POST"])
def chat():
    """
    Accepts conversational queries and routes them to Groq AI (Llama 3.1 8B Instant).
    Fast, reliable, and strictly guarded for Nepalese agricultural queries.
    """
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({
                "success": False, 
                "error_type": "INVALID_JSON",
                "message": "Missing parameters. Provide a valid 'message' string."
            }), 400

        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({
                "success": False, 
                "error_type": "EMPTY_MESSAGE",
                "message": "Message body cannot be empty."
            }), 400

        if groq_client is None:
            return jsonify({
                "success": False,
                "error_type": "GROQ_NOT_CONFIGURED",
                "message": "Groq client is uninitialized. Ensure GROQ_API_KEY is set in .env file."
            }), 500

        logger.info("Forwarding query sequence to Groq Engine: %s", user_message)

        system_instruction = (
            "You are an expert agronomy AI assistant specialized strictly in Nepalese maize cultivation, "
            "soil health, crop protection, and local pest management (e.g., Fall Armyworm, Stem Borers, Common Rust, GLS). "
            "Your target users are farmers and engineering research students in Nepal. "
            "Answer agriculture-related questions accurately, concisely, and practically. "
            "If a query is completely unrelated to agriculture or plant care, politely decline and remind "
            "the user that you are exclusively optimized to assist with maize crop cultivation."
        )

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_message}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=500
        )

        bot_response = chat_completion.choices[0].message.content

        return jsonify({
            "success": True,
            "response": bot_response
        }), 200

    except Exception as chat_err:
        logger.exception("Chat routing matrix caught an unhandled exception:")
        return jsonify({
            "success": False,
            "error_type": "CHAT_SERVER_ERROR",
            "message": f"Internal Chat Server Error: {str(chat_err)}"
        }), 500


# --------------------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)