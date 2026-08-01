"""
Maize Leaf Disease Detection - Flask Backend (Two-Stage TFLite Engine & Groq Chat Engine)
Major Project: Maize Leaf Disease Detection Using CNN
Pokhara University, 2026

Endpoints:
  1. POST /predict -> Accepts a leaf image, runs validation + classification TFLite layers,
                      applies a 50% confidence threshold, and returns symptoms, treatments,
                      and diagnostic recommendation packages.
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
# CONFIDENCE THRESHOLD CONFIGURATION
# ==========================================================================
CONFIDENCE_THRESHOLD = 0.50  # 50% minimum threshold for valid predictions

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
# SYMPTOMS AND DISEASE DATABASE
# --------------------------------------------------------------------------
SYMPTOMS_DATABASE = {
    "Blight": {
        "symptoms": [
            "Long, elliptical, cigar-shaped tan or grayish lesions on leaves.",
            "Lesions merge to cause large areas of dead tissue.",
            "Lower leaf infection spreads upward during wet conditions."
        ],
        "description": "Northern Corn Leaf Blight is a destructive fungal infection caused by Exserohilum turcicum.",
        "treatment": "Apply fungicides containing azoxystrobin or pyraclostrobin at early lesion onset.",
        "prevention": "Rotate crops with legumes or soybeans and plant disease-resistant maize hybrids."
    },
    "Northern Leaf Blight": {
        "symptoms": [
            "Elongated grayish-green to tan leaf spots (2-15 cm long).",
            "Dark sporangia giving lesions a dirty appearance in humid weather.",
            "Premature drying and foliage death."
        ],
        "description": "Fungal foliage infection prevalent during cool, humid mid-season growing periods.",
        "treatment": "Spray foliar fungicides like Mancozeb or propiconazole if lesion thresholds exceed 10%.",
        "prevention": "Incorporate infected crop residues post-harvest and maintain recommended plant spacing."
    },
    "Southern Leaf Blight": {
        "symptoms": [
            "Small, rectangular tan spots with reddish-brown borders.",
            "Pustules restricted by leaf veins.",
            "Widespread leaf scorching under warm and moist weather."
        ],
        "description": "Caused by Bipolaris maydis, thriving in warm, moist, high-temperature climates.",
        "treatment": "Use foliar sprays like Tebraconazole when disease appears early in the canopy.",
        "prevention": "Utilize resistant seed varieties and clear field stubble post-harvest."
    },
    "Common Rust": {
        "symptoms": [
            "Powdery cinnamon-brown pustules on both upper and lower leaf surfaces.",
            "Pustules turn dark black as plant matures.",
            "Yellowing around pustules followed by premature leaf death."
        ],
        "description": "Fungal leaf rust caused by Puccinia sorghi, favored by cool, moist conditions.",
        "treatment": "Apply copper-based or triazole fungicides early when pustules appear.",
        "prevention": "Plant resistant hybrids and avoid late-season sowing."
    },
    "Southern Rust": {
        "symptoms": [
            "Tiny, dense, bright orange to reddish-orange pustules predominantly on the upper leaf surface.",
            "Pustules break through the leaf surface cleanly.",
            "Rapid leaf wilting under hot, humid environments."
        ],
        "description": "Aggressive rust disease caused by Puccinia polysora, spreading fast in warm zones.",
        "treatment": "Prompt application of systemic triazole fungicides upon early detection.",
        "prevention": "Select southern rust-tolerant maize lines and monitor fields frequently."
    },
    "Gray Leaf Spot": {
        "symptoms": [
            "Narrow, rectangular tan-to-gray lesions parallel to leaf veins.",
            "Lesions turn opaque and fuse, blighting whole leaves.",
            "Extensive leaf destruction leading to lodging."
        ],
        "description": "Severe fungal disease caused by Cercospora zeae-maydis, common in minimum-tillage fields.",
        "treatment": "Spray strobilurin or triazole group fungicides during early silking.",
        "prevention": "Practice 2-year crop rotation and deep tillage of crop residue."
    },
    "Gray_Leaf_Spot": {
        "symptoms": [
            "Narrow, rectangular tan-to-gray lesions parallel to leaf veins.",
            "Lesions turn opaque and fuse, blighting whole leaves.",
            "Extensive leaf destruction leading to lodging."
        ],
        "description": "Severe fungal disease caused by Cercospora zeae-maydis, common in minimum-tillage fields.",
        "treatment": "Spray strobilurin or triazole group fungicides during early silking.",
        "prevention": "Practice 2-year crop rotation and deep tillage of crop residue."
    },
    "Banded Leaf and Sheath Blight": {
        "symptoms": [
            "Concentric tan/brown bands on leaf sheaths and leaves.",
            "White fungal cobweb-like mycelial growth on lower stems.",
            "Ear rots in severe stages."
        ],
        "description": "Soil-borne fungal disease caused by Rhizoctonia solani f. sp. sasakii.",
        "treatment": "Foliar application of Carbendazim or Validamycin solution near lower leaf sheaths.",
        "prevention": "Avoid waterlogging, maintain wide row distance, and remove infected bottom leaves."
    },
    "Maize Streak Virus": {
        "symptoms": [
            "Fine, translucent pale-yellow streaks running parallel to leaf veins.",
            "Stunted plant growth and broken streak patterns across leaves.",
            "Deformed ears or complete failure of cob development."
        ],
        "description": "Viral infection transmitted by leafhopper vector species (Cicadulina spp.).",
        "treatment": "No cure for viral infection; control leafhoppers using imidacloprid sprays.",
        "prevention": "Use virus-resistant seed cultivars and eliminate grassy weed hosts around field margins."
    },
    "Brown Spot": {
        "symptoms": [
            "Small, yellow-to-brown spots on leaf blades, sheaths, and stalks.",
            "Purplish-brown spots merging into large dark patches.",
            "Stalk breakage at the nodes."
        ],
        "description": "Physoderma brown spot caused by Physoderma maydis in warm, wet weather.",
        "treatment": "Foliar fungicide application if infection reaches upper canopy leaves prior to tasseling.",
        "prevention": "Improve field soil drainage and implement strict crop rotation."
    },
    "Downy Mildew": {
        "symptoms": [
            "Chlorotic yellow-green stripes along leaves.",
            "White, downy fungal growth on lower leaf surfaces in morning dew.",
            "Stunted growth and 'crazy top' tassel malformation."
        ],
        "description": "Systemic oomycete infection resulting from waterlogged conditions and high humidity.",
        "treatment": "Apply systemic metalaxyl-based fungicide spray.",
        "prevention": "Treat seeds with Metalaxyl-M prior to planting and ensure field drainage."
    },
    "Healthy": {
        "symptoms": [
            "Uniform green leaf color with no lesions, pustules, or chlorotic streaks.",
            "Intact structural leaf integrity."
        ],
        "description": "The leaf appears healthy with normal photosynthetic structure and zero visible disease marks.",
        "treatment": "No chemical treatment required.",
        "prevention": "Continue standard irrigation, nitrogen balance, and routine field monitoring."
    }
}

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
        "active_classes": CLASS_NAMES,
        "confidence_threshold": f"{CONFIDENCE_THRESHOLD * 100}%"
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts an image, verifies plant type via Gatekeeper model, 
    applies 50% confidence threshold, and returns diagnostic predictions 
    with symptoms, treatment, and recommendations.
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
                "message": "Invalid crop image detected. Please upload a clear maize leaf image."
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

        # STAGE 3 PIPELINE: CONFIDENCE THRESHOLD CHECK (50%)
        if confidence < CONFIDENCE_THRESHOLD:
            logger.warning("Low confidence prediction rejected: %.2f%% < %.2f%%", 
                           confidence * 100, CONFIDENCE_THRESHOLD * 100)
            return jsonify({
                "success": False,
                "error_type": "LOW_CONFIDENCE",
                "message": f"Uncertain prediction ({round(confidence * 100, 2)}%). Please provide a clearer, well-lit photo of the maize leaf.",
                "confidence": round(confidence * 100, 2),
                "threshold_required": round(CONFIDENCE_THRESHOLD * 100, 2)
            }), 422

        if predicted_index >= len(CLASS_NAMES):
            logger.error("Predicted index %d out of bounds for CLASS_NAMES array", predicted_index)
            return jsonify({
                "success": False,
                "error_type": "DIMENSION_MISMATCH",
                "message": "Model output dimensions mismatch backend layout maps."
            }), 500

        disease_label = CLASS_NAMES[predicted_index]
        
        # Fallback to utils recommendation or local symptom database
        recommendation = get_recommendation(disease_label)
        symptom_details = SYMPTOMS_DATABASE.get(disease_label, {
            "symptoms": ["No specific symptoms recorded for this class."],
            "description": recommendation.get("description", "No detailed information available."),
            "treatment": recommendation.get("treatment", "Consult local agricultural extension."),
            "prevention": recommendation.get("prevention", "Maintain crop rotation and health practices.")
        })

        response = {
            "success": True,
            "disease": disease_label,
            "confidence": round(confidence * 100, 2),
            "is_healthy": disease_label.lower() == "healthy",
            "severity": recommendation.get("severity", "Medium"),
            "description": symptom_details.get("description", recommendation.get("description")),
            "symptoms": symptom_details.get("symptoms", []),
            "treatment": symptom_details.get("treatment", recommendation.get("treatment")),
            "prevention": symptom_details.get("prevention", recommendation.get("prevention")),
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