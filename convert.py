import os
import shutil
import tensorflow as tf
import keras

# Force float32 everywhere BEFORE loading the model. Otherwise a
# baked-in mixed_float16 training policy forces Conv2D/DepthwiseConv2D/
# Relu6 into fp16, and the native TFLite converter can't lower those,
# demanding Flex ops instead -- which app.py's lightweight
# tflite_runtime/ai_edge_litert interpreter cannot execute on Render.
keras.mixed_precision.set_global_policy("float32")

KERAS_MODEL_PATH = "model/gatekeeper_final.keras"
SAVED_MODEL_DIR = "model/temp_saved_model"
TFLITE_OUTPUT_PATH = "model/gatekeeper_model.tflite"

print(f"🔄 Loading Keras model from {KERAS_MODEL_PATH}...")

model = keras.models.load_model(
    KERAS_MODEL_PATH,
    compile=False
)

print("✅ Keras model loaded successfully.")

# ---------------------------------------------------------
# Create a clean inference model
# ---------------------------------------------------------

print("🧹 Creating clean inference model...")

inputs = keras.Input(
    shape=(224, 224, 3),
    dtype="float32",
    name="input"
)

outputs = model(inputs, training=False)

# Force final output to float32
outputs = keras.ops.cast(outputs, "float32")

inference_model = keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="gatekeeper_inference"
)

print("✅ Inference model created.")

# ---------------------------------------------------------
# Export to SavedModel
# ---------------------------------------------------------

if os.path.exists(SAVED_MODEL_DIR):
    print("🗑️ Removing old temporary SavedModel...")
    shutil.rmtree(SAVED_MODEL_DIR)

print("💾 Exporting model to SavedModel format...")

inference_model.export(SAVED_MODEL_DIR)

print("✅ SavedModel exported.")

# ---------------------------------------------------------
# Convert SavedModel → TFLite
# ---------------------------------------------------------

print("⚡ Converting SavedModel to TFLite...")

converter = tf.lite.TFLiteConverter.from_saved_model(
    SAVED_MODEL_DIR
)

# Standard optimization
converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

# IMPORTANT: do NOT enable SELECT_TF_OPS / Flex here.
#
# app.py deliberately loads models with the lightweight
# tflite_runtime (or ai_edge_litert) interpreter to keep RAM
# usage under Render's free-tier 512MB limit. That runtime
# does not include the Flex delegate, so a Flex-dependent
# .tflite file converts fine locally but crashes in
# production at interpreter.allocate_tensors().
#
# The AddV2/Conv2D/DepthwiseConv2dNative/Pad/Relu6 ops that
# previously required Flex only did so because the model was
# running in fp16 (see the mixed_precision policy override
# above). With float32 forced, these are all natively
# supported TFLite builtins.
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS
]

# Keep float32 input/output
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()

# ---------------------------------------------------------
# Save TFLite model
# ---------------------------------------------------------

with open(TFLITE_OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

print("✅ TFLite conversion successful!")

# ---------------------------------------------------------
# Cleanup
# ---------------------------------------------------------

if os.path.exists(SAVED_MODEL_DIR):
    shutil.rmtree(SAVED_MODEL_DIR)

size_mb = os.path.getsize(TFLITE_OUTPUT_PATH) / (1024 * 1024)

print("----------------------------------------")
print(f"📦 TFLite model: {TFLITE_OUTPUT_PATH}")
print(f"📏 Model size: {size_mb:.2f} MB")
print("----------------------------------------")