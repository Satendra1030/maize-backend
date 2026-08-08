import os
import shutil
import tensorflow as tf
import keras

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

# Allow TensorFlow Select operators when required.
#
# This is important because the conversion log shows:
# AddV2
# Conv2D
# DepthwiseConv2dNative
# Pad
# Relu6
#
# are currently being rejected by the native TFLite converter.
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
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