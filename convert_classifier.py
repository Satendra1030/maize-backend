import os
import shutil
import tensorflow as tf
import keras

# Force float32 everywhere BEFORE loading the model. train.py does not
# set a mixed_float16 policy, so this should be a no-op in practice --
# but forcing it explicitly guards against a silent regression if the
# model is ever retrained on GPU with mixed precision enabled, which
# would otherwise reproduce the same "needs Flex ops" conversion error
# hit with the gatekeeper model.
keras.mixed_precision.set_global_policy("float32")

KERAS_MODEL_PATH = "model/final_model.keras"
SAVED_MODEL_DIR = "model/final_model_saved_model"
TFLITE_OUTPUT_PATH = "model/final_model.tflite"


print("=" * 60)
print("DISEASE CLASSIFIER (EfficientNetB0) TFLITE CONVERSION")
print("=" * 60)

# ---------------------------------------------------------
# 1. Load the trained classifier model
# ---------------------------------------------------------

print("\n[1/5] Loading trained Keras model...")

model = keras.models.load_model(
    KERAS_MODEL_PATH,
    compile=False
)

print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)

mixed_precision_layers = [
    layer.name for layer in model.layers
    if "float16" in str(getattr(layer, "dtype_policy", "")).lower()
]
if mixed_precision_layers:
    print(
        "WARNING: These layers still report a float16 policy "
        f"after the global override: {mixed_precision_layers}"
    )
else:
    print("Confirmed: model is running under a float32 policy.")

print("\nModel layers:")
for layer in model.layers:
    print(" -", layer.name, type(layer).__name__)


# ---------------------------------------------------------
# 2. Build inference model
# ---------------------------------------------------------
#
# EfficientNetB0 has its own rescaling built in and expects
# raw [0, 255] pixel input -- do not add any Rescaling layer
# here. utils/preprocessing.py must send raw pixels to match.
# ---------------------------------------------------------

print("\n[2/5] Building inference model...")

inputs = keras.Input(
    shape=(224, 224, 3),
    dtype="float32",
    name="input"
)

outputs = model(inputs, training=False)

# Force final output to float32 (harmless if already float32; a
# safety net in case a future retrain reintroduces mixed precision).
outputs = keras.ops.cast(outputs, "float32")

inference_model = keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="disease_classifier_inference"
)

print("Inference model created.")


# ---------------------------------------------------------
# 3. Export SavedModel
# ---------------------------------------------------------

print("\n[3/5] Exporting SavedModel...")

if os.path.exists(SAVED_MODEL_DIR):
    shutil.rmtree(SAVED_MODEL_DIR)

inference_model.export(SAVED_MODEL_DIR)

print("SavedModel exported to:")
print(SAVED_MODEL_DIR)


# ---------------------------------------------------------
# 4. Convert to TFLite
# ---------------------------------------------------------

print("\n[4/5] Converting to TFLite...")

converter = tf.lite.TFLiteConverter.from_saved_model(
    SAVED_MODEL_DIR
)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

# IMPORTANT: do NOT enable SELECT_TF_OPS / Flex here. app.py loads
# models with the lightweight tflite_runtime / ai_edge_litert
# interpreter (no Flex delegate available) to stay under Render's
# free-tier RAM limit. A Flex-dependent model converts fine locally
# but crashes in production at interpreter.allocate_tensors().
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS
]

# Keep float32 input/output
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()


# ---------------------------------------------------------
# 5. Save model
# ---------------------------------------------------------

print("\n[5/5] Saving TFLite model...")

os.makedirs("model", exist_ok=True)

with open(TFLITE_OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

size_mb = os.path.getsize(TFLITE_OUTPUT_PATH) / (1024 * 1024)

print("\n" + "=" * 60)
print("CONVERSION SUCCESSFUL")
print("=" * 60)
print("Output:", TFLITE_OUTPUT_PATH)
print(f"Size: {size_mb:.2f} MB")


# ---------------------------------------------------------
# Cleanup
# ---------------------------------------------------------

if os.path.exists(SAVED_MODEL_DIR):
    shutil.rmtree(SAVED_MODEL_DIR)

print("\nTemporary SavedModel folder removed.")
print("Done.")