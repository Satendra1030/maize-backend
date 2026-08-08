import os
import shutil
import tensorflow as tf
import keras
from keras import layers

# ---------------------------------------------------------
# CRITICAL FIX (v2):
#
# keras.mixed_precision.set_global_policy("float32") only
# affects layers created AFTER that call. It does NOT
# override a model loaded via keras.models.load_model(),
# because each layer in a .keras file restores its OWN
# dtype_policy from its saved config -- it never consults
# the global policy at load time. That's why the previous
# fix still produced fp16 tensors (and even MORE ops
# requiring Flex: RealDiv, Sub -- the internal
# preprocess_input scaling math).
#
# The reliable fix: rebuild the EXACT SAME architecture used
# in training (see gatekeeper_code.txt) as brand-new layers
# -- which DO respect the float32 policy set below -- then
# load ONLY the trained weights into that fresh architecture,
# instead of loading the full serialized model with its
# baked-in policy.
# ---------------------------------------------------------
keras.mixed_precision.set_global_policy("float32")

KERAS_MODEL_PATH = "model/gatekeeper_final.keras"
SAVED_MODEL_DIR = "model/gatekeeper_saved_model"
TFLITE_OUTPUT_PATH = "model/gatekeeper_model.tflite"

IMG_SIZE = (224, 224)


print("=" * 60)
print("GATEKEEPER TFLITE CONVERSION (v2 - float32 rebuild)")
print("=" * 60)

# ---------------------------------------------------------
# 1. Rebuild the exact training architecture, fresh, under
#    the float32 policy set above.
# ---------------------------------------------------------

print("\n[1/6] Rebuilding gatekeeper architecture (float32)...")

inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), name="input")

# Augmentation layers are identity at inference (training=False
# is used automatically during export/predict), kept here only
# so layer names/order match the saved weights exactly.
x = layers.RandomFlip("horizontal_and_vertical")(inputs)
x = layers.RandomRotation(0.15)(x)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

# weights=None: we do NOT want ImageNet weights here -- we're
# about to overwrite everything with the actual trained weights
# via load_weights() below. This also avoids an unnecessary
# network download.
base_model = tf.keras.applications.MobileNetV2(
    weights=None,
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)
base_model.trainable = False

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

gatekeeper_model = tf.keras.Model(inputs, outputs, name="gatekeeper")

print("Architecture rebuilt.")


# ---------------------------------------------------------
# 2. Load ONLY the trained weights (not the architecture
#    config, which is what carries the fp16 policy).
# ---------------------------------------------------------

print("\n[2/6] Loading trained weights from:", KERAS_MODEL_PATH)

gatekeeper_model.load_weights(KERAS_MODEL_PATH)

print("Weights loaded.")

mixed_precision_layers = [
    layer.name for layer in gatekeeper_model.layers
    if "float16" in str(getattr(layer, "dtype_policy", "")).lower()
]
if mixed_precision_layers:
    print(
        "WARNING: these layers still report a float16 policy: "
        f"{mixed_precision_layers}"
    )
else:
    print("Confirmed: model is running under a float32 policy.")


# ---------------------------------------------------------
# 3. Build inference model
# ---------------------------------------------------------

print("\n[3/6] Building inference model...")

inf_inputs = keras.Input(shape=(*IMG_SIZE, 3), dtype="float32", name="input")
inf_outputs = gatekeeper_model(inf_inputs, training=False)
inf_outputs = keras.ops.cast(inf_outputs, "float32")

inference_model = keras.Model(inputs=inf_inputs, outputs=inf_outputs, name="gatekeeper_inference")

print("Inference model created.")


# ---------------------------------------------------------
# 4. Export SavedModel
# ---------------------------------------------------------

print("\n[4/6] Exporting SavedModel...")

if os.path.exists(SAVED_MODEL_DIR):
    shutil.rmtree(SAVED_MODEL_DIR)

inference_model.export(SAVED_MODEL_DIR)

print("SavedModel exported to:", SAVED_MODEL_DIR)


# ---------------------------------------------------------
# 5. Convert to TFLite (no Flex -- see earlier notes: the
#    production runtime, tflite_runtime/ai_edge_litert, has
#    no Flex delegate available).
# ---------------------------------------------------------

print("\n[5/6] Converting to TFLite...")

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()


# ---------------------------------------------------------
# 6. Save model
# ---------------------------------------------------------

print("\n[6/6] Saving TFLite model...")

os.makedirs("model", exist_ok=True)

with open(TFLITE_OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

size_mb = os.path.getsize(TFLITE_OUTPUT_PATH) / (1024 * 1024)

print("\n" + "=" * 60)
print("CONVERSION SUCCESSFUL")
print("=" * 60)
print("Output:", TFLITE_OUTPUT_PATH)
print(f"Size: {size_mb:.2f} MB")

if os.path.exists(SAVED_MODEL_DIR):
    shutil.rmtree(SAVED_MODEL_DIR)

print("\nTemporary SavedModel folder removed.")
print("Done.")