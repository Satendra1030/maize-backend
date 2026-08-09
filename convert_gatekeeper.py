import os
import shutil
import tensorflow as tf
import keras

# ---------------------------------------------------------
# CRITICAL FIX (v3):
#
# v1 (set_global_policy before load_model) failed because
# load_model() restores each layer's OWN baked-in dtype_policy
# from its saved config, ignoring the global policy entirely.
#
# v2 (rebuild architecture + load_weights) failed because
# load_weights() matches weights to layers by name, and the
# freshly rebuilt nested MobileNetV2 submodel almost certainly
# got a different auto-generated name than the one in the
# saved file (e.g. "mobilenetv2_1.00_224_1" vs
# "mobilenetv2_1.00_224"), silently loading the WRONG weights
# into the backbone. Confirmed by testing: the converted
# gatekeeper scored a real maize leaf at only 12.79% Maize_Leaf
# -- confidently wrong, consistent with a backbone that has
# scrambled/mismatched weights, not merely fp16 rounding.
#
# v3 (this version) avoids rebuilding entirely. Key fact: in
# Keras's mixed_float16 policy, WEIGHTS ARE ALWAYS STORED IN
# FLOAT32 -- only the forward-pass compute happens in float16
# for speed. So load_model() already gives us the correct,
# undamaged, float32-stored weights. All we need to do is
# recursively override every layer's compute dtype_policy to
# float32 (including inside the nested MobileNetV2 submodel)
# BEFORE exporting -- no weight reloading, no architecture
# rebuild, no name-matching risk at all.
# ---------------------------------------------------------

KERAS_MODEL_PATH = "model/gatekeeper_final.keras"
SAVED_MODEL_DIR = "model/gatekeeper_saved_model"
TFLITE_OUTPUT_PATH = "model/gatekeeper_model.tflite"


def force_float32_policy(layer, depth=0):
    """Recursively force every layer (including nested submodels
    like the MobileNetV2 backbone) onto a pure float32 compute
    policy. Weights themselves are untouched -- only the policy
    controlling forward-pass compute dtype changes."""
    try:
        layer.dtype_policy = keras.dtype_policies.DTypePolicy("float32")
    except Exception as e:
        print(f"  (could not set policy on {layer.name}: {e})")

    # Recurse into nested models/layers (e.g. the MobileNetV2 backbone
    # is itself a full model living inside this model as a single layer).
    if hasattr(layer, "layers"):
        for sublayer in layer.layers:
            force_float32_policy(sublayer, depth + 1)


print("=" * 60)
print("GATEKEEPER TFLITE CONVERSION (v3 - policy override, no rebuild)")
print("=" * 60)

# ---------------------------------------------------------
# 1. Load the actual trained model (correct weights, just
#    running under an fp16 compute policy for now).
# ---------------------------------------------------------

print("\n[1/6] Loading trained Keras model...")

model = keras.models.load_model(
    KERAS_MODEL_PATH,
    compile=False
)

print("Model loaded. Layers:")
for layer in model.layers:
    print(" -", layer.name, type(layer).__name__)


# ---------------------------------------------------------
# 2. Force float32 compute policy everywhere, recursively.
# ---------------------------------------------------------

print("\n[2/6] Forcing float32 compute policy on all layers (recursive)...")

force_float32_policy(model)

mixed_precision_layers = [
    layer.name for layer in model.layers
    if "float16" in str(getattr(layer, "dtype_policy", "")).lower()
]
if mixed_precision_layers:
    print(f"WARNING: top-level layers still report float16: {mixed_precision_layers}")
else:
    print("Confirmed: top-level layers report float32 policy.")

# Also check inside the nested backbone specifically, since that's
# where v2 went wrong.
for layer in model.layers:
    if hasattr(layer, "layers"):
        nested_fp16 = [
            sub.name for sub in layer.layers
            if "float16" in str(getattr(sub, "dtype_policy", "")).lower()
        ]
        if nested_fp16:
            print(f"WARNING: nested layers inside '{layer.name}' still float16: {nested_fp16}")
        else:
            print(f"Confirmed: all {len(layer.layers)} nested layers inside '{layer.name}' are float32.")


# ---------------------------------------------------------
# 3. Build inference model
# ---------------------------------------------------------

print("\n[3/6] Creating clean inference model...")

inputs = keras.Input(shape=(224, 224, 3), dtype="float32", name="input")
outputs = model(inputs, training=False)
outputs = keras.ops.cast(outputs, "float32")

inference_model = keras.Model(inputs=inputs, outputs=outputs, name="gatekeeper_inference")

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
# 5. Convert to TFLite (no Flex)
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