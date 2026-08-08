import os
import shutil
import tensorflow as tf

print("Starting clean native TFLite model conversion...")

# 1. Load the original Keras model
model_path = "model/gatekeeper_final.keras"
full_model = tf.keras.models.load_model(model_path, compile=False)

# 2. Rebuild pure inference graph (Bypassing data augmentation/preprocessing)
# We locate the backbone (MobileNetV2 or input layer after augmentation)
try:
    # If MobileNetV2 or backbone is nested, isolate the core input and output
    backbone = full_model.get_layer("mobilenetv2_1.00_224")
    inputs = tf.keras.Input(shape=(224, 224, 3), dtype=tf.float32, name="input_1")
    
    # Pass pure float32 tensor directly to backbone/layers
    x = backbone(inputs)
    
    # Connect remaining top layers (e.g., Dense/GlobalAveragePooling)
    # If the model has top layers after backbone:
    top_layers_started = False
    for layer in full_model.layers:
        if layer == backbone:
            top_layers_started = True
            continue
        if top_layers_started:
            x = layer(x)
            
    inference_model = tf.keras.Model(inputs=inputs, outputs=x)
except Exception:
    # Fallback to direct input/output binding if custom backbone structural isolation isn't required
    inference_model = full_model

# 3. Save standard Keras inference model to temporary directory
temp_dir = "temp_saved_model"
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)

inference_model.export(temp_dir, format="tf_saved_model")

# 4. Configure converter strictly for Built-in TFLite Ops (No Flex/Select TF Ops)
converter = tf.lite.TFLiteConverter.from_saved_model(temp_dir)

# Restrict strictly to standard built-in ops
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS
]

# Standard optimizations (Default quantization keeps compatibility with standard ops)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Enforce standard float32 for fallback operations
converter.target_spec.supported_types = [tf.float32]

# 5. Convert model
tflite_model = converter.convert()

# 6. Save final native .tflite model
output_path = "model/gatekeeper_model.tflite"
with open(output_path, "wb") as f:
    f.write(tflite_model)

# 7. Cleanup temporary artifacts
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)

print(f"SUCCESS! Clean native TFLite model created at: {output_path}")