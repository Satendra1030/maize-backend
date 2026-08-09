import sys
import numpy as np
from PIL import Image
import tensorflow as tf
from utils.preprocessing import preprocess_image

MODEL_PATH = "model/gatekeeper_model.tflite"

# Class mapping from gatekeeper_code.txt:
#   Index 0 = Foreign_Object
#   Index 1 = Maize_Leaf
# Sigmoid output near 1.0 = Maize_Leaf, near 0.0 = Foreign_Object


def test_gatekeeper(image_path: str):
    print(f"Loading gatekeeper TFLite model from: {MODEL_PATH}")

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Input shape :", input_details[0]['shape'])
    print("Output shape:", output_details[0]['shape'])

    print(f"\nProcessing target image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    processed_img = preprocess_image(img, target_size=(224, 224))
    input_data = np.array(processed_img, dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    print("\n--- GATEKEEPER RAW OUTPUT ---")
    print("Raw sigmoid output:", predictions)
    maize_probability = float(predictions[0])
    print(f"Maize_Leaf probability : {maize_probability * 100:.2f}%")
    print(f"Foreign_Object probability: {(1 - maize_probability) * 100:.2f}%")
    print("--------------------------------")
    print(f"DECISION: {'MAIZE LEAF' if maize_probability > 0.5 else 'REJECTED (foreign object)'}")


if __name__ == "__main__":
    # Usage: python test_gatekeeper.py [path_to_image]
    # Defaults to test_leaf.jpg if no argument given.
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_leaf.jpg"
    test_gatekeeper(image_path)