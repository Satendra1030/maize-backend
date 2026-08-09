import numpy as np
from PIL import Image
import tensorflow as tf
from utils.preprocessing import preprocess_image

# NOTE: point this at the compiled .tflite file, not the .keras archive --
# tf.lite.Interpreter cannot load a .keras SavedModel/archive directly.
MODEL_PATH = "model/final_model.tflite"
IMAGE_PATH = "test_leaf.jpg"

# Confirmed correct order -- matches app.py's CLASS_NAMES exactly.
CLASS_NAMES = ["Blight", "Common Rust", "Gray_Leaf_Spot", "Healthy"]


def test_prediction():
    print(f"Loading local TFLite model from: {MODEL_PATH}")

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Processing target image: {IMAGE_PATH}")
    img = Image.open(IMAGE_PATH).convert("RGB")
    processed_img = preprocess_image(img, target_size=(224, 224))

    input_data = np.array(processed_img, dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_index = np.argmax(predictions)

    print("\n--- LOCAL PREDICTION RESULTS ---")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}: {predictions[i]*100:.2f}%")

    print("--------------------------------")
    print(f"FINAL DECISION: {CLASS_NAMES[predicted_index]} ({predictions[predicted_index]*100:.2f}%)")


if __name__ == "__main__":
    test_prediction()