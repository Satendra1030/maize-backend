import numpy as np
from PIL import Image
import tensorflow as tf  # Using the core engine fallback
from utils.preprocessing import preprocess_image

# 1. Path configurations
MODEL_PATH = "model/final_model.tflite"
IMAGE_PATH = "G:\maize-backend\test_leaf.jpg"  # <-- Change this to your test leaf image path!

# Correct Keras alphabetical sorting mapping
CLASS_NAMES = ["Common Rust", "Gray Leaf Spot", "Healthy", "Northern Leaf Blight"]

def test_prediction():
    print(f"Loading local TFLite model from: {MODEL_PATH}")
    
    # 2. Load TFLite Model using core tensorflow engine
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 3. Load and preprocess target leaf image
    print(f"Processing target image: {IMAGE_PATH}")
    img = Image.open(IMAGE_PATH).convert("RGB")
    processed_img = preprocess_image(img, target_size=(224, 224))
    
    # Ensure correct float32 precision data type matching
    input_data = np.array(processed_img, dtype=np.float32)
    
    # 4. Fire Inference Loop
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # 5. Extract and Decode outputs
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_index = np.argmax(predictions)
    
    print("\n--- LOCAL PREDICTION RESULTS ---")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}: {predictions[i]*100:.2f}%")
        
    print("--------------------------------")
    print(f"FINAL DECISION: {CLASS_NAMES[predicted_index]} ({predictions[predicted_index]*100:.2f}%)")

if __name__ == "__main__":
    test_prediction()