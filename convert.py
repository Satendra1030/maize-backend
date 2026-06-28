import tensorflow as tf

print("Loading heavy .keras model...")
# FIXED: Pointed to your actual model file path
model = tf.keras.models.load_model('model/final_model.keras')

print("Converting to TFLite format...")
# Initialize the TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Convert the model
tflite_model = converter.convert()

print("Saving lightweight .tflite model...")
# FIXED: Saving it as final_model.tflite inside your model directory
with open('model/final_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Success! Your model is optimized.")