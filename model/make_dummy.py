import os
import tensorflow as tf

# 1. Define the same setup your real model will have
IMG_SIZE = (224, 224, 3)  # Height, Width, RGB Channels
NUM_CLASSES = 10

# 2. Build a super simple fake neural network
inputs = tf.keras.Input(shape=IMG_SIZE)
# Flatten the image pixels and pass them through a dummy layer
x = tf.keras.layers.Flatten()(inputs)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

dummy_model = tf.keras.Model(inputs=inputs, outputs=outputs)

# 3. Ensure the 'model' directory exists and save it
os.makedirs("model", exist_ok=True)
model_path = "model/maize_model.h5"
dummy_model.save(model_path)

print(f"🎉 Success! Dummy model created and saved to: {model_path}")