"""
Image preprocessing utilities.

Matches proposal section 3.7 (Image Processing and Augmentation):
- Resize to 224x224 (MobileNetV2 input size)
- Normalize pixel values to [0, 1] by dividing by 255.0

NOTE: If your training pipeline used
tf.keras.applications.mobilenet_v2.preprocess_input (which scales to [-1, 1]
instead of [0, 1]), you MUST use that exact same function here instead.
Whatever normalization was used during training must be replicated exactly
at inference time, or predictions will be inaccurate.
"""

import numpy as np
from PIL import Image

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def preprocess_image(img: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """
    Prepares a PIL image for model inference.

    Args:
        img: A PIL Image (already converted to RGB).
        target_size: (width, height) expected by the model.

    Returns:
        A numpy array of shape (1, height, width, 3), ready for model.predict().
    """
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)

    # Normalize to [0, 1] -- matches proposal section 3.7.
    # If you trained using mobilenet_v2.preprocess_input, swap this line for:
    #   from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    #   img_array = preprocess_input(img_array)
    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array
