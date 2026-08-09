"""
Image preprocessing utilities.

IMPORTANT: the two TFLite models in this project need DIFFERENT
preprocessing. Use the right function for each stage.

- Gatekeeper (MobileNetV2, binary Foreign_Object/Maize_Leaf):
  `mobilenet_v2.preprocess_input` is baked INSIDE the exported model
  graph itself (confirmed by inspecting the training script and by
  testing). Feed it RAW [0, 255] pixels -- use preprocess_image().
  Do NOT pre-scale before calling this model; it scales internally.

- Disease classifier (MobileNetV2, 4-class):
  NO preprocessing layer is baked into this model's graph (confirmed
  by inspecting model/final_model.keras's config.json directly --
  it's a Sequential model with no Rescaling/Lambda layer before the
  MobileNetV2 backbone). This model needs [-1, 1] scaling applied
  EXTERNALLY before inference -- use preprocess_image_classifier().
  Confirmed empirically: running the same test image through three
  candidate preprocessing schemes, [-1,1] scaling gave a dramatically
  more confident, consistent prediction (96%+) than raw [0,255] or
  plain [0,1] scaling.

NOTE ON DEPENDENCIES: this module intentionally does NOT import
TensorFlow/Keras. mobilenet_v2.preprocess_input (mode='tf', the
default) is just: x = x / 127.5; x = x - 1.0 -- a plain linear
rescale, reimplemented below with numpy. TensorFlow is NOT listed in
requirements.txt (the production runtime deliberately uses the
lightweight tflite_runtime instead, to stay under Render's free-tier
RAM limit), so importing tensorflow.keras here would work locally
(where TF happens to be installed in the dev venv) but CRASH in
production the moment this function is called, since the import
would silently fail there. Keeping this dependency-free avoids that
entirely, while producing numerically identical output to the real
mobilenet_v2.preprocess_input.

If either model is ever retrained differently, these functions must
be updated to match -- whatever the training pipeline actually did
must be replicated exactly at inference time. When in doubt, use
diagnose_preprocessing.py to empirically re-verify rather than
guessing.
"""

import numpy as np
from PIL import Image

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def preprocess_image(img: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """
    Prepares a PIL image for the GATEKEEPER model.

    Returns raw [0, 255] pixel values -- the gatekeeper's exported
    graph has mobilenet_v2.preprocess_input baked in internally, so
    external scaling here would double-normalize and corrupt results.

    Args:
        img: A PIL Image (already converted to RGB).
        target_size: (width, height) expected by the model.

    Returns:
        A numpy array of shape (1, height, width, 3) with raw pixel
        values in [0, 255].
    """
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def preprocess_image_classifier(img: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """
    Prepares a PIL image for the DISEASE CLASSIFIER model.

    Applies the mobilenet_v2 [-1, 1] rescale externally, since this
    model's graph does NOT have that baked in. Confirmed empirically
    via diagnose_preprocessing.py. Implemented with plain numpy
    (x/127.5 - 1.0) rather than importing tensorflow.keras, to avoid
    a hard dependency the production runtime doesn't otherwise need.

    Args:
        img: A PIL Image (already converted to RGB).
        target_size: (width, height) expected by the model.

    Returns:
        A numpy array of shape (1, height, width, 3) with pixel
        values scaled to [-1, 1].
    """
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 127.5
    img_array = img_array - 1.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array