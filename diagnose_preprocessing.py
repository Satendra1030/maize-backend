"""
Diagnostic script: figure out which preprocessing the classifier
(model/final_model.keras) actually expects, since the training
pipeline's exact preprocessing step is unknown.

Runs the SAME image through the model three different ways:
  A) raw [0, 255]           -- no scaling at all
  B) [0, 1]                 -- divide by 255
  C) [-1, 1]                -- mobilenet_v2.preprocess_input (x/127.5 - 1)

Whichever produces a single clearly-dominant, confident class
(as opposed to four near-equal, muddy percentages) is the most
likely correct preprocessing -- a model fed the wrong input scale
usually still produces SOME output, but confidence tends to be
flat/uncertain rather than sharply peaked.

This is a heuristic, not proof -- ideally you'd confirm against
an image with a definitively known true label. But in the absence
of that, "which one gives a confident, sharp answer" is a
reasonable practical signal.

Usage:
    python diagnose_preprocessing.py [path_to_image]
    (defaults to test_leaf.jpg)
"""

import importlib
import sys
import numpy as np
from PIL import Image
import tensorflow as tf


def mobilenet_preprocess_input(x):
    try:
        module = importlib.import_module("keras.applications.mobilenet_v2")
    except ModuleNotFoundError:
        try:
            module = importlib.import_module("tensorflow.keras.applications.mobilenet_v2")
        except ModuleNotFoundError:  # pragma: no cover
            x = np.asarray(x, dtype=np.float32)
            return x / 127.5 - 1.0
    return module.preprocess_input(x)

MODEL_PATH = "model/final_model.keras"
CLASS_NAMES = ["Blight", "Common Rust", "Gray_Leaf_Spot", "Healthy"]


def load_and_resize(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert("RGB").resize(target_size)
    return np.array(img, dtype=np.float32)


def run_variant(model, raw_pixels, label, transform_fn):
    batch = np.expand_dims(transform_fn(raw_pixels.copy()), axis=0)
    predictions = model.predict(batch, verbose=0)[0]

    print(f"\n--- Variant: {label} ---")
    print(f"  Input range: min={batch.min():.2f}, max={batch.max():.2f}")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_name}: {predictions[i] * 100:.2f}%")
    top_idx = int(np.argmax(predictions))
    top_conf = float(predictions[top_idx])
    print(f"  TOP: {CLASS_NAMES[top_idx]} ({top_conf * 100:.2f}%)")
    return top_conf


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_leaf.jpg"

    print(f"Loading classifier from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    print(f"Loading image: {image_path}")
    raw_pixels = load_and_resize(image_path)

    print("\n" + "=" * 60)
    print("COMPARING PREPROCESSING VARIANTS")
    print("=" * 60)

    results = {}
    results["A) raw [0,255]"] = run_variant(
        model, raw_pixels, "A) raw [0,255] -- no scaling",
        lambda x: x
    )
    results["B) [0,1]"] = run_variant(
        model, raw_pixels, "B) [0,1] -- divide by 255",
        lambda x: x / 255.0
    )
    results["C) [-1,1]"] = run_variant(
        model, raw_pixels, "C) [-1,1] -- mobilenet_v2.preprocess_input",
        lambda x: mobilenet_preprocess_input(x)
    )

    print("\n" + "=" * 60)
    print("SUMMARY -- highest top-1 confidence usually indicates")
    print("the correct preprocessing (most confident, sharpest answer)")
    print("=" * 60)
    for label, conf in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {label}: {conf * 100:.2f}% confidence")


if __name__ == "__main__":
    main()