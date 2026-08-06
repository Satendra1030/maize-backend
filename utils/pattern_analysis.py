"""
Pattern-recognition layer for maize leaf images.

Runs AFTER the CNN classification and returns interpretable evidence
about the leaf that the CNN alone does not expose. The pipeline:

  1. Segment the leaf from the background using a green-channel HSV
     mask (captures the green leaf area regardless of background).
  2. Inside the leaf mask, detect lesions / spots using colour
     thresholds in HSV space plus a Sobel-edge response.
  3. Compute colour statistics (mean HSV, green-dominance ratio,
     brown/yellow/red lesion ratio) and a co-occurrence-free texture
     proxy (edge density).
  4. Return a structured dict the API can echo back to the client
     and use to flag common failure modes (e.g. CNN says "Healthy"
     but 25% of the leaf is brown lesions).

This is a *parallel* analysis. The disease decision is still the
CNN's; the pattern layer adds evidence and a sanity check.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import cv2  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# Segmentation helpers
# ---------------------------------------------------------------------------

def _segment_leaf(bgr: np.ndarray) -> np.ndarray:
    """
    Build a binary mask of the leaf pixels.

    Maize leaves are predominantly green. We deliberately use a
    permissive HSV green range rather than a strict classifier so this
    works on yellowed / blighted leaves too. Anything not matching the
    green range is treated as background.

    Args:
        bgr: OpenCV BGR image (H, W, 3), uint8.

    Returns:
        uint8 mask of shape (H, W), values 0 or 255.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Broad green range: H 25-90, S>30, V>30.
    # Lower bound intentionally wide so yellowing leaves are still kept.
    lower_green = np.array([25, 30, 30], dtype=np.uint8)
    upper_green = np.array([90, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological cleanup so small specks of background don't count.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep only the largest connected component (the leaf).
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    leaf_mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    return leaf_mask


def _lesion_mask(bgr: np.ndarray, leaf_mask: np.ndarray) -> np.ndarray:
    """
    Identify lesion / diseased pixels INSIDE the leaf mask.

    Lesions on maize leaves tend to be brown, tan, or yellow rather
    than healthy green. We union two HSV ranges with the leaf mask to
    pull out non-green regions.

    Returns:
        uint8 mask of lesion pixels, same shape as leaf_mask.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Brown / tan lesions (and gray GLS): H 10-30, low-to-mid value.
    lower_brown = np.array([10, 30, 20], dtype=np.uint8)
    upper_brown = np.array([30, 255, 200], dtype=np.uint8)
    brown = cv2.inRange(hsv, lower_brown, upper_brown)

    # Red / rust pustules: H 0-10 or 170-180.
    lower_red1 = np.array([0, 70, 40], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 230], dtype=np.uint8)
    lower_red2 = np.array([170, 70, 40], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 230], dtype=np.uint8)
    rust = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    # Yellow chlorosis: H 20-35, low saturation accounted for, value>120.
    lower_yellow = np.array([20, 50, 120], dtype=np.uint8)
    upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
    yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lesions = cv2.bitwise_or(brown, cv2.bitwise_or(rust, yellow))
    lesions = cv2.bitwise_and(lesions, leaf_mask)

    # Small noise clean-up.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    lesions = cv2.morphologyEx(lesions, cv2.MORPH_OPEN, kernel, iterations=1)
    return lesions


# ---------------------------------------------------------------------------
# Texture / edge proxy
# ---------------------------------------------------------------------------

def _edge_density(bgr: np.ndarray, leaf_mask: np.ndarray) -> float:
    """Ratio of edge pixels within the leaf, in [0, 1]."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    edge_pixels = cv2.countNonZero(cv2.bitwise_and(edges, leaf_mask))
    leaf_area = cv2.countNonZero(leaf_mask)
    return float(edge_pixels) / float(leaf_area) if leaf_area > 0 else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_leaf_patterns(img_rgb: np.ndarray) -> dict:
    """
    Run the full pattern-recognition pipeline on one image.

    Args:
        img_rgb: HxWx3 RGB uint8 image (matching how PIL hands it back).

    Returns:
        dict with the following keys:
          - leaf_coverage_ratio: fraction of the image that is leaf (0-1)
          - lesion_ratio: fraction of leaf area covered by lesions (0-1)
          - mean_h, mean_s, mean_v: mean HSV inside the leaf mask
          - green_dominance: 1.0 - lesion_ratio as a quick health proxy
          - edge_density: Sobel/Canny edge density, 0-1
          - detected_patterns: list of human-readable pattern tags
          - consistency_flag: "ok" | "warning" | "critical"
              warns when CNN output and pattern evidence contradict
    """
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    leaf_mask = _segment_leaf(bgr)
    lesion_mask = _lesion_mask(bgr, leaf_mask)

    leaf_area = int(cv2.countNonZero(leaf_mask))
    total_area = int(leaf_mask.shape[0] * leaf_mask.shape[1])
    lesion_area = int(cv2.countNonZero(lesion_mask))

    leaf_coverage_ratio = leaf_area / total_area if total_area else 0.0
    lesion_ratio = lesion_area / leaf_area if leaf_area else 0.0

    # Mean HSV inside the leaf only.
    if leaf_area > 0:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        masked_hsv = hsv[np.where(leaf_mask > 0)]
        mean_h = float(np.mean(masked_hsv[:, 0]))
        mean_s = float(np.mean(masked_hsv[:, 1]))
        mean_v = float(np.mean(masked_hsv[:, 2]))
    else:
        mean_h = mean_s = mean_v = 0.0

    edge_density = _edge_density(bgr, leaf_mask)

    # Human-readable pattern tags surfaced to the user.
    detected_patterns = []
    if leaf_coverage_ratio < 0.05:
        detected_patterns.append("no_leaf_detected")
    if lesion_ratio > 0.05:
        detected_patterns.append("brown_or_tan_lesions")
    if lesion_ratio > 0.15:
        detected_patterns.append("extensive_lesion_coverage")
    if mean_h > 35 and mean_s < 100:
        detected_patterns.append("yellow_chlorosis")
    if edge_density > 0.15:
        detected_patterns.append("high_texture_complexity")
    if leaf_area > 0 and lesion_ratio < 0.02 and mean_h < 70:
        detected_patterns.append("uniformly_green")

    green_dominance = max(0.0, 1.0 - lesion_ratio)

    return {
        "leaf_coverage_ratio": round(leaf_coverage_ratio, 4),
        "lesion_ratio": round(lesion_ratio, 4),
        "mean_h": round(mean_h, 2),
        "mean_s": round(mean_s, 2),
        "mean_v": round(mean_v, 2),
        "green_dominance": round(green_dominance, 4),
        "edge_density": round(edge_density, 4),
        "detected_patterns": detected_patterns,
    }


def consistency_check(cnn_label: str, pattern: dict) -> str:
    """
    Compare the CNN's classification against the pattern evidence.

    Returns:
        "ok"          - evidence agrees with CNN
        "warning"     - mild disagreement (e.g. some lesions but CNN says Healthy)
        "critical"    - strong disagreement worth surfacing to the user
    """
    label = (cnn_label or "").lower()
    lesion_ratio = pattern.get("lesion_ratio", 0.0)
    detected = set(pattern.get("detected_patterns", []))
    mean_h = pattern.get("mean_h", 0.0)
    mean_s = pattern.get("mean_s", 0.0)

    # If the image isn't really a leaf, the CNN was operating on junk.
    if "no_leaf_detected" in detected:
        return "critical"

    # CNN says healthy but visual evidence says otherwise.
    if label == "healthy":
        if lesion_ratio > 0.20 or "extensive_lesion_coverage" in detected:
            return "critical"
        if lesion_ratio > 0.08 or "brown_or_tan_lesions" in detected:
            return "warning"

    # CNN says a specific disease but image looks predominantly healthy.
    if label != "healthy":
        if lesion_ratio < 0.05 and mean_h < 70 and mean_s > 80:
            return "warning"

    # CNN says a specific disease but image is uniformly green.
    if "uniformly_green" in detected and lesion_ratio < 0.02 and label != "healthy":
        return "warning"

    return "ok"