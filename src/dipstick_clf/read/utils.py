"""Improved dipstick reading utils with better color comparison."""

import math
from pathlib import Path
import numpy as np
from ultralytics import YOLO #type: ignore
from loguru import logger
import torch
import cv2
from skimage.color.delta_e import deltaE_ciede2000
from src.config import MODELS_DIR
from src.dipstick_clf.preprocessing.utils import rotate_image

# CONFIGURATION
ROTATION_MODEL_PATH = MODELS_DIR / "dipstick_read/rotation_02/weights/best.pt"
DIPSTICK_MODEL_PATH = MODELS_DIR / "dipstick_read/simple_03/weights/best.pt"
DEVICE = 'cpu' if not torch.backends.mps.is_available() else 'mps'

# Thresholds
DELTA_E_THRESHOLD = 25.0  # Mark results as uncertain if delta E exceeds this
CONFIDENCE_THRESHOLD = 0.7

# Pad index to reference index mapping
TEST_MAP = {
    0: 10, 1: 11, 2: 12, 3: 13, 4: 14,
    5: 15, 6: 16, 7: 17, 8: 18, 9: 19
}

# Reference values mapping for each test
REFERENCE_VALUES = {
    10: ["0.1", "1(16)", "2(33)", "4(66)", "8(131)"],
    11: ["neg", "100(5.5)", "250(14)", "500(28)", "1000(55)"],
    12: ["neg", "+", "++", "+++"],
    13: ["neg", "+-5(0.5)", "+15(1.5)", "++40(3.9)", "+++100(10)"],
    14: ["1.000", "1.010", "1.020", "1.030", "1.040", "1.050", "1.060"],
    15: ["neg", "hemolysis+10", "++50", "+++250", "non hemolysis+10", "++50"],
    16: ["5", "6", "6.5", "7", "8", "9"],
    17: ["neg", "trace", "+30(0.3)", "++100(1.0)", "+++300(3.0)", "++++1000(10)"],
    18: ["neg", "trace", "pos"],
    19: ["neg", "+25", "++75", "+++500"]
}

# Initialize model
rotation_model = YOLO(ROTATION_MODEL_PATH)
dipstick_model = YOLO(DIPSTICK_MODEL_PATH)

def correct_rotation(image_path, model):
    """
    Uses a YOLO model to find the 'strip annotation' and determines the
    rotation correction needed to move it to the left of frame.

    Args:
        image_path (Path or str): Path to image.
        model (YOLO): Loaded Ultralytics model.
    """
    # Run inference
    results = model.predict(image_path, iou=0.7, conf=0.5, verbose=False)[0]

    # Filter for strip boxes (class 1)
    strip_boxes = [box for box in results.boxes if int(box.cls[0]) == 0]

    # If no strip detected, return None
    if not strip_boxes:
        logger.error("Strip not detected. Please check image.")
        return None

    # Get box centre coordinates
    x_center, y_center = strip_boxes[0].xywh[0][0].item(), strip_boxes[0].xywh[0][1].item()

    # Get image dimensions
    height, width = results.orig_shape

    dx = x_center - width/2
    dy = y_center - height/2

    if abs(dx) > abs(dy):
        # Strip is left or right
        rotation = 0 if dx < 0 else 180
    else:
        # Strip is top or bottom
        rotation = 90 if dy < 0 else 270

    rotate_image(image_path, rotation)


def calculate_delta_e(lab1, lab2, method='ciede2000'):
    """
    Calculate color difference between two L*a*b* colors.
    
    Args:
        lab1, lab2: (L, a, b) tuples or arrays
        method: 'ciede2000' (perceptually uniform) or 'cie76' (Euclidean)
    
    Returns:
        Delta E value (lower = more similar)
    """
    if method == 'ciede2000':
        lab1_reshaped = np.array(lab1).reshape(1, 3)
        lab2_reshaped = np.array(lab2).reshape(1, 3)
        return float(deltaE_ciede2000(lab1_reshaped, lab2_reshaped)[0])
    else:  # CIE76
        dL = lab1[0] - lab2[0]
        da = lab1[1] - lab2[1]
        db = lab1[2] - lab2[2]
        return math.sqrt(dL**2 + da**2 + db**2)

def sample_color_lab(img, box, sample_method='center_weighted'):
    """
    Samples color from a bounding box region with improved accuracy.
    
    Args:
        img: BGR image (OpenCV format)
        box: [x_min, y_min, x_max, y_max]
        sample_method: 
            - 'center_weighted': Sample center 60% of region (default)
            - 'grid': 9-point grid sampling with median
            - 'full': Average entire region
    
    Returns:
        Average L*a*b* color tuple or None if invalid
    """
    x_min, y_min, x_max, y_max = map(int, box)
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(img.shape[1], x_max), min(img.shape[0], y_max)

    roi_bgr = img[y_min:y_max, x_min:x_max]

    if roi_bgr.size == 0:
        logger.warning(f"Empty ROI detected for box {box}")
        return None

    # Apply sampling strategy
    if sample_method == 'center_weighted':
        # Sample center 60% to avoid edge effects (shadows, reflections)
        h, w = roi_bgr.shape[:2]
        margin_h = int(h * 0.2)
        margin_w = int(w * 0.2)
        roi_bgr = roi_bgr[margin_h:h-margin_h, margin_w:w-margin_w]

        if roi_bgr.size == 0:
            roi_bgr = img[y_min:y_max, x_min:x_max]  # Fallback to full region

    elif sample_method == 'grid':
        # Sample 9 points and use median to reduce outlier influence
        h, w = roi_bgr.shape[:2]
        samples = []
        for y_frac in [0.25, 0.5, 0.75]:
            for x_frac in [0.25, 0.5, 0.75]:
                y = min(int(h * y_frac), h - 1)
                x = min(int(w * x_frac), w - 1)
                samples.append(roi_bgr[y, x])

        # Convert samples to LAB
        samples_bgr = np.array(samples, dtype=np.uint8)
        samples_bgr = samples_bgr.reshape((9, 1, 3))
        samples_lab = cv2.cvtColor(samples_bgr, cv2.COLOR_BGR2LAB).reshape(9, 3) # pylint: disable=no-member
        return np.median(samples_lab, axis=0)

    # Convert to L*a*b* and calculate mean
    roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB) # pylint: disable=no-member
    avg_lab = np.mean(roi_lab, axis=(0, 1))

    return avg_lab

def read_dipstick(image_path: Path, sample_method='center_weighted', delta_e_method='ciede2000'):
    """
    Performs inference and color comparison to determine dipstick results.
    
    Args:
        image_path: Path to dipstick image
        sample_method: Color sampling strategy ('center_weighted', 'grid', 'full')
        delta_e_method: Color distance metric ('ciede2000' or 'cie76')
    
    Returns:
        Dictionary of test results with values and confidence metrics
    """
    # Rotation correction
    correct_rotation(image_path, rotation_model)

    # Reload image
    logger.info(f"Loading image from {image_path.name}")
    img_bgr = cv2.imread(str(image_path)) # pylint: disable=no-member

    if img_bgr is None:
        logger.error("Image failed to load.")
        return {}

    # Run dipstick detection
    logger.info("Running detection inference...")
    results = dipstick_model.predict(img_bgr,
                            conf=CONFIDENCE_THRESHOLD,
                            iou=0.5,
                            verbose=False,
                            device=DEVICE)

    detections = results[0].boxes.cpu().numpy() #type: ignore

    # Organize detections by class
    all_boxes = {}
    for box in detections:
        class_id = int(box.cls[0])
        if 0 <= class_id <= 19:  # Only pads (0-9) and references (10-19)
            if class_id not in all_boxes:
                all_boxes[class_id] = []
            all_boxes[class_id].append({
                'coords': box.xyxy[0],
                'confidence': float(box.conf[0])
            })

    final_readings = {}

    # Color Matching Analysis
    logger.info(f"Starting color matching using {delta_e_method} with {sample_method} sampling...")

    for pad_id, ref_id in TEST_MAP.items():
        if pad_id not in all_boxes or ref_id not in all_boxes:
            pad_name = dipstick_model.names[pad_id].replace("pad_", "").upper()
            logger.warning(f"Missing detections for {pad_name}. Skipping.")
            final_readings[pad_name] = "NOT DETECTED"
            continue

        # Sample pad color (use highest confidence detection if multiple)
        pad_boxes = sorted(all_boxes[pad_id], key=lambda x: x['confidence'], reverse=True)
        pad_lab = sample_color_lab(img_bgr, pad_boxes[0]['coords'], sample_method)

        if pad_lab is None:
            continue

        # Sample all reference squares for this test
        ref_boxes = all_boxes[ref_id]
        ref_boxes.sort(key=lambda x: x['coords'][0])  # Sort by x-coordinate (left to right)

        min_delta_e = float('inf')
        second_min_delta_e = float('inf')
        best_match_value = "UNDETECTED"

        for i, ref_box in enumerate(ref_boxes):
            ref_lab = sample_color_lab(img_bgr, ref_box['coords'], sample_method)
            if ref_lab is None:
                continue

            # Calculate color distance
            delta_e = calculate_delta_e(pad_lab, ref_lab, method=delta_e_method)

            # Track best and second-best matches
            if delta_e < min_delta_e:
                second_min_delta_e = min_delta_e
                min_delta_e = delta_e
                if i < len(REFERENCE_VALUES.get(ref_id, [])):
                    best_match_value = REFERENCE_VALUES[ref_id][i]
                else:
                    best_match_value = f"Index {i} (Unknown)"
            elif delta_e < second_min_delta_e:
                second_min_delta_e = delta_e

        # Assess confidence and store result
        pad_name = dipstick_model.names[pad_id].replace("pad_", "").upper()

        # Calculate confidence score based on separation between best and second-best
        if second_min_delta_e != float('inf'):
            separation = second_min_delta_e - min_delta_e
            confidence = "high" if separation > 5.0 else "medium" if separation > 2.0 else "low"
        else:
            confidence = "only_match"

        # Mark uncertain results
        uncertainty_flag = "?" if min_delta_e > DELTA_E_THRESHOLD else ""

        final_readings[pad_name] = {
            'value': f"{best_match_value}{uncertainty_flag}",
            'delta_e': round(min_delta_e, 2),
            'confidence': confidence,
            'detection_conf': round(pad_boxes[0]['confidence'], 2)
        }

        if min_delta_e > DELTA_E_THRESHOLD:
            logger.warning(f"{pad_name}: High deltaE ({min_delta_e:.2f}), result may be unreliable")

    return final_readings
