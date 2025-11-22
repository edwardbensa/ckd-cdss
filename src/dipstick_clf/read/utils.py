"""Dipstick reading utils using YOLOv8 detection and colour comparison."""

# Imports
import math
from pathlib import Path
import numpy as np
from ultralytics import YOLO # type: ignore
from loguru import logger
import torch
import cv2
from src.config import MODELS_DIR


# CONFIGURATION
MODEL_PATH = MODELS_DIR / "dipstick_detection5/yolov8_dipstick_simple/weights/best.pt"
DEVICE = 'cpu' if not torch.backends.mps.is_available() else 'mps'

# Pad index to reference index mapping
TEST_MAP = {
    # Pad Class ID (0-9) : Reference Class ID (10-19)
    0: 10,  # pad_urobilinogen -> urobilinogen
    1: 11,  # pad_glucose      -> glucose
    2: 12,  # pad_bilirubin    -> bilirubin
    3: 13,  # pad_ketones      -> ketones
    4: 14,  # pad_SG           -> SG
    5: 15,  # pad_blood        -> blood
    6: 16,  # pad_ph           -> ph
    7: 17,  # pad_protein      -> protein
    8: 18,  # pad_nitrite      -> nitrite
    9: 19,  # pad_leukocytes   -> leukocytes
}

# Reference values mapping for each test
REFERENCE_VALUES = {
    10: ["0.1", "1(16)", "2(33)", "4(66)", "8(131)"], # Urobilinogen mg/dL(umol/L)
    11: ["neg", "100(5.5)", "250(14)", "500(28)", "1000(55)"], # Glucose mg/dL(mmol/L)
    12: ["neg", "+", "++", "+++"], # Bilirubin
    13: ["neg", "+-5(0.5)", "+15(1.5)", "++40(3.9)", "+++100(10)"], # Ketones mg/dL(mmol/L)
    14: ["1.000", "1.010", "1.020", "1.030", "1.040", "1.050", "1.060"], # SG
    15: ["neg", "hemolysis+10", "++50", "+++250", "non hemolysis+10", "++50"], # Blood RBC/uL
    16: ["5", "6", "6.5", "7", "8", "9"], # pH
    17: ["neg", "trace", "+30(0.3)", "++100(1.0)", "+++300(3.0)", "++++1000(10)"], # Protein mg/dl(g/L)
    18: ["neg", "trace", "pos"], # Nitrite
    19: ["neg", "+25", "++75", "+++500"] # Leukocytes WBC/uL
}


# Initialize model
model = YOLO(MODEL_PATH)

def calculate_delta_e(lab1, lab2):
    """
    Calculates the Euclidean distance (Delta E 2000 is more accurate but complex,
    Delta E 76 is the simple Euclidean distance, suitable for a first pass).
    lab1 and lab2 are (L, a, b) tuples or NumPy arrays.
    """
    # Assuming input is already L*a*b*
    dL = lab1[0] - lab2[0]
    da = lab1[1] - lab2[1]
    db = lab1[2] - lab2[2]
    return math.sqrt(dL**2 + da**2 + db**2)

def sample_color_lab(img, box):
    """
    Samples the average color from a bounding box region in the image.
    Returns the average L*a*b* color tuple.
    """
    x_min, y_min, x_max, y_max = map(int, box)

    # Clip coordinates to image bounds
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(img.shape[1], x_max), min(img.shape[0], y_max)

    # Crop the BGR region
    roi_bgr = img[y_min:y_max, x_min:x_max]

    if roi_bgr.size == 0:
        logger.warning(f"Empty ROI detected for box {box}")
        return None
        
    # Convert from BGR (OpenCV default) to L*a*b*
    roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    
    # Calculate the average L, a, and b values across the region
    avg_lab = np.mean(roi_lab, axis=(0, 1))
    return avg_lab

def read_dipstick(image_path: Path):
    """Performs inference and color comparison to determine results."""
    logger.info(f"Loading image from {image_path.name}")
    img_bgr = cv2.imread(str(image_path))
    
    if img_bgr is None:
        logger.error("Image failed to load.")
        return

    # 1. Run Detection (Inference)
    logger.info("Running detection inference...")
    results = model.predict(img_bgr, conf=0.7, iou=0.7, verbose=False, device=DEVICE)
    
    # results[0].boxes contains all detected objects
    detections = results[0].boxes.cpu().numpy()
    
    # Prepare data structure for color sampling
    all_boxes = {} # {class_id: [box1_coords, box2_coords, ...]}
    
    for box in detections:
        class_id = int(box.cls[0])
        # Only process Pad (0-9) and Reference (10-19) classes
        if 0 <= class_id <= 19:
            if class_id not in all_boxes:
                all_boxes[class_id] = []
            all_boxes[class_id].append(box.xyxy[0])
            
    final_readings = {}

    # 2. Match Pads to References
    logger.info("Starting color matching (Delta E) analysis...")
    
    for pad_id, ref_id in TEST_MAP.items():
        if pad_id not in all_boxes or ref_id not in all_boxes:
            logger.warning(f"Missing detections for Pad {pad_id} or Ref {ref_id}. Skipping.")
            continue
            
        # 2a. Sample Pad Color (Only one pad exists per test, so take the first box)
        pad_box = all_boxes[pad_id][0]
        pad_lab = sample_color_lab(img_bgr, pad_box)
        
        if pad_lab is None: continue

        min_delta_e = float('inf')
        best_match_value = "UNDETECTED"
        
        # 2b. Sample ALL Reference Squares for this test
        # We assume the reference boxes are *already sorted* by your relabeling process,
        # so we can directly map the array index (0, 1, 2, ...) to the REFERENCE_VALUES list.
        
        ref_boxes = all_boxes[ref_id]
        
        # Sort the reference boxes by X-coordinate to ensure index 0 maps to Value 0, etc.
        # This is a safety measure against inconsistent detection order.
        ref_boxes.sort(key=lambda box: box[0]) 

        for i, ref_box in enumerate(ref_boxes):
            ref_lab = sample_color_lab(img_bgr, ref_box)
            if ref_lab is None: continue
            
            # Calculate Color Distance
            delta_e = calculate_delta_e(pad_lab, ref_lab)
            
            # Check for the best match
            if delta_e < min_delta_e:
                min_delta_e = delta_e
                # Map the index (i) to the actual reference value (e.g., "100" or "++")
                if i < len(REFERENCE_VALUES.get(ref_id, [])):
                    best_match_value = REFERENCE_VALUES[ref_id][i]
                else:
                    best_match_value = f"ID {i} (Unknown Value)"

        # 3. Store Result
        pad_name = model.names[pad_id].replace("pad_", "").upper()
        final_readings[pad_name] = f"{best_match_value} (ΔE: {min_delta_e:.2f})"

    logger.info("\n--- FINAL DIPSTICK READING ---")
    for test, result in final_readings.items():
        print(f"{test.ljust(15)}: {result}")
    logger.info("----------------------------")

    return final_readings
