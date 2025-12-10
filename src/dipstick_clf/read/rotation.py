import os
import shutil
import torch
from ultralytics import YOLO #type:ignore
from loguru import logger
from src.config import MODELS_DIR, DIPSTICK_IMAGES_DIR
from src.dipstick_clf.preprocessing.utils import rotate_image

IMAGES_DIR = DIPSTICK_IMAGES_DIR / "images_raw/test"

# CONFIGURATION
ROTATION_MODEL_PATH = MODELS_DIR / "dipstick_read/rotation_03/weights/best.pt"
DEVICE = 'cpu' if not torch.backends.mps.is_available() else 'mps'
rotation_model = YOLO(ROTATION_MODEL_PATH)

def correct_rotation(image_path, model):
    """
    Uses a YOLO model to find the 'hand annotation' and determines the
    rotation correction needed to move it to the bottom-right corner.

    Args:
        model (YOLO): Loaded Ultralytics model.
        image_path (Path or str): Path to image.
        conf_threshold (float): Confidence threshold for detection.
    """
    # Run inference
    results = model.predict(image_path, iou=0.7, conf=0.5, verbose=False,  device=DEVICE)[0]

    # Filter for hand boxes (class 1)
    hand_boxes = [box for box in results.boxes if int(box.cls[0]) == 1]

    # If no hand detected, return None
    if not hand_boxes:
        logger.error("Hand not detected. Please check image.")
        return None

    # Get hand with highest confidence
    best_box = max(results.boxes, key=lambda x: x.conf[0])

    # Get box coordinates and calculate center coordinates
    x1, y1, x2, y2 = best_box.xyxy[0].tolist()
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    # Get image dimensions
    height, width = results.orig_shape

    # Clockwise rotation based on hand position
    if x_center > width/2 and y_center > height/2:
        rotation = 0
    elif x_center <= width/2 and y_center > height/2:
        rotation = 90
    elif x_center <= width/2 and y_center <= height/2:
        rotation = 180
    else:
        rotation = 270

    #rotate_image(image_path, rotation)

    print(f"rotation: {rotation}, img height: {height}, img width: {width}")
    print(best_box)

def correct_rotation2(image_path, model):
    """
    Uses a YOLO model to find the 'strip annotation' and determines the
    rotation correction needed to move it to the left of frame.

    Args:
        model (YOLO): Loaded Ultralytics model.
        image_path (Path or str): Path to image.
        conf_threshold (float): Confidence threshold for detection.

    Returns:
        int or None: PIL.Image.Transpose constant (ROTATE_90, etc.) or None if 0°.
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

    print(f"rotation: {rotation}, img height: {height}, img width: {width}")
    #print(strip_boxes)


def main():
    """Main function to read dipstick images."""
    # Copy images
    output_dir = IMAGES_DIR.parents[2] / "images"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    shutil.copytree(IMAGES_DIR, output_dir)

    for img_path in output_dir.iterdir():
        logger.info(f"Reading dipstick from image: {img_path}")
        correct_rotation2(img_path, rotation_model)

if __name__ == '__main__':
    main()
