"""YOLOv8 Training Script for Dipstick Detection"""

# Import libraries
import torch
from ultralytics import YOLO # type: ignore
from loguru import logger
from src.config import RAW_DATA_DIR, MODELS_DIR

# Dataset paths
IMG_DIR = RAW_DATA_DIR / "dipstick_imgs_simple"
DATA_CONFIG = IMG_DIR / "dipstick.yaml"

# Model paths
MODELS_DIR = MODELS_DIR / "dipstick_detection5"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

# Training parameters
MODEL_ARCH = PRETRAINED_DIR / "yolov8m.pt"
EPOCHS = 200
IMG_SIZE = 1000
BATCH_SIZE = 8
LEARNING_RATE = 0.01
PATIENCE = 20

torch.cuda.empty_cache()

def train():
    """Train YOLOv8 model for dipstick detection."""
    logger.info("Starting YOLOv8 training for dipstick detection...")

    # Load model
    model = YOLO(MODEL_ARCH)
    # Use CUDA if available
    if torch.cuda.is_available():
        device = "cuda"
        model.to(device)

    # Train
    results = model.train(
        data=str(DATA_CONFIG),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        lr0=LEARNING_RATE,
        patience=PATIENCE,
        mosaic=0,        # disable mosaic for color fidelity
        hsv_h=0.0,       # disable hue jitter
        hsv_s=0.2,       # mild saturation jitter
        hsv_v=0.2,       # mild brightness jitter
        degrees=7,       # small rotations
        translate=0.05,  # small translations
        scale=0.05,      # small scaling
        shear=0.0,
        perspective=0.02, # mild perspective warp
        project=str(MODELS_DIR),
        name="yolov8_dipstick_simple"
    )

    logger.success("Training complete.")
    logger.info(f"Results saved to: {results.save_dir}") #type: ignore

if __name__ == "__main__":
    train()
