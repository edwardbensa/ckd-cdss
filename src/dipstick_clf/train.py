"""YOLOv8 training script for dipstick detection"""

# Import libraries
import torch
from ultralytics import YOLO # type: ignore
from loguru import logger
from src.config import DIPSTICK_IMAGES_DIR, MODELS_DIR

RELABEL_METHOD = "rotation"

# Dataset paths
IMG_DIR = DIPSTICK_IMAGES_DIR / f"imgs_{RELABEL_METHOD}"
DATA_CONFIG = IMG_DIR / "dipstick.yaml"

# Dipstck models path
DIPSTICK_MODELS_DIR = MODELS_DIR / "dipstick_read"
DIPSTICK_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Count models per relabel method and name model folder
current_model_count = sum(
    1 for entry in DIPSTICK_MODELS_DIR.iterdir()
    if entry.is_dir() and entry.name.startswith(RELABEL_METHOD)
)
model_index = str(current_model_count+1).zfill(2)
MODEL_NAME = f"{RELABEL_METHOD}_{model_index}"

# Run path
MODEL_DIR = DIPSTICK_MODELS_DIR / MODEL_NAME
PRETRAINED_DIR = MODEL_DIR / "pretrained"
PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

# Training parameters
MODEL_ARCH = PRETRAINED_DIR / "yolov8m.pt"
EPOCHS = 200
IMG_SIZE = 1280
BATCH_SIZE = 8
LEARNING_RATE = 0.01
PATIENCE = 20

def train():
    """Train YOLOv8 model for dipstick detection."""
    logger.info("Starting YOLOv8 training for dipstick detection...")
    logger.info(f"Model run directory: {MODEL_DIR}")

    # Load model
    model = YOLO(MODEL_ARCH)

    # Select device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Device selected: {device}")

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
        perspective=0.02,
        device=device, # mild perspective warp
        project=str(MODEL_DIR),
        name=f"yolov8_dipstick_{RELABEL_METHOD}"
    )

    logger.success("Training complete.")
    logger.info(f"Results saved to: {results.save_dir}") #type: ignore

if __name__ == "__main__":
    train()
