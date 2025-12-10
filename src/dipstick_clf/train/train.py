"""YOLO training script for dipstick detection"""

# Import libraries
from loguru import logger
from src.dipstick_clf.train.utils import prepare_dataset, load_yolo, generate_run_name, cleanup
from src.config import DIPSTICK_IMAGES_DIR, MODELS_DIR

LABEL_METHOD = "rotation"
YOLO_MODEL = "yolo11m"

# Paths
DATA_DIR = DIPSTICK_IMAGES_DIR
MODELS_DIR = MODELS_DIR / "dipstick_read"
PRETRAINED_MODELS_DIR = MODELS_DIR / "_pretrained-models"

# Training parameters
EPOCHS = 50
IMG_SIZE = 736
PATIENCE = 10
BATCH_SIZE = 8


def train():
    """Train YOLO model for dipstick detection."""

    # Prepare dataset
    prepare_dataset(LABEL_METHOD, DATA_DIR)
    yaml_path = DATA_DIR / f"dipstick_{LABEL_METHOD}.yaml"

    # Load model and select device
    model, device = load_yolo(PRETRAINED_MODELS_DIR, YOLO_MODEL)

    # Define output project and name
    project_dir = MODELS_DIR
    run_name = generate_run_name(LABEL_METHOD, MODELS_DIR)

    logger.info(f"Starting training for variant: {LABEL_METHOD}")

    try:
        # Train
        results = model.train(
            data=str(yaml_path),
            project=str(project_dir),
            name=run_name,
            device=device,
            batch=BATCH_SIZE,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            patience=PATIENCE,
            save=True,
            exist_ok=True # Overwrite existing run folder if same name
        )
        logger.success("Training complete.")
        logger.info(f"Results saved to: {results.save_dir}") #type: ignore
    finally:
        # Cleanup symlinks
        cleanup(DATA_DIR)
        logger.success("Cleanup complete.")

if __name__ == "__main__":
    train()
