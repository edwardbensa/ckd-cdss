"""Utility functions for training dipstick object identification model."""

# Imports
import os
import shutil
from pathlib import Path
from loguru import logger
from ultralytics import YOLO # type: ignore
import torch

from src.dipstick_clf.preprocessing.utils import rotate_images

# Load YOLO model and determine PyTorch device
def load_yolo(pretrained_dir, model_name):
    """
    Loads YOLO model.
    Selects CUDA or MPS for model training if they're available.
    """

    # Load model
    model_path = pretrained_dir / f"{model_name}.pt"
    model = YOLO(str(model_path))
    logger.info(f"{model_name} model loaded.")

    # Select device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Device selected: {device}")

    return model, device

# Generate YOLO run name
def generate_run_name(label_method, models_dir):
    """Determine the run name (and iteration) for a given label method."""
    models_dir = Path(models_dir)

    # Count models per label method and name model folder
    current_model_count = sum(
        1 for entry in models_dir.iterdir()
        if entry.is_dir() and entry.name.startswith(label_method)
    )
    model_index = str(current_model_count+1).zfill(2)
    run_name = f"{label_method}_{model_index}"

    return run_name

# Symlink
def setup_symlink(link_name, source_name, data_dir):
    """
    Generic function to create a symlink.
    """
    target_link = data_dir / link_name
    source_dir = data_dir / source_name

    # Validation
    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder not found: {source_dir}")

    # Cleanup existing link or folder
    if target_link.exists() or target_link.is_symlink():
        if target_link.is_symlink():
            target_link.unlink()
            logger.debug(f"Removed existing symlink: {link_name}")
        elif target_link.is_dir():
            # If a real directory exists with the target name, back it up to avoid data loss
            backup_name = f"{link_name}_backup_found_during_runtime"
            logger.warning(f"Found real directory '{link_name}'. Renaming to '{backup_name}'.")
            target_link.rename(data_dir / backup_name)

    # Create Symlink
    try:
        os.symlink(source_dir, target_link)
        logger.info(f"Linked: {link_name} -> {source_name}")
    except OSError as e:
        logger.error(f"Failed to create symlink. If on Windows, turn on Developer Mode. Error: {e}")
        raise

def prepare_dataset(label_method, data_dir):
    """
    Orchestrates the symlinks for the training run.
    """
    # Copy labels
    input_dir = data_dir / f"labels_{label_method}"
    output_dir = data_dir / "labels"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    shutil.copytree(input_dir, output_dir)

    # Copy images
    input_dir = data_dir / "images_raw"
    output_dir = data_dir / "images"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    shutil.copytree(input_dir, output_dir)

    # Rotate images if label method is not 'rotation'
    if label_method != "rotation":
        rotate_images(data_dir, naive=False)


def cleanup(data_dir):
    """Remove temporary folders created for training."""
    for i in ["images", "labels"]:
        target_link = data_dir / i
        if target_link.is_symlink():
            target_link.unlink()
            logger.info(f"Cleanup: Removed temporary {i} symlink.")
        shutil.rmtree(data_dir / i)
