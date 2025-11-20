"""YOLO preprocessing pipeline for dipstick image dataset."""


from pathlib import Path
from loguru import logger
from src.config import RAW_DATA_DIR
from src.dipstick_clf.preprocessing.utils import (yolo_folders, generate_yolo_txt,
                                                  relabel_annotations, generate_yolo_yaml)

IMG_DIR = RAW_DATA_DIR / "dipstick_imgs_simple"

def main(dataset_dir: Path, relabel_method: str = "simple"):
    """Run preprocessing pipeline: prepare, relabel, convert."""
    if relabel_method not in ["simple", "granular"]:
        raise ValueError("relabel_method must be 'simple' or 'granular'")

    for split in ["train", "valid", "test"]:
        # Prepare folder structure
        yolo_folders(dataset_dir, split)
        logger.success(f"Prepared YOLO folder structure for {split} split.")

        # Relabel pads
        if relabel_method == "granular":
            relabel_annotations(dataset_dir, split, relabel_method="granular")
            logger.success(f"Relabelled pads in {split} split using granular method.")
        else:
            relabel_annotations(dataset_dir, split, relabel_method="simple")
            logger.success(f"Relabelled pads in {split} split using simple method.")

        # Generate YOLO TXT files
        generate_yolo_txt(dataset_dir, split)
        logger.success(f"Converted {split} split to YOLO format.")

    # Generate YOLO YAML config
    generate_yolo_yaml(dataset_dir, relabel_method=relabel_method)

    logger.success("Preprocessing pipeline completed.")

if __name__ == "__main__":
    main(IMG_DIR, relabel_method="simple")
