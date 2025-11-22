"""YOLO preprocessing pipeline for dipstick image dataset."""


from loguru import logger
from src.dipstick_clf.preprocessing.utils import (create_folders, yolo_folders, rotate_images,
                                                  modify_annotations, generate_yolo_txt,
                                                  generate_yolo_yaml)
from src.config import RELABEL_METHOD

IMG_DIR = "/Users/Edward/Downloads/imgs"

def main():
    """Run preprocessing pipeline: prepare, relabel, convert."""
    if RELABEL_METHOD not in ["plain", "simple", "granular"]:
        raise ValueError("relabel_method must be 'plain', 'simple', or 'granular'.")
    logger.info("Creating folders for relabeling methods...")
    relabel_methods, new_dirs = create_folders(IMG_DIR, RELABEL_METHOD)

    for dataset_dir, relabel_method in zip(new_dirs, relabel_methods):
        logger.info(f"Creating {relabel_method} dataset...")
        for split in ["test", "train", "valid"]:
            # Prepare folder structure
            yolo_folders(dataset_dir, split)
            logger.success(f"Prepared YOLO folder structure for {split} split.")

            # Rotate images
            logger.info("Rotating images...")
            rotate_images(dataset_dir, split, relabel_method)

            # Modify annotations
            modify_annotations(dataset_dir, split, relabel_method)

            # Generate YOLO TXT files
            generate_yolo_txt(dataset_dir, split)
            logger.success(f"Converted {split} split to YOLO format.")

        # Generate YOLO YAML config
        generate_yolo_yaml(dataset_dir, relabel_method=relabel_method)

    logger.success("Preprocessing pipeline completed.")

if __name__ == "__main__":
    main()
