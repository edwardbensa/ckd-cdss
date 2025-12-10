"""YOLO preprocessing pipeline for dipstick image dataset."""


from loguru import logger
from src.dipstick_clf.preprocessing.utils import (copy_images, copy_labels,
                                                  modify_annotations, generate_yolo_txt,
                                                  generate_yolo_yaml)
from src.config import RAW_DATA_DIR

IMG_DIR = RAW_DATA_DIR / "imgs"

def main():
    """Run preprocessing pipeline: prepare, relabel, convert."""
    relabel_methods = ["plain", "simple", "granular", "rotation"]
    splits = ["test", "train", "valid"]

    # Copy images
    for split in splits:
        copy_images(IMG_DIR, split)
        logger.success("Images copied.")

    # Copy and modify annotations
    for relabel_method in relabel_methods:
        for split in splits:
            dataset_dir = copy_labels(IMG_DIR, split, relabel_method)
            logger.success(f"Labels copied for {relabel_method} split.")

            # Modify annotations
            modify_annotations(dataset_dir, split, relabel_method)

            # Generate YOLO TXT files
            generate_yolo_txt(dataset_dir)
            logger.success(f"Converted {split} split to YOLO format.")

        # Generate YOLO YAML config
        generate_yolo_yaml(dataset_dir, relabel_method)

    logger.success("Preprocessing pipeline completed.")

if __name__ == "__main__":
    main()
