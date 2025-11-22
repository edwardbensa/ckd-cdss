"""Dipstick reading main script."""

# Imports
import json
from loguru import logger
from src.dipstick_clf.read.utils import read_dipstick
from src.config import RAW_DATA_DIR, REPORTS_DIR

IMAGES_DIR = RAW_DATA_DIR / "dipstick_imgs_simple/test/images"
RESULTS_DIR = REPORTS_DIR / "dipstick_read"
dipstick_results = []

def main():
    """Main function to read dipstick images."""
    for img_path in IMAGES_DIR.iterdir():
        logger.info(f"Reading dipstick from image: {img_path}")
        results = read_dipstick(img_path)
        dipstick_results.append({
            "image": img_path.name,
            "results": results
        })

    # Save results to JSON
    results_path = RESULTS_DIR / "dipstick_read_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(dipstick_results, f, indent=4)
    logger.success(f"Dipstick reading results saved to {results_path}")

if __name__ == '__main__':
    main()
