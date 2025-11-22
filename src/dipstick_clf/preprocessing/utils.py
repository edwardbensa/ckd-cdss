"""Utility functions for preprocessing dipstick dataset."""

import os
import json
import shutil
from pathlib import Path
from loguru import logger
from PIL import Image
from src.config import DIPSTICK_IMAGES_DIR

# --- RELABELING MAPPINGS AND FUNCTIONS ---

# CLASS MAPPING
PLAIN_CLASS_MAP = {
    # Pad class
    "pad": 0,
    # Reference square classes (10)
    "urobilinogen": 1, "glucose": 2, "bilirubin": 3, "ketones": 4, "SG": 5,
    "blood": 6, "ph": 7, "protein": 8, "nitrite": 9, "leukocytes": 10
}

SIMPLE_CLASS_MAP = {
    # Pad classes (10)
    "pad_urobilinogen": 0, "pad_glucose": 1, "pad_bilirubin": 2, "pad_ketones": 3,
    "pad_SG": 4, "pad_blood": 5, "pad_ph": 6, "pad_protein": 7, "pad_nitrite": 8,
    "pad_leukocytes": 9,
    # Reference square classes (10)
    "urobilinogen": 10, "glucose": 11, "bilirubin": 12, "ketones": 13, "SG": 14,
    "blood": 15, "ph": 16, "protein": 17, "nitrite": 18, "leukocytes": 19
}

GRANULAR_CLASS_MAP = {
    # Pad classes (10)
    "pad_urobilinogen": 0, "pad_glucose": 1, "pad_bilirubin": 2, "pad_ketones": 3,
    "pad_SG": 4, "pad_blood": 5, "pad_ph": 6, "pad_protein": 7, "pad_nitrite": 8,
    "pad_leukocytes": 9,
    # Reference square classes (51)
    "urobilinogen1": 10, "urobilinogen2": 11, "urobilinogen3": 12, "urobilinogen4": 13, "urobilinogen5": 14,
    "glucose1": 15, "glucose2": 16, "glucose3": 17, "glucose4": 18, "glucose5": 19,
    "bilirubin1": 20, "bilirubin2": 21, "bilirubin3": 22, "bilirubin4": 23,
    "ketones1": 24, "ketones2": 25, "ketones3": 26, "ketones4": 27, "ketones5": 28,
    "SG1": 29, "SG2": 30, "SG3": 31, "SG4": 32, "SG5": 33, "SG6": 34, "SG7": 35,
    "blood1": 36, "blood2": 37, "blood3": 38, "blood4": 39, "blood5": 40, "blood6": 41,
    "ph1": 42, "ph2": 43, "ph3": 44, "ph4": 45, "ph5": 46, "ph6": 47,
    "protein1": 48, "protein2": 49, "protein3": 50, "protein4": 51, "protein5": 52, "protein6": 53,
    "nitrite1": 54, "nitrite2": 55, "nitrite3": 56,
    "leukocytes1": 57, "leukocytes2": 58, "leukocytes3": 59, "leukocytes4": 60
}

ROTATION_CLASS_MAP = {
    # Utility classes
    "strip": 0, "hand":1
}

# Helper map for relabeling pads
PAD_INDEX_TO_TEST = {
    0: "urobilinogen", 1: "glucose", 2: "bilirubin", 3: "ketones", 4: "SG",
    5: "blood", 6: "ph", 7: "protein", 8: "nitrite", 9: "leukocytes"
}

def determine_rotation(image, category_map, anns):
    """Determine image rotation based on hand annotation position."""
    width = image["width"]
    height = image["height"]

    hand_ann = next((a for a in anns if a["category_id"] == category_map.get("hand")), None)
    if hand_ann is None:
        logger.warning(f"No hand annotation found for image id {image['id']}. Defaulting to 0°.")
        return 0

    x_min, y_min = hand_ann["bbox"][:2] #type: ignore

    # clockwise rotation based on hand position
    if x_min > width/2 and y_min > height/2:
        return 0
    elif x_min <= width/2 and y_min > height/2:
        return 90
    elif x_min <= width/2 and y_min <= height/2:
        return 180
    else:
        return 270

def rotate_anns(image, anns, rotation):
    """
    Rotates COCO bounding box annotations (x_min, y_min, w, h) to match a 
    counterclockwise image rotation applied to correct a clockwise rotation.
    """
    if rotation == 0:
        return anns

    W, H = image["width"], image["height"]
    corrected_anns = []

    # Update image dimensions for 90/270 rotations
    if rotation in [90, 270]:
        image["width"], image["height"] = H, W

    # Correct bounding boxes
    for ann in anns:
        x, y, w, h = ann["bbox"]

        # Initialize new bbox with original values
        x_new, y_new, w_new, h_new = x, y, w, h

        if rotation == 90:
            # 90° Clockwise original means 90° CCW correction is needed.
            # (x, y) -> (y, W - x - w)
            x_new = y
            y_new = W - x - w
            w_new = h
            h_new = w

        elif rotation == 180:
            # 180° Clockwise original means 180° CCW correction is needed.
            # (x, y) -> (W - x - w, H - y - h)
            x_new = W - x - w
            y_new = H - y - h
            w_new = w
            h_new = h

        elif rotation == 270:
            # 270° Clockwise original means 270° CCW correction is needed.
            # (x, y) -> (H - y - h, x)
            x_new = H - y - h
            y_new = x
            w_new = h
            h_new = w

        else:
            logger.error(f"Invalid rotation angle encountered: {rotation}")

        new_ann = ann.copy()
        new_ann["bbox"] = [x_new, y_new, w_new, h_new]
        corrected_anns.append(new_ann)

    return corrected_anns

def relabel_pads(category_map, anns, relabel_method):
    """Select, sort and relabel the dipstick pad annotations."""
    relabeled_anns = []

    selected_anns = [a.copy() for a in anns if a["category_id"] == category_map.get("stripe_color")]
    sorted_anns = sorted(selected_anns, key=lambda a: a["bbox"][1])

    for i, ann in enumerate(sorted_anns):
        if relabel_method == "plain":
            new_name = "pad"
            ann["category_id"] = PLAIN_CLASS_MAP[new_name]
        elif relabel_method == "simple":
            new_name = f"pad_{PAD_INDEX_TO_TEST[i]}"
            ann["category_id"] = SIMPLE_CLASS_MAP[new_name]
        else: # relabel_method == "granular":
            new_name = f"pad_{PAD_INDEX_TO_TEST[i]}"
            ann["category_id"] = GRANULAR_CLASS_MAP[new_name]
        relabeled_anns.append(ann)

    return relabeled_anns

def relabel_references(category_map, anns, relabel_method):
    """Select, sort and relabel the dipstick pad annotations."""
    tests = list(PAD_INDEX_TO_TEST.values())
    relabeled_anns = []

    for test in tests:
        selected_anns = [a.copy() for a in anns if a["category_id"] == category_map.get(test)]
        sorted_anns = sorted(selected_anns, key=lambda a: a["bbox"][0])

        for i, ann in enumerate(sorted_anns):
            if relabel_method == "plain":
                new_name = test
                ann["category_id"] = PLAIN_CLASS_MAP[new_name]
            elif relabel_method == "simple":
                new_name = test
                ann["category_id"] = SIMPLE_CLASS_MAP[new_name]
            else: # relabel_method == "granular":
                new_name = f"{test}{i+1}"
                ann["category_id"] = GRANULAR_CLASS_MAP[new_name]
            relabeled_anns.append(ann)

    return relabeled_anns

def relabel_utils(category_map, anns):
    """Select, sort and relabel the dipstick pad annotations."""
    utils = list(ROTATION_CLASS_MAP.keys())
    relabeled_anns = []

    for util in utils:
        selected_anns = [a.copy() for a in anns if a["category_id"] == category_map.get(util)]

        for ann in selected_anns:
            ann["category_id"] = ROTATION_CLASS_MAP[util]
            relabeled_anns.append(ann)

    return relabeled_anns


def generate_anns(image, anns, category_map, relabel_method):
    """Generate relabelled annotations for pads and reference squares in an image."""

    # Rotations
    rotation = determine_rotation(image, category_map, anns)
    rotated_anns = rotate_anns(image, anns, rotation)
    new_anns = []

    if relabel_method != "rotation":
        # Relabel pads
        pad_anns = relabel_pads(category_map, rotated_anns, relabel_method)
        new_anns.extend(pad_anns)

        # Relabel reference squares
        ref_anns = relabel_references(category_map, rotated_anns, relabel_method)
        new_anns.extend(ref_anns)
    else:
        # Relabel utility classes
        util_anns = relabel_utils(category_map, anns)
        new_anns.extend(util_anns)

    return new_anns

def update_categories(relabel_method):
    """Generate new categories list based on relabel method."""
    class_maps = {
        "plain": PLAIN_CLASS_MAP,
        "simple": SIMPLE_CLASS_MAP,
        "granular": GRANULAR_CLASS_MAP,
        "rotation": ROTATION_CLASS_MAP
    }

    new_categories = []
    for name, new_id in class_maps[relabel_method].items():
        if relabel_method != "rotation":
            name = name[:-1] if relabel_method == "granular" else name
            new_categories.append({
                "id": new_id,
                "name": name,
                "supercategory": "pads" if name.startswith("pad")
                else "reference"
                })
        else:
            new_categories.append({
                "id": new_id,
                "name": name,
                "supercategory": "utility"
                })

    return new_categories

def modify_annotations(directory, split: str, relabel_method: str):
    """
    Rotate and relabel pads and reference squares in COCO annotations for a split
    to match the specified relabel method.
    """
    ann_path = directory / split / "labels" / "_annotations.coco.json"
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Build category maps
    categories = coco["categories"]
    for cat in categories:
        if cat["name"] == "stripes":
            cat["name"] = "strip"
    category_map = {c["name"]: c["id"] for c in categories}

    # Create new annotations list for images
    new_annotations = []
    for img in coco["images"]:
        anns = [a for a in coco["annotations"] if a["image_id"] == img["id"]]
        new_anns = generate_anns(img, anns, category_map, relabel_method=relabel_method)
        new_annotations.extend(new_anns)
        logger.info(f"Processed image id {img['id']} with {len(anns)} annotations.")
    for i, ann in enumerate(new_annotations):
        ann["id"] = i
    coco["annotations"] = new_annotations

    # Update categories
    coco["categories"] = update_categories(relabel_method)

    # Save updated JSON
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)
    logger.success(f"Modified {split} split with {relabel_method} method. Saved to {ann_path}")

def rotate_images(directory, split: str, relabel_method):
    """
    Rotate images to 0 degrees if rotation is 90, 180, or 270 clockwise.
    """
    if relabel_method == "rotation":
        logger.info("Relabel method is 'rotation'. No image rotation necessary")
    else:
        img_dir = directory / split / "images"

        ann_path = directory / split / "labels" / "_annotations.coco.json"
        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # Build category maps
        categories = coco["categories"]
        category_map = {c["name"]: c["id"] for c in categories}

        # Determine image rotation and rotate images
        for img_ann in coco["images"]:
            img_file = img_dir / img_ann["file_name"]
            img = Image.open(img_file)
            anns = [a for a in coco["annotations"] if a["image_id"] == img_ann["id"]]
            rotation = determine_rotation(img_ann, category_map, anns)

            # Map clockwise rotations to counterclockwise corrections
            rotation_map = {
                0: None,
                90: Image.Transpose.ROTATE_90,
                180: Image.Transpose.ROTATE_180,
                270: Image.Transpose.ROTATE_270
            }

            if rotation not in rotation_map:
                logger.error(f"Invalid rotation value: {rotation}")
                raise ValueError("Rotation must be one of [0, 90, 180, 270]")

            if rotation != 0:
                logger.debug(f"Rotated {img_ann["file_name"]} {rotation}° counterclockwise.")
                img = img.transpose(rotation_map[rotation])
                img.save(img_file)
            else:
                logger.info(f"No rotation needed for {img_ann["file_name"]}.")
        logger.success("Image rotations complete.")


def create_folders(directory, relabel_method):
    """Makes version of image folder for selected relabelling method."""

    # Refresh DIPSTICK_IMAGES_DIR
    if os.path.exists(DIPSTICK_IMAGES_DIR):
        shutil.rmtree(DIPSTICK_IMAGES_DIR)
    os.mkdir(DIPSTICK_IMAGES_DIR)

    relabel_methods = [relabel_method, "rotation"]
    new_dirs = []

    # Create dataset copies for rotation and selected relabeling method
    for method in relabel_methods:
        output_dir = DIPSTICK_IMAGES_DIR / f"imgs_{method}"
        new_dir = shutil.copytree(directory, output_dir)
        new_dirs.append(new_dir)
        logger.success(f"Image dataset created for {method} relabeling.")

    return relabel_methods, new_dirs

def yolo_folders(directory: Path, split: str):
    """Restructure a split into YOLO format: images/, labels/."""
    split_dir = directory / split
    if not split_dir.exists():
        logger.warning(f"Split {split} not found at {split_dir}")
        return

    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    # Move annotation and image files into appropriate directories
    for file in split_dir.glob("*.*"):
        if file.name.startswith("_annotations"):
            dest = labels_dir / file.name
            shutil.move(str(file), str(dest))
            logger.info(f"Moved annotation to {dest}")
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            dest = images_dir / file.name
            if not dest.exists():
                shutil.move(str(file), str(dest))



def coco_to_yolo(coco_json_path: Path, labels_dir: Path):
    """Convert COCO annotations to YOLO TXT format with proper id→index mapping."""
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Build mapping from COCO category_id → YOLO index (0..nc-1)
    categories = coco["categories"]
    id_to_index = {c["id"]: i for i, c in enumerate(categories)}

    # Map image_id to image info
    images = {img["id"]: img for img in coco["images"]}

    # Clear existing .txt files
    for txt_file in labels_dir.glob("*.txt"):
        txt_file.unlink()

    ann_count = 0
    for ann in coco["annotations"]:
        img_info = images[ann["image_id"]]
        img_w, img_h = img_info["width"], img_info["height"]
        x, y, w, h = ann["bbox"]

        # Convert to YOLO format (normalized)
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        # Remap category_id to YOLO index
        if ann["category_id"] not in id_to_index:
            logger.warning(f"Annotation {ann['id']} has unknown category_id {ann['category_id']}")
            continue
        class_id = id_to_index[ann["category_id"]]

        # Write to per-image label file
        img_filename = Path(img_info["file_name"]).stem
        label_file = labels_dir / f"{img_filename}.txt"
        with open(label_file, "a", encoding="utf-8") as lf:
            lf.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        ann_count += 1

    logger.success(f"Converted {ann_count} annotations from {coco_json_path} -> YOLO TXT in {labels_dir}")

def generate_yolo_txt(directory: Path, split: str):
    """Convert a split's COCO JSON to YOLO TXT files."""
    coco_json = directory / split / "labels" / "_annotations.coco.json"
    labels_dir = directory / split / "labels"
    if not coco_json.exists():
        logger.error(f"COCO JSON not found at {coco_json}")
        return
    coco_to_yolo(coco_json, labels_dir)

def generate_yolo_yaml(output_dir: Path, relabel_method: str = "simple"):
    """
    Generate a YOLO dataset YAML in output_dir using the chosen relabel_method.
    """
    output_dir = Path(output_dir)
    class_maps = {
        "plain": PLAIN_CLASS_MAP,
        "simple": SIMPLE_CLASS_MAP,
        "granular": GRANULAR_CLASS_MAP,
        "rotation": ROTATION_CLASS_MAP
    }

    yaml_name = "dipstick.yaml"
    yaml_path = output_dir / yaml_name

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# YOLOv8 dataset config for dipstick detection\n")
        f.write("\n")
        f.write("# Paths\n")
        f.write(f"path: {output_dir}\n")
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        f.write("test: test/images\n")
        f.write("\n")
        f.write("# Classes\n")
        f.write("names:\n")

        for name, idx in class_maps[relabel_method].items():
            f.write(f"  {idx}: {name}\n")

    logger.success(f"Generated YOLO YAML at {yaml_path} ({relabel_method} classes)")
