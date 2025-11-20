"""Utility functions for preprocessing dipstick dataset."""

import json
import shutil
from pathlib import Path
from loguru import logger

# --- RELABELING MAPPINGS AND FUNCTIONS ---

# CLASS MAPPING
PLAIN_CLASS_MAP = {
    # Pad class
    "pad": 0,
    # Reference square classes (10)
    "urobilinogen": 1, "glucose": 2, "bilirubin": 3, "ketones": 4, "SG": 5,
    "blood": 6, "ph": 7, "protein": 8, "nitrite": 9, "leukocytes": 10,
    # Utility classes
    "hand": 11
}

SIMPLE_CLASS_MAP = {
    # Pad classes (10)
    "pad_urobilinogen": 0, "pad_glucose": 1, "pad_bilirubin": 2, "pad_ketones": 3,
    "pad_SG": 4, "pad_blood": 5, "pad_ph": 6, "pad_protein": 7, "pad_nitrite": 8,
    "pad_leukocytes": 9,
    # Reference square classes (10)
    "urobilinogen": 10, "glucose": 11, "bilirubin": 12, "ketones": 13, "SG": 14,
    "blood": 15, "ph": 16, "protein": 17, "nitrite": 18, "leukocytes": 19,
    # Utility classes
    "hand": 20
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
    "leukocytes1": 57, "leukocytes2": 58, "leukocytes3": 59, "leukocytes4": 60,
    # Utility classes
    "hand": 61
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

def select_anns(category_map, category_name, anns, ann_type):
    """Select annotations of a given type (pad, reference, or hand)."""
    if ann_type == "pad":
        selected_anns = [a for a in anns if a["category_id"] == category_map.get("stripe_color")]
    elif ann_type == "reference":
        selected_anns = [a for a in anns if a["category_id"] == category_map.get(category_name)]
    elif ann_type == "hand":
        selected_anns = [a for a in anns if a["category_id"] == category_map.get("hand")]
    else:
        raise ValueError("ann_type must be 'pad', 'reference', or 'hand'")
    return selected_anns

def sort_anns(selected_anns, rotation, ann_type):
    """Sort selected annotations based on image rotation and annotation type."""
    if ann_type not in ["pad", "reference"]:
        raise ValueError("ann_type must be 'pad' or 'reference'")

    pad_sort_rules = {
        0:  (lambda a: a["bbox"][1], False),   # y_min
        90: (lambda a: a["bbox"][0], True),    # x_min desc
        180:(lambda a: a["bbox"][1], True),    # y_min desc
        270:(lambda a: a["bbox"][0], False)    # x_min
    }

    ref_sort_rules = {
        0:  (lambda a: a["bbox"][0], False),   # x_min
        90: (lambda a: a["bbox"][1], True),    # y_min desc
        180:(lambda a: a["bbox"][0], True),    # x_min desc
        270:(lambda a: a["bbox"][1], False)    # y_min
    }

    sort_rules = pad_sort_rules if ann_type == "pad" else ref_sort_rules
    key_fn, reverse_flag = sort_rules[rotation]

    sorted_anns = sorted(selected_anns, key=key_fn, reverse=reverse_flag)
    return sorted_anns

def relabel_anns(sorted_anns, category_name, ann_type, relabel_method):
    """Relabel sorted annotations using class maps based on relabel method."""
    if relabel_method not in ["simple", "granular"]:
        raise ValueError("relabel_method must be 'simple' or 'granular'")

    relabeled_anns = []
    for i, ann in enumerate(sorted_anns):
        new_ann = ann.copy()
        if ann_type == "pad":
            if i in PAD_INDEX_TO_TEST:
                new_name = f"pad_{PAD_INDEX_TO_TEST[i]}"
                new_ann["category_id"] = SIMPLE_CLASS_MAP[new_name]
        elif ann_type == "reference" and relabel_method == "simple":
            new_ann["category_id"] = SIMPLE_CLASS_MAP[category_name]
        elif ann_type == "reference" and relabel_method == "granular":
            new_name = f"{category_name}{i+1}"
            new_ann["category_id"] = GRANULAR_CLASS_MAP[new_name]
        else:
            raise ValueError("ann_type must be 'pad' or 'reference'")
        relabeled_anns.append(new_ann)

    return relabeled_anns

def generate_anns(image, anns, category_map, relabel_method):
    """Generate relabelled annotations for pads and reference squares in an image."""
    rotation = determine_rotation(image, category_map, anns)
    new_anns = []

    # Relabel pads
    pad_anns = select_anns(category_map, "", anns, "pad")
    sorted_pads = sort_anns(pad_anns, rotation, "pad")
    relabelled_pads = relabel_anns(sorted_pads, "", "pad", relabel_method)
    new_anns.extend(relabelled_pads)

    # Relabel reference squares
    for test in PAD_INDEX_TO_TEST.values():
        ref_anns = select_anns(category_map, test, anns, "reference")
        sorted_refs = sort_anns(ref_anns, rotation, "reference")
        relabelled_refs = relabel_anns(sorted_refs, test, "reference", relabel_method)
        new_anns.extend(relabelled_refs)

    # Relabel hand annotation
    hand_anns = select_anns(category_map, "", anns, "hand")
    for hand_ann in hand_anns:
        if relabel_method == "simple":
            hand_ann["category_id"] = SIMPLE_CLASS_MAP["hand"]
        else:
            hand_ann["category_id"] = GRANULAR_CLASS_MAP["hand"]
        new_anns.append(hand_ann)

    return new_anns

def update_categories(relabel_method):
    """Generate new categories list based on relabel method."""
    if relabel_method not in ["simple", "granular"]:
        raise ValueError("relabel_method must be 'simple' or 'granular'")

    class_map = SIMPLE_CLASS_MAP if relabel_method == "simple" else GRANULAR_CLASS_MAP

    new_categories = []
    for name, new_id in class_map.items():
        name = name if relabel_method == "simple" else name[:-1]
        new_categories.append({
            "id": new_id,
            "name": name,
            "supercategory": "pads" if name.startswith("pad")
            else "reference" if name in list(PAD_INDEX_TO_TEST.values())
            else "utility"
        })
    return new_categories

def relabel_annotations(directory, split: str, relabel_method: str):
    """Relabel pads and reference squares in COCO annotations for a split
    to match the specified relabel method.
    """
    if relabel_method not in ["simple", "granular"]:
        raise ValueError("relabel_method must be 'simple' or 'granular'")

    ann_path = directory / split / "labels" / "_annotations.coco.json"
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Build category maps
    categories = coco["categories"]
    category_map = {c["name"]: c["id"] for c in categories}

    # Create new annotations list for images
    new_annotations = []
    for img in coco["images"]:
        anns = [a for a in coco["annotations"] if a["image_id"] == img["id"]]
        new_anns = generate_anns(img, anns, category_map, relabel_method=relabel_method)
        new_annotations.extend(new_anns)
        logger.info(f"Processed image id {img['id']} with {len(anns)} annotations.")
    coco["annotations"] = new_annotations

    # Update categories
    coco["categories"] = update_categories(relabel_method)

    # Save updated JSON
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)
    logger.success(f"Processed {split}, saved {ann_path}")


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

    # Move image files into images/
    for file in split_dir.glob("*.*"):
        if file.name.startswith("_annotations"):
            continue
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            dest = images_dir / file.name
            if not dest.exists():
                shutil.move(str(file), str(dest))
                logger.info(f"Moved {file.name} → {dest}")

    # Handle annotation JSONs
    relabelled_json = split_dir / "_annotations.relabelled.coco.json"
    old_json = split_dir / "_annotations.coco.json"

    # Prefer relabelled JSON if present
    src_json = relabelled_json if relabelled_json.exists() else old_json
    if src_json.exists():
        new_json = labels_dir / "_annotations.coco.json"
        shutil.move(str(src_json), str(new_json))
        logger.info(f"Moved {src_json} → {new_json}")
    else:
        logger.warning(f"No annotation JSON found in {split_dir}")


def coco_to_yolo(coco_json_path: Path, labels_dir: Path):
    """Convert COCO annotations to YOLO TXT format with proper id→index mapping."""
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Build mapping from COCO category_id → YOLO index (0..nc-1)
    categories = coco["categories"]
    id_to_index = {c["id"]: i for i, c in enumerate(categories)}

    # Map image_id → image info
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

    logger.success(f"Converted {ann_count} annotations from {coco_json_path} → YOLO TXT in {labels_dir}")

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
    if relabel_method not in {"simple", "granular"}:
        raise ValueError("relabel_method must be 'simple' or 'granular'")

    class_map = SIMPLE_CLASS_MAP if relabel_method == "simple" else GRANULAR_CLASS_MAP

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

        for name, idx in class_map.items():
            f.write(f"  {idx}: {name}\n")

    logger.success(f"Generated YOLO YAML at {yaml_path} ({relabel_method} classes)")
