"""
augment_stir_september.py
Augments the soil_moisture_stir_september dataset using albumentations.
Generates new images + YOLO label files for train, valid, and test splits.

Targets:
  train : 39 originals  → ~500 total (12-13 augmented versions each)
  valid : 12 originals  → ~75  total (5-6 augmented versions each)
  test  :  5 originals  → ~50  total (9-10 augmented versions each)

IMPORTANT: Hue shifts are intentionally small — LED colour IS the class signal.
"""

import os
import cv2
import numpy as np
import albumentations as A
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────

DATASET_ROOT = Path("data/training-data/downloaded/soil_moisture_stir_september")

SPLITS = {
    "train": 13,   # augmented copies per original image
    "valid": 6,
    "test":  10,
}

SEED = 42

# ── Augmentation pipeline ────────────────────────────────────────────────────
# BboxParams with YOLO format — albumentations handles coordinate transforms.
# min_visibility=0.3: drop bbox only if <30% of it remains after crop.

def build_pipeline(seed_offset: int) -> A.Compose:
    return A.Compose(
        [
            # --- Geometric ---
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=20, border_mode=cv2.BORDER_REFLECT_101, p=0.6),
            A.Perspective(scale=(0.03, 0.08), p=0.4),
            A.RandomResizedCrop(
                size=(640, 640),
                scale=(0.75, 1.0),
                ratio=(0.9, 1.1),
                p=0.5,
            ),
            A.ShiftScaleRotate(
                shift_limit=0.08,
                scale_limit=0.15,
                rotate_limit=10,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.4,
            ),

            # --- Pixel-level (lighting / noise) ---
            # Brightness/contrast — LED must remain distinguishable
            A.RandomBrightnessContrast(
                brightness_limit=(-0.35, 0.25),
                contrast_limit=(-0.2, 0.3),
                p=0.8,
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.MotionBlur(blur_limit=(3, 9), p=0.2),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.4),
            A.ImageCompression(quality_range=(60, 95), p=0.3),

            # --- Colour — KEEP HUE SHIFTS SMALL (LED colour = class signal) ---
            A.HueSaturationValue(
                hue_shift_limit=8,       # tiny hue shift only
                sat_shift_limit=30,
                val_shift_limit=25,
                p=0.5,
            ),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),

            # --- Simulate environment ---
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_limit=(1, 2),
                shadow_dimension=5,
                p=0.3,
            ),
            A.RandomFog(fog_coef_range=(0.05, 0.15), alpha_coef=0.1, p=0.1),
            A.Downscale(scale_range=(0.6, 0.9), p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
            clip=True,
        ),
        seed=SEED + seed_offset,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def read_label(label_path: Path):
    """Return list of (class_id, cx, cy, w, h)."""
    boxes, classes = [], []
    if label_path.exists():
        with open(label_path) as f:
            for line in f.read().strip().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cls = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    classes.append(cls)
                    boxes.append(coords)
    return classes, boxes


def write_label(label_path: Path, classes, boxes):
    with open(label_path, "w") as f:
        for cls, box in zip(classes, boxes):
            cx, cy, w, h = box
            f.write(f"{cls} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}\n")


def augment_split(split: str, n_augments: int):
    img_dir = DATASET_ROOT / split / "images"
    lbl_dir = DATASET_ROOT / split / "labels"

    image_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    print(f"\n[{split}] {len(image_files)} originals → generating {n_augments} copies each")

    generated = 0
    skipped   = 0

    for idx, img_path in enumerate(image_files):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        classes, boxes = read_label(lbl_path)

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  WARNING: could not read {img_path.name}, skipping")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for aug_i in range(n_augments):
            pipeline = build_pipeline(seed_offset=idx * 1000 + aug_i)

            try:
                result = pipeline(
                    image=image,
                    bboxes=boxes,
                    class_labels=classes,
                )
            except Exception as e:
                print(f"  WARNING: augmentation failed for {img_path.name} aug {aug_i}: {e}")
                skipped += 1
                continue

            aug_img    = result["image"]
            aug_boxes  = result["bboxes"]
            aug_cls    = result["class_labels"]

            # Skip if all bboxes were lost (heavy crop)
            if len(aug_boxes) == 0 and len(boxes) > 0:
                skipped += 1
                continue

            stem     = f"{img_path.stem}_aug{aug_i:03d}"
            out_img  = img_dir / f"{stem}.jpg"
            out_lbl  = lbl_dir / f"{stem}.txt"

            aug_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_img), aug_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            write_label(out_lbl, aug_cls, aug_boxes)
            generated += 1

        if (idx + 1) % 10 == 0 or (idx + 1) == len(image_files):
            print(f"  Processed {idx + 1}/{len(image_files)} originals ...")

    print(f"  Done — {generated} images generated, {skipped} dropped (bbox lost in crop)")
    return generated


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Augmenting: soil_moisture_stir_september")
    print("=" * 60)

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

    total = 0
    for split, n in SPLITS.items():
        total += augment_split(split, n)

    print(f"\n{'=' * 60}")
    print(f"COMPLETE — {total} new images generated across all splits.")
    print(f"{'=' * 60}")

    # Print final counts
    print("\nFinal image counts per split:")
    for split in SPLITS:
        img_dir = DATASET_ROOT / split / "images"
        count = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
        print(f"  {split:6s}: {count} images")


if __name__ == "__main__":
    main()
