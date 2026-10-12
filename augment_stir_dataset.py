"""
augment_stir_dataset.py

Offline augmentation for the soil_moisture_stir_september dataset.
Generates 8 augmented versions of each original (non-aug) image and
saves them alongside the originals with matching YOLO label files.

Only processes original images (skips existing _aug* files to avoid
double-augmenting).

Usage:
    python augment_stir_dataset.py
    python augment_stir_dataset.py --dry-run   # preview counts only
"""

import cv2
import numpy as np
import argparse
from pathlib import Path

DATASET_ROOT = Path("data/training-data/downloaded/soil_moisture_stir_september")
SPLITS       = ["train", "valid", "test"]

# ── Augmentation functions ────────────────────────────────────────────────────
# Each function takes (image, boxes) and returns (augmented_image, new_boxes).
# boxes: list of [cls, cx, cy, w, h] in normalised YOLO format.

def flip_horizontal(img, boxes):
    img = cv2.flip(img, 1)
    new_boxes = []
    for cls, cx, cy, w, h in boxes:
        new_boxes.append([cls, 1.0 - cx, cy, w, h])
    return img, new_boxes

def flip_vertical(img, boxes):
    img = cv2.flip(img, 0)
    new_boxes = []
    for cls, cx, cy, w, h in boxes:
        new_boxes.append([cls, cx, 1.0 - cy, w, h])
    return img, new_boxes

def rotate_90(img, boxes):
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    new_boxes = []
    for cls, cx, cy, w, h in boxes:
        new_boxes.append([cls, 1.0 - cy, cx, h, w])
    return img, new_boxes

def rotate_180(img, boxes):
    img = cv2.rotate(img, cv2.ROTATE_180)
    new_boxes = []
    for cls, cx, cy, w, h in boxes:
        new_boxes.append([cls, 1.0 - cx, 1.0 - cy, w, h])
    return img, new_boxes

def rotate_270(img, boxes):
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    new_boxes = []
    for cls, cx, cy, w, h in boxes:
        new_boxes.append([cls, cy, 1.0 - cx, h, w])
    return img, new_boxes

def brightness_contrast(img, boxes):
    alpha = np.random.uniform(0.7, 1.3)   # contrast
    beta  = np.random.randint(-40, 40)    # brightness
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img, boxes

def blur_noise(img, boxes):
    img = cv2.GaussianBlur(img, (3, 3), 0)
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img, boxes

def combined(img, boxes):
    # flip horizontal + rotate 90 + brightness shift
    img, boxes = flip_horizontal(img, boxes)
    img, boxes = rotate_90(img, boxes)
    img, boxes = brightness_contrast(img, boxes)
    return img, boxes

AUGMENTATIONS = [
    ("augx00", flip_horizontal),
    ("augx01", flip_vertical),
    ("augx02", rotate_90),
    ("augx03", rotate_180),
    ("augx04", rotate_270),
    ("augx05", brightness_contrast),
    ("augx06", blur_noise),
    ("augx07", combined),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def read_label(lbl_path):
    boxes = []
    p = Path(lbl_path)
    if p.exists():
        for line in p.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) == 5:
                boxes.append([int(float(parts[0]))] + [float(x) for x in parts[1:]])
    return boxes

def write_label(lbl_path, boxes):
    lines = [f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in boxes]
    Path(lbl_path).write_text("\n".join(lines) + "\n")

def get_originals(img_dir):
    """Return only original (non-augmented) image files."""
    all_imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    return sorted([p for p in all_imgs if "_aug" not in p.stem])

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print counts only, do not write files")
    args = parser.parse_args()

    total_written = 0

    for split in SPLITS:
        img_dir = DATASET_ROOT / split / "images"
        lbl_dir = DATASET_ROOT / split / "labels"

        if not img_dir.exists():
            print(f"Skipping {split} — images dir not found")
            continue

        originals = get_originals(img_dir)
        print(f"\n{split}: {len(originals)} originals → "
              f"{len(originals) * len(AUGMENTATIONS)} new images")

        if args.dry_run:
            continue

        for img_path in originals:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  WARNING: could not read {img_path.name}, skipping")
                continue

            lbl_path = lbl_dir / (img_path.stem + ".txt")
            boxes    = read_label(lbl_path)

            for suffix, aug_fn in AUGMENTATIONS:
                out_stem    = f"{img_path.stem}_{suffix}"
                out_img     = img_dir / f"{out_stem}{img_path.suffix}"
                out_lbl     = lbl_dir / f"{out_stem}.txt"

                # Skip if already exists
                if out_img.exists():
                    continue

                aug_img, aug_boxes = aug_fn(img.copy(), [b[:] for b in boxes])
                cv2.imwrite(str(out_img), aug_img)
                write_label(out_lbl, aug_boxes)
                total_written += 1

        print(f"  Done — {split}")

    if not args.dry_run:
        print(f"\nTotal new images written: {total_written}")
    else:
        print(f"\nDry run complete — no files written")


if __name__ == "__main__":
    main()
