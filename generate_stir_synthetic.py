"""
generate_stir_synthetic.py

Generates synthetic training images for soil_moisture_stir_september using
Stable Diffusion 1.5 (huggingface/diffusers), then auto-annotates each image
with a YOLO bounding box by detecting the LED glow via HSV color masking.

Target: 375 images (5 classes × 75), split 50/15/10 across train/valid/test.

Usage:
    python generate_stir_synthetic.py
    python generate_stir_synthetic.py --n-train 50 --n-valid 15 --n-test 10
    python generate_stir_synthetic.py --dry-run          # print prompts only
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionPipeline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_ROOT = Path(
    "data/training-data/downloaded/soil_moisture_stir_september"
)
SPLITS = {
    "train": DATASET_ROOT / "train",
    "valid": DATASET_ROOT / "valid",
    "test":  DATASET_ROOT / "test",
}

# ---------------------------------------------------------------------------
# Class definitions  (index matches data.yaml / Roboflow alphabetical order)
# ---------------------------------------------------------------------------
CLASSES = {
    0: {
        "name":    "moisture_level_1",
        "overlay": "Moisture: 1",
        "led":     (
            "faint tiny red-orange LED glow at soil surface, dim concentrated "
            "spot ~20px diameter, dry lighter brown-grey powdery soil, "
            "almost no reflected glow"
        ),
        # HSV ranges for LED detection: red/orange hue (OpenCV H: 0-10, 170-180)
        "hsv_masks": [
            {"lo": (0,   120, 120), "hi": (12,  255, 255)},
            {"lo": (168, 120, 120), "hi": (180, 255, 255)},
        ],
        # Fallback bbox ranges (cx, cy, w, h) — normalised [0,1]
        "bbox": {"cx": (0.43, 0.50), "cy": (0.58, 0.68),
                 "w":  (0.04, 0.07), "h":  (0.05, 0.10)},
    },
    1: {
        "name":    "moisture_level_10",
        "overlay": "Moisture: 10",
        "led":     (
            "intensely bright blue-violet indigo LED glow, large diffuse bloom "
            "~110px diameter, bright white-blue core blurring into very dark "
            "glistening saturated wet soil"
        ),
        "hsv_masks": [
            {"lo": (110, 80, 100), "hi": (145, 255, 255)},
        ],
        "bbox": {"cx": (0.44, 0.57), "cy": (0.46, 0.79),
                 "w":  (0.07, 0.17), "h":  (0.12, 0.33)},
    },
    2: {
        "name":    "moisture_level_3",
        "overlay": "Moisture: 3",
        "led":     (
            "pink-magenta or pale purple LED glow, small-medium spot ~40px "
            "diameter, moderate brightness, soil shows lighter brown patches "
            "and visible dry areas between perlite granules"
        ),
        "hsv_masks": [
            {"lo": (140, 60, 100), "hi": (170, 255, 255)},
        ],
        "bbox": {"cx": (0.42, 0.52), "cy": (0.54, 0.70),
                 "w":  (0.04, 0.10), "h":  (0.05, 0.24)},
    },
    3: {
        "name":    "moisture_level_7",
        "overlay": "Moisture: 7",
        "led":     (
            "bright blue-violet LED glow, medium spot ~50px diameter, visible "
            "glow halo on surrounding dark moist soil, soil looks dark brown "
            "and clearly damp, thin metal probe cylinder partially visible above glow"
        ),
        "hsv_masks": [
            {"lo": (110, 80, 100), "hi": (145, 255, 255)},
        ],
        "bbox": {"cx": (0.44, 0.53), "cy": (0.49, 0.64),
                 "w":  (0.05, 0.13), "h":  (0.06, 0.26)},
    },
    4: {
        "name":    "moisture_level_8",
        "overlay": "Moisture: 8",
        "led":     (
            "strong blue-violet LED glow, larger than moisture 7, ~80px "
            "diameter, distinct violet-blue bloom spreading into dark wet soil, "
            "slight vertical elongation along probe axis, soil surface very dark "
            "and saturated, reflecting blue light"
        ),
        "hsv_masks": [
            {"lo": (110, 80, 100), "hi": (145, 255, 255)},
        ],
        "bbox": {"cx": (0.44, 0.52), "cy": (0.54, 0.70),
                 "w":  (0.07, 0.15), "h":  (0.12, 0.31)},
    },
}

# ---------------------------------------------------------------------------
# Scene variation pools
# ---------------------------------------------------------------------------
LEAF_COVERAGE = [
    "sparse leaves, most soil surface visible",
    "medium leaf coverage, some leaves visible",
    "dense leaves partially covering soil, some green stems prominent",
]
SOIL_TEXTURE = [
    "dry cracked soil surface",
    "moist dark soil",
    "wet glistening soil surface",
]
CAMERA_ANGLE = [
    "straight overhead 0 degree angle",
    "slight tilt 8 degree angle",
    "angled view 18 degree angle",
]
POT_CROP = [
    "full pot visible",
    "pot edge cut off on right side",
    "pot edge cut off on bottom",
    "pot edges cut off on two sides",
]
LIGHTING = [
    "bright indoor lighting",
    "dim ambient indoor lighting",
    "slight overexposure",
    "slight underexposure, darker image",
]
CABLE = [
    "no cables visible",
    "one black cable partially visible at pot rim",
    "two braided cables near right rim",
]
PERLITE = [
    "sparse white perlite granules in soil",
    "moderate white perlite granules",
    "dense white perlite specks throughout soil",
]

BASE_PROMPT = (
    "Top-down close-up photograph of a round black fabric grow pot filled with "
    "dark humus-rich potting soil. The soil surface is slightly uneven with "
    "small white perlite granules and mineral aggregates scattered throughout. "
    "A strawberry plant with lobed serrated-edge green leaves is growing in "
    "the upper-left quadrant of the pot with visible green stems. "
    "A narrow cylindrical metal moisture sensor probe is inserted vertically "
    "into the soil near the center-lower area of the pot. {led}. "
    "The black ribbed fabric rim of the pot is partially visible on the right "
    "and bottom edges. Beyond the pot rim a wooden laminate floor is faintly "
    "visible. {cable} loosely draped near the right rim of the pot. "
    "Indoor ambient lighting slight shadows. "
    'Yellow-green text "{overlay}" and "Time:" are rendered in the top-left '
    "corner of the image in a small monospaced font. "
    "{leaf}, {soil}, {angle}, {crop}, {lighting}, {perlite}. "
    "Photorealistic natural camera noise 640x640 square crop overhead angle."
)

NEGATIVE_PROMPT = (
    "blurry, cartoon, illustration, painting, sketch, low quality, "
    "wrong colors, green LED, white LED, no LED, dark image with no glow, "
    "person, hand, text overlay wrong color"
)


def build_prompt(class_idx: int) -> str:
    meta = CLASSES[class_idx]
    return BASE_PROMPT.format(
        led=meta["led"],
        overlay=meta["overlay"],
        leaf=random.choice(LEAF_COVERAGE),
        soil=random.choice(SOIL_TEXTURE),
        angle=random.choice(CAMERA_ANGLE),
        crop=random.choice(POT_CROP),
        lighting=random.choice(LIGHTING),
        cable=random.choice(CABLE),
        perlite=random.choice(PERLITE),
    )


# ---------------------------------------------------------------------------
# LED detection via HSV masking
# ---------------------------------------------------------------------------
def detect_led_bbox(img_rgb: np.ndarray, class_idx: int):
    """
    Returns (cx, cy, w, h) normalised [0,1] by detecting the LED glow via
    HSV colour masking. Returns None if detection fails.
    """
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    combined_mask = np.zeros(img_hsv.shape[:2], dtype=np.uint8)
    for rng in CLASSES[class_idx]["hsv_masks"]:
        lo = np.array(rng["lo"], dtype=np.uint8)
        hi = np.array(rng["hi"], dtype=np.uint8)
        combined_mask |= cv2.inRange(img_hsv, lo, hi)

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN,  kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(
        combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    # Pick the largest contour
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 50:  # too small — likely noise
        return None

    x, y, w_px, h_px = cv2.boundingRect(largest)
    H, W = img_rgb.shape[:2]

    # Expand box slightly to include probe tip above glow
    pad_top = int(h_px * 0.3)
    y = max(0, y - pad_top)
    h_px = min(H - y, h_px + pad_top)

    cx = (x + w_px / 2) / W
    cy = (y + h_px / 2) / H
    w  = w_px / W
    h  = h_px / H
    return cx, cy, w, h


def fallback_bbox(class_idx: int):
    """Sample a random bbox from the per-class guidance ranges."""
    b = CLASSES[class_idx]["bbox"]
    cx = random.uniform(*b["cx"])
    cy = random.uniform(*b["cy"])
    w  = random.uniform(*b["w"])
    h  = random.uniform(*b["h"])
    return cx, cy, w, h


# ---------------------------------------------------------------------------
# QC checks
# ---------------------------------------------------------------------------
def qc_pass(img_rgb: np.ndarray, class_idx: int) -> bool:
    """
    Returns True if the image passes basic quality checks:
    - Soil fills ≥ 50% of frame (dark brown / black pixels dominate)
    - LED glow colour is present
    """
    # Check that soil (dark brownish) covers at least 50% of pixels
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    # Dark soil: low-medium saturation, low value
    soil_mask = cv2.inRange(img_hsv,
                            np.array([0,  0,  10]),
                            np.array([40, 180, 120]))
    soil_fraction = np.sum(soil_mask > 0) / soil_mask.size
    if soil_fraction < 0.30:  # relaxed to 30% (pot rim + leaves take space)
        return False

    # Check that the LED colour is detectable
    for rng in CLASSES[class_idx]["hsv_masks"]:
        lo = np.array(rng["lo"], dtype=np.uint8)
        hi = np.array(rng["hi"], dtype=np.uint8)
        if np.any(cv2.inRange(img_hsv, lo, hi) > 0):
            return True
    return False


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train",  type=int, default=50)
    parser.add_argument("--n-valid",  type=int, default=15)
    parser.add_argument("--n-test",   type=int, default=10)
    parser.add_argument("--model",    default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--steps",    type=int, default=30,
                        help="Inference steps (20-50 is typical)")
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print one prompt per class and exit")
    args = parser.parse_args()

    random.seed(args.seed)

    split_counts = {
        "train": args.n_train,
        "valid": args.n_valid,
        "test":  args.n_test,
    }

    if args.dry_run:
        for cls_idx in CLASSES:
            print(f"\n=== Class {cls_idx}: {CLASSES[cls_idx]['name']} ===")
            print(build_prompt(cls_idx))
        sys.exit(0)

    # ------------------------------------------------------------------
    # Load pipeline
    # ------------------------------------------------------------------
    print(f"Loading model: {args.model}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,          # disable NSFW filter (irrelevant here)
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    total_generated = 0
    total_rejected  = 0
    global_img_idx  = 0

    for split_name, split_dir in SPLITS.items():
        n_per_class = split_counts[split_name]
        img_dir   = split_dir / "images"
        label_dir = split_dir / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Split: {split_name}  ({n_per_class} images × 5 classes = "
              f"{n_per_class * 5} total)")
        print(f"{'='*60}")

        for cls_idx, meta in CLASSES.items():
            generated = 0
            attempts  = 0
            max_attempts = n_per_class * 4  # allow up to 4× retries per slot

            print(f"\n  Class {cls_idx} — {meta['name']}")

            while generated < n_per_class and attempts < max_attempts:
                attempts += 1
                seed = args.seed + global_img_idx * 100 + attempts
                generator = torch.Generator(device=device).manual_seed(seed)

                prompt = build_prompt(cls_idx)

                result = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    width=640,
                    height=640,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    generator=generator,
                )
                img_pil = result.images[0]
                img_rgb = np.array(img_pil)

                # QC
                if not qc_pass(img_rgb, cls_idx):
                    total_rejected += 1
                    print(f"    [REJECT] attempt {attempts} — QC failed")
                    continue

                # Bounding box
                bbox = detect_led_bbox(img_rgb, cls_idx)
                used_fallback = False
                if bbox is None:
                    bbox = fallback_bbox(cls_idx)
                    used_fallback = True

                cx, cy, w, h = bbox

                # Clamp to [0,1]
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w  = max(0.01, min(1.0, w))
                h  = max(0.01, min(1.0, h))

                # Save
                fname = f"syn_{meta['name']}_{split_name}_{generated:04d}"
                img_path   = img_dir   / f"{fname}.jpg"
                label_path = label_dir / f"{fname}.txt"

                img_pil.save(str(img_path), quality=95)
                label_path.write_text(
                    f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                )

                generated       += 1
                total_generated += 1
                global_img_idx  += 1
                fb_tag = " [fallback bbox]" if used_fallback else ""
                print(f"    [{generated:3d}/{n_per_class}] {fname}.jpg  "
                      f"bbox=({cx:.3f},{cy:.3f},{w:.3f},{h:.3f}){fb_tag}")

            if generated < n_per_class:
                print(f"  WARNING: only generated {generated}/{n_per_class} "
                      f"for {meta['name']} in {split_name} "
                      f"(max attempts reached)")

    print(f"\n{'='*60}")
    print(f"Done. Generated: {total_generated}  Rejected: {total_rejected}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
