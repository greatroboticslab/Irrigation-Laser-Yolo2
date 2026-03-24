"""
generate_synthetic_stir.py
Generates synthetic training images for soil_moisture_stir_september using
Google Imagen 3 via the Gemini API, then writes paired YOLO label files.

Requirements:
    pip install google-genai pillow

Setup:
    export GEMINI_API_KEY="your_key_from_aistudio.google.com"

Target output per class:
    train : 50 images
    valid : 15 images
    test  : 10 images
    ──────────────────
    total : 375 images  (5 classes × 75)

Resume-safe: skips any image/label pair that already exists on disk.
"""

import io
import os
import random
import time
from pathlib import Path

from PIL import Image
from google import genai
from google.genai import types

# ── Config ────────────────────────────────────────────────────────────────────

DATASET_ROOT = Path("data/training-data/downloaded/soil_moisture_stir_september")
IMAGEN_MODEL = "imagen-4.0-generate-001"
SEED         = 99
DELAY_S      = 2.0   # seconds between requests (free tier: 30 req/min for Imagen)

SPLIT_COUNTS = {
    "train": 50,
    "valid": 15,
    "test":  10,
}

random.seed(SEED)

# ── Base prompt ───────────────────────────────────────────────────────────────

BASE_PROMPT = (
    "Top-down close-up photograph of a round black fabric grow pot filled with dark "
    "humus-rich potting soil. The soil surface is slightly uneven with small white "
    "perlite granules and mineral aggregates scattered throughout. A strawberry plant "
    "with lobed, serrated-edge green leaves is growing in the upper-left quadrant of the "
    "pot, with visible green stems. A narrow cylindrical metal moisture sensor probe is "
    "inserted vertically into the soil near the center-lower area of the pot. {led}. "
    "The black ribbed fabric rim of the pot is partially visible on the right and bottom edges. "
    "Beyond the pot rim, a wooden laminate floor is faintly visible. Black and silver braided "
    "cables are loosely draped near the right rim of the pot. Indoor ambient lighting, slight "
    "shadows. Yellow-green text \"{overlay}\" and \"Time:\" are rendered in the top-left corner "
    "of the image in a small monospaced font. Photorealistic, natural camera noise, "
    "640x640 square crop, overhead angle. {variation}"
)

# ── Per-class config ──────────────────────────────────────────────────────────

CLASSES = [
    {
        "idx":     0,
        "name":    "moisture_level_1",
        "overlay": "Moisture: 1",
        "led": (
            "The probe tip emits a faint, tiny red-orange LED glow at the soil surface. "
            "The light is dim and concentrated in a very small spot (~15-30px diameter). "
            "The soil around the probe looks dry, lighter brown/grey in color, with a "
            "powdery or cracked surface texture. Almost no reflected glow on surrounding soil."
        ),
        # YOLO bbox guidance ranges: (min, max)
        "bbox": {
            "cx": (0.43, 0.50),
            "cy": (0.58, 0.68),
            "w":  (0.04, 0.07),
            "h":  (0.05, 0.10),
        },
    },
    {
        "idx":     1,
        "name":    "moisture_level_10",
        "overlay": "Moisture: 10",
        "led": (
            "The probe tip emits an intensely bright blue-violet/indigo LED glow at the "
            "soil surface. The light blooms outward, creating a large diffuse bright blue "
            "spot (~80-140px diameter) with a bright white-blue core that spreads and blurs "
            "into the very dark, saturated wet soil. The surrounding soil looks almost black "
            "and glistening, clearly saturated with water."
        ),
        "bbox": {
            "cx": (0.44, 0.57),
            "cy": (0.46, 0.79),
            "w":  (0.07, 0.17),
            "h":  (0.12, 0.33),
        },
    },
    {
        "idx":     2,
        "name":    "moisture_level_3",
        "overlay": "Moisture: 3",
        "led": (
            "The probe tip emits a pink-magenta or pale purple LED glow at the soil surface. "
            "The spot is small to medium (~25-55px diameter) with moderate brightness. "
            "The soil has some moisture but still shows lighter brown patches and visible "
            "dry areas between the perlite granules."
        ),
        "bbox": {
            "cx": (0.42, 0.52),
            "cy": (0.54, 0.70),
            "w":  (0.04, 0.10),
            "h":  (0.05, 0.24),
        },
    },
    {
        "idx":     3,
        "name":    "moisture_level_7",
        "overlay": "Moisture: 7",
        "led": (
            "The probe tip emits a bright blue-violet LED glow at the soil surface. "
            "The glow is clearly blue/indigo, medium in size (~35-70px diameter), with a "
            "visible glow halo on the surrounding dark moist soil. The soil looks dark brown "
            "and clearly damp. The probe itself (a thin metal cylinder) is partially visible "
            "above the glow."
        ),
        "bbox": {
            "cx": (0.44, 0.53),
            "cy": (0.49, 0.64),
            "w":  (0.05, 0.13),
            "h":  (0.06, 0.26),
        },
    },
    {
        "idx":     4,
        "name":    "moisture_level_8",
        "overlay": "Moisture: 8",
        "led": (
            "The probe tip emits a strong blue-violet LED glow at the soil surface. "
            "Larger and brighter than moisture 7, (~60-100px diameter), with a distinct "
            "violet-blue bloom spreading into the dark wet soil. Soil surface appears very "
            "dark and saturated, reflecting some blue light. The glow may show slight "
            "vertical elongation along the probe axis."
        ),
        "bbox": {
            "cx": (0.44, 0.52),
            "cy": (0.54, 0.70),
            "w":  (0.07, 0.15),
            "h":  (0.12, 0.31),
        },
    },
]

# ── Scene variation pool ──────────────────────────────────────────────────────

VARIATIONS = [
    "Sparse leaf coverage, few leaves visible.",
    "Medium leaf coverage over the soil.",
    "Dense leaf coverage, leaves partially obscure the soil.",
    "Dry, slightly cracked soil surface texture.",
    "Moist, dark soil surface texture.",
    "Wet, glistening soil surface texture.",
    "Straight overhead camera angle (0 degrees tilt).",
    "Slightly tilted camera angle (5-15 degrees off vertical).",
    "Angled camera view (15-25 degrees off vertical).",
    "Full pot visible within frame.",
    "Pot edge cut off on one side of the frame.",
    "Bright indoor lighting, well-lit scene.",
    "Dim ambient indoor lighting, softer shadows.",
    "Slightly overexposed lighting.",
    "Probe positioned slightly left of center.",
    "Probe positioned slightly right of center.",
    "No cables visible near the pot.",
    "One or two cables partially visible at the pot edge.",
    "Probe fully visible above the soil surface.",
    "Probe partially occluded by a leaf edge.",
    "Sparse white perlite granules visible on soil surface.",
    "Dense white perlite specks scattered across soil surface.",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_prompt(cls: dict) -> str:
    variation = random.choice(VARIATIONS)
    return BASE_PROMPT.format(
        led=cls["led"],
        overlay=cls["overlay"],
        variation=variation,
    )


def sample_bbox(cls: dict) -> tuple:
    b = cls["bbox"]
    cx = random.uniform(*b["cx"])
    cy = random.uniform(*b["cy"])
    w  = random.uniform(*b["w"])
    h  = random.uniform(*b["h"])
    # Clip so bbox stays inside image
    cx = max(w / 2, min(1.0 - w / 2, cx))
    cy = max(h / 2, min(1.0 - h / 2, cy))
    return cx, cy, w, h


def save_label(path: Path, cls_idx: int, bbox: tuple):
    cx, cy, w, h = bbox
    path.write_text(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def call_imagen(client, prompt: str) -> bytes:
    """Call Imagen 3 and return raw JPEG bytes."""
    response = client.models.generate_images(
        model=IMAGEN_MODEL,
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio="1:1",
            output_mime_type="image/jpeg",
        ),
    )
    generated = response.generated_images
    if not generated:
        raise RuntimeError("Imagen returned no images (possible content filter rejection).")
    return generated[0].image.image_bytes


def resize_to_640(raw_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    img = img.resize((640, 640), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    api_key = "AIzaSyAav0AcRFJMP6kDGgfblJWbzJu3d97taJM"

    client = genai.Client(api_key=api_key)

    total_target  = sum(SPLIT_COUNTS.values()) * len(CLASSES)
    total_done    = 0
    total_skipped = 0
    total_errors  = 0

    print("=" * 64)
    print("Synthetic data generation — soil_moisture_stir_september")
    print(f"  Model  : {IMAGEN_MODEL}")
    print(f"  Target : {total_target} images  ({len(CLASSES)} classes × {sum(SPLIT_COUNTS.values())} each)")
    print(f"  Delay  : {DELAY_S}s between requests")
    print("=" * 64)

    for cls in CLASSES:
        print(f"\n▶  Class {cls['idx']}  {cls['name']}")

        for split, count in SPLIT_COUNTS.items():
            img_dir = DATASET_ROOT / split / "images"
            lbl_dir = DATASET_ROOT / split / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            print(f"   [{split:5s}] {count} images ...")
            split_done = 0

            for i in range(count):
                stem     = f"synthetic_{cls['name']}_{split}_{i:03d}"
                img_path = img_dir / f"{stem}.jpg"
                lbl_path = lbl_dir / f"{stem}.txt"

                # Resume: skip if already generated
                if img_path.exists() and lbl_path.exists():
                    total_skipped += 1
                    continue

                prompt = build_prompt(cls)
                bbox   = sample_bbox(cls)

                success = False
                for attempt in range(3):
                    try:
                        raw_bytes   = call_imagen(client, prompt)
                        final_bytes = resize_to_640(raw_bytes)
                        img_path.write_bytes(final_bytes)
                        save_label(lbl_path, cls["idx"], bbox)
                        total_done += 1
                        split_done += 1
                        success = True
                        break
                    except Exception as exc:
                        wait = 10 * (2 ** attempt)
                        print(f"     attempt {attempt + 1}/3 failed: {exc}")
                        if attempt < 2:
                            print(f"     retrying in {wait}s ...")
                            time.sleep(wait)

                if not success:
                    print(f"     SKIP {stem} (3 failures)")
                    total_errors += 1

                time.sleep(DELAY_S)

            print(f"          → {split_done} generated")

    print(f"\n{'=' * 64}")
    print("COMPLETE")
    print(f"  Generated : {total_done}")
    print(f"  Skipped   : {total_skipped}  (already existed)")
    print(f"  Errors    : {total_errors}")
    print(f"{'=' * 64}")

    print("\nFinal image counts per split:")
    for split in SPLIT_COUNTS:
        img_dir = DATASET_ROOT / split / "images"
        n = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
        print(f"  {split:6s}: {n} images")


if __name__ == "__main__":
    main()
