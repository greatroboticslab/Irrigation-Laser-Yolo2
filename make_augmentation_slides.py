"""
make_augmentation_slides.py
Generates a PDF slide deck comparing original vs augmented
soil_moisture_stir_september images.

Output: plots/stir_september_augmentation_slides.pdf
"""

import os
import random
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

# ── Config ───────────────────────────────────────────────────────────────────

DATASET_ROOT = Path("data/training-data/downloaded/soil_moisture_stir_september")
OUTPUT_DIR   = Path("plots")
OUTPUT_PDF   = OUTPUT_DIR / "stir_september_augmentation_slides.pdf"

CLASSES = [
    "moisture_level_1",   # 0 — very dry,  LED: red/orange
    "moisture_level_10",  # 1 — very wet,  LED: intense blue-violet
    "moisture_level_3",   # 2 — dry,       LED: pink-magenta
    "moisture_level_7",   # 3 — moist,     LED: bright blue-violet
    "moisture_level_8",   # 4 — wet,       LED: strong blue-violet
]

# Colours for bounding boxes per class (BGR→RGB already converted)
CLASS_COLORS = [
    (1.0, 0.35, 0.0),   # orange-red   — level 1
    (0.45, 0.0, 0.9),   # deep violet  — level 10
    (0.9, 0.2, 0.6),    # pink-magenta — level 3
    (0.2, 0.4, 1.0),    # bright blue  — level 7
    (0.0, 0.7, 1.0),    # cyan-blue    — level 8
]

SLIDE_BG  = "#0d1117"
TITLE_CLR = "#e6edf3"
BODY_CLR  = "#8b949e"
ACCENT    = "#58a6ff"
GREEN     = "#3fb950"
YELLOW    = "#d29922"

SEED = 7
random.seed(SEED)

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_image_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_label(label_path: Path):
    classes, boxes = [], []
    if label_path.exists():
        for line in label_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) == 5:
                classes.append(int(float(parts[0])))
                boxes.append(list(map(float, parts[1:])))
    return classes, boxes


def draw_boxes(ax, img: np.ndarray, classes, boxes, linewidth=2):
    h, w = img.shape[:2]
    ax.imshow(img)
    for cls, (cx, cy, bw, bh) in zip(classes, boxes):
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        color = CLASS_COLORS[cls % len(CLASS_COLORS)]
        rect = mpatches.FancyBboxPatch(
            (x1, y1), bw * w, bh * h,
            boxstyle="square,pad=0",
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 4,
            CLASSES[cls].replace("moisture_level_", "Lvl "),
            color=color, fontsize=7, fontweight="bold",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", fc=SLIDE_BG, ec="none", alpha=0.7),
        )
    ax.axis("off")


def slide_fig(figsize=(16, 9)):
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(SLIDE_BG)
    return fig


def slide_title(fig, title, subtitle=None, y_title=0.88, y_sub=0.79):
    fig.text(0.5, y_title, title, ha="center", va="center",
             color=TITLE_CLR, fontsize=26, fontweight="bold")
    if subtitle:
        fig.text(0.5, y_sub, subtitle, ha="center", va="center",
                 color=BODY_CLR, fontsize=14)


def get_originals(split: str):
    img_dir = DATASET_ROOT / split / "images"
    lbl_dir = DATASET_ROOT / split / "labels"
    files = sorted([p for p in img_dir.glob("*.jpg") if "_aug" not in p.name] +
                   [p for p in img_dir.glob("*.png") if "_aug" not in p.name])
    return img_dir, lbl_dir, files


def get_augmented_for(original_stem: str, split: str):
    img_dir = DATASET_ROOT / split / "images"
    lbl_dir = DATASET_ROOT / split / "labels"
    augs = sorted(img_dir.glob(f"{original_stem}_aug*.jpg"))
    return img_dir, lbl_dir, augs


def count_classes_in_split(split: str):
    lbl_dir = DATASET_ROOT / split / "labels"
    counts = defaultdict(int)
    for lbl in lbl_dir.glob("*.txt"):
        for line in lbl.read_text().strip().splitlines():
            parts = line.split()
            if parts:
                counts[int(float(parts[0]))] += 1
    return counts


def count_originals_only(split: str):
    lbl_dir = DATASET_ROOT / split / "labels"
    counts = defaultdict(int)
    for lbl in lbl_dir.glob("*.txt"):
        if "_aug" not in lbl.stem:
            for line in lbl.read_text().strip().splitlines():
                parts = line.split()
                if parts:
                    counts[int(float(parts[0]))] += 1
    return counts


# ── Slide builders ────────────────────────────────────────────────────────────

def slide_cover(pdf):
    fig = slide_fig()
    # Decorative accent bar
    fig.add_axes([0.08, 0.46, 0.84, 0.004]).set_facecolor(ACCENT)
    plt.gca().axis("off")
    fig.text(0.5, 0.65, "soil_moisture_stir_september",
             ha="center", color=ACCENT, fontsize=18, fontweight="bold",
             fontfamily="monospace")
    fig.text(0.5, 0.54, "Data Augmentation Overview",
             ha="center", color=TITLE_CLR, fontsize=34, fontweight="bold")
    fig.text(0.5, 0.42, "Original 56 images  →  685 images after albumentations pipeline",
             ha="center", color=BODY_CLR, fontsize=15)
    fig.text(0.5, 0.32, "5 moisture-level classes  ·  LED colour = class signal  ·  YOLO bbox labels preserved",
             ha="center", color=BODY_CLR, fontsize=12)
    pdf.savefig(fig, facecolor=SLIDE_BG)
    plt.close(fig)


def slide_dataset_overview(pdf):
    fig = slide_fig()
    slide_title(fig, "Dataset Overview", "Before and after augmentation — per split")

    ax = fig.add_axes([0.1, 0.1, 0.8, 0.6])
    ax.set_facecolor(SLIDE_BG)
    ax.axis("off")

    headers = ["Split", "Originals", "Copies each", "Total (aug)", "% increase"]
    rows = [
        ["train",  "39",  "13", "546", "+1300%"],
        ["valid",  "12",  " 6", " 84", "+ 600%"],
        ["test",   " 5",  "10", " 55", "+1000%"],
        ["TOTAL",  "56",  "—",  "685", "+1123%"],
    ]

    col_x = [0.05, 0.25, 0.45, 0.62, 0.80]
    header_y = 0.92
    row_h = 0.14

    for i, h in enumerate(headers):
        ax.text(col_x[i], header_y, h,
                transform=ax.transAxes, color=ACCENT,
                fontsize=13, fontweight="bold")

    for r, row in enumerate(rows):
        y = header_y - row_h * (r + 1)
        bg_color = "#161b22" if r % 2 == 0 else "#0d1117"
        rect = mpatches.FancyBboxPatch(
            (0.0, y - 0.04), 1.0, row_h - 0.01,
            transform=ax.transAxes,
            boxstyle="round,pad=0.005",
            facecolor=bg_color, edgecolor="none",
        )
        ax.add_patch(rect)
        is_total = row[0] == "TOTAL"
        clr = YELLOW if is_total else TITLE_CLR
        for i, val in enumerate(row):
            ax.text(col_x[i], y + 0.02, val,
                    transform=ax.transAxes,
                    color=clr if i == 0 else (GREEN if i == 4 else BODY_CLR),
                    fontsize=13 if is_total else 12,
                    fontweight="bold" if is_total else "normal")

    pdf.savefig(fig, facecolor=SLIDE_BG)
    plt.close(fig)


def slide_originals_grid(pdf):
    """Show all 39 train originals in a grid with bboxes."""
    _, lbl_dir, files = get_originals("train")

    n = len(files)
    ncols = 8
    nrows = int(np.ceil(n / ncols))

    fig = slide_fig(figsize=(16, 9))
    slide_title(fig, "Original Training Images (39)", "All originals — train split, with ground-truth bboxes",
                y_title=0.97, y_sub=0.935)

    gs = GridSpec(nrows, ncols, figure=fig,
                  left=0.01, right=0.99, top=0.91, bottom=0.01,
                  hspace=0.05, wspace=0.04)

    for idx, img_path in enumerate(files):
        r, c = divmod(idx, ncols)
        ax = fig.add_subplot(gs[r, c])
        ax.set_facecolor(SLIDE_BG)
        img = load_image_rgb(img_path)
        lbl = lbl_dir / (img_path.stem + ".txt")
        cls, boxes = read_label(lbl)
        draw_boxes(ax, img, cls, boxes, linewidth=1.5)

    # blank remaining cells
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = fig.add_subplot(gs[r, c])
        ax.set_facecolor(SLIDE_BG)
        ax.axis("off")

    pdf.savefig(fig, facecolor=SLIDE_BG)
    plt.close(fig)


def slide_augmentation_showcase(pdf):
    """One slide per train original: original (full left col) + all 13 augments (right grid)."""
    _, lbl_dir, files = get_originals("train")

    for img_idx, orig_path in enumerate(files):
        stem = orig_path.stem
        _, _, aug_files = get_augmented_for(stem, "train")

        orig_lbl = lbl_dir / (stem + ".txt")
        orig_cls, orig_boxes = read_label(orig_lbl)
        orig_img = load_image_rgb(orig_path)

        # Determine class label for subtitle
        class_name = CLASSES[orig_cls[0]].replace("moisture_level_", "Level ") if orig_cls else "unknown"

        fig = slide_fig()

        # Header
        fig.text(0.5, 0.975,
                 f"Image {img_idx + 1} / {len(files)}  ·  {class_name}",
                 ha="center", color=ACCENT, fontsize=11, fontweight="bold")
        fig.text(0.5, 0.955,
                 f"Original  +  {len(aug_files)} augmented copies  ·  {stem[:55]}",
                 ha="center", color=BODY_CLR, fontsize=9)

        # Layout: 4 rows × 5 cols
        # Col 0 (spans all 4 rows) = original
        # Cols 1-4 × 4 rows = 16 cells → fill with augments (up to 13)
        N_ROWS, N_COLS = 4, 5
        gs = GridSpec(N_ROWS, N_COLS, figure=fig,
                      left=0.01, right=0.99, top=0.94, bottom=0.01,
                      hspace=0.05, wspace=0.03)

        # Original — tall left column
        ax_orig = fig.add_subplot(gs[:, 0])
        ax_orig.set_facecolor(SLIDE_BG)
        draw_boxes(ax_orig, orig_img, orig_cls, orig_boxes, linewidth=2)
        ax_orig.set_title("ORIGINAL", color=YELLOW, fontsize=10,
                          fontweight="bold", pad=3)

        # Augmented — fill right grid row by row
        aug_cells = [(r, c) for r in range(N_ROWS) for c in range(1, N_COLS)]
        for cell_i, aug_path in enumerate(aug_files):
            if cell_i >= len(aug_cells):
                break
            r, c = aug_cells[cell_i]
            ax = fig.add_subplot(gs[r, c])
            ax.set_facecolor(SLIDE_BG)
            aug_lbl = DATASET_ROOT / "train" / "labels" / (aug_path.stem + ".txt")
            aug_cls, aug_boxes = read_label(aug_lbl)
            aug_img = load_image_rgb(aug_path)
            draw_boxes(ax, aug_img, aug_cls, aug_boxes, linewidth=1.2)
            ax.set_title(f"aug {cell_i:03d}", color=ACCENT, fontsize=7, pad=1)

        # Blank any unused cells
        for cell_i in range(len(aug_files), len(aug_cells)):
            r, c = aug_cells[cell_i]
            ax = fig.add_subplot(gs[r, c])
            ax.set_facecolor(SLIDE_BG)
            ax.axis("off")

        pdf.savefig(fig, facecolor=SLIDE_BG)
        plt.close(fig)


def slide_class_distribution(pdf):
    fig = slide_fig()
    slide_title(fig, "Class Distribution — Before vs After Augmentation",
                "train split  ·  annotation counts per class")

    orig  = count_originals_only("train")
    total = count_classes_in_split("train")

    x = np.arange(len(CLASSES))
    w = 0.35

    ax = fig.add_axes([0.1, 0.13, 0.82, 0.6])
    ax.set_facecolor("#161b22")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    orig_vals  = [orig.get(i, 0)  for i in range(len(CLASSES))]
    total_vals = [total.get(i, 0) for i in range(len(CLASSES))]

    bars1 = ax.bar(x - w/2, orig_vals,  w, color=YELLOW, label="Original",   zorder=3)
    bars2 = ax.bar(x + w/2, total_vals, w, color=GREEN,  label="After augmentation", zorder=3)

    ax.set_xticks(x)
    short_names = [c.replace("moisture_level_", "Lvl ") for c in CLASSES]
    ax.set_xticklabels(short_names, color=TITLE_CLR, fontsize=11)
    ax.tick_params(axis="y", colors=BODY_CLR)
    ax.yaxis.grid(True, color="#30363d", zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel("Annotation count", color=BODY_CLR, fontsize=11)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(int(bar.get_height())), ha="center", va="bottom",
                color=YELLOW, fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(int(bar.get_height())), ha="center", va="bottom",
                color=GREEN, fontsize=9)

    ax.legend(facecolor="#161b22", edgecolor="#30363d",
              labelcolor=TITLE_CLR, fontsize=11)

    pdf.savefig(fig, facecolor=SLIDE_BG)
    plt.close(fig)


def slide_pipeline_summary(pdf):
    fig = slide_fig()
    slide_title(fig, "Augmentation Pipeline", "albumentations — all transforms applied")

    sections = [
        ("Geometric", ACCENT, [
            "HorizontalFlip  p=0.5",
            "VerticalFlip  p=0.2",
            "Rotate ±20°  (reflect border)  p=0.6",
            "Perspective warp  scale 3–8%  p=0.4",
            "RandomResizedCrop → 640×640  scale 75–100%  p=0.5",
            "ShiftScaleRotate  shift ±8%, scale ±15%, rotate ±10°  p=0.4",
        ]),
        ("Pixel / Lighting", YELLOW, [
            "RandomBrightnessContrast  brightness ±0.35/0.25, contrast ±0.2/0.3  p=0.8",
            "GaussianBlur  kernel 3–7  p=0.3",
            "MotionBlur  kernel 3–9  p=0.2",
            "GaussNoise  std 0.02–0.08  p=0.4",
            "JPEG Compression  quality 60–95%  p=0.3",
        ]),
        ("Colour  ⚠ hue shifts intentionally small — LED colour = class signal", GREEN, [
            "HueSaturationValue  hue ±8°, sat ±30, val ±25  p=0.5",
            "RGBShift  ±15 per channel  p=0.3",
        ]),
        ("Environment Simulation", "#f0883e", [
            "RandomShadow  1–2 shadows  p=0.3",
            "RandomFog  coef 0.05–0.15  p=0.1",
            "Downscale  60–90%  p=0.2",
        ]),
    ]

    x0, y0 = 0.05, 0.83
    dy_section = 0.035
    dy_item    = 0.028
    x_item     = 0.10

    for section, color, items in sections:
        fig.text(x0, y0, section, color=color, fontsize=12, fontweight="bold")
        y0 -= dy_section
        for item in items:
            fig.text(x_item, y0, f"• {item}", color=BODY_CLR, fontsize=10)
            y0 -= dy_item
        y0 -= 0.01  # extra gap between sections

    fig.text(0.05, 0.07,
             "BboxParams: format=yolo  min_visibility=0.3  clip=True  →  0 bboxes dropped",
             color=ACCENT, fontsize=10, style="italic")

    pdf.savefig(fig, facecolor=SLIDE_BG)
    plt.close(fig)


def slide_class_legend(pdf):
    """Visual legend: one representative original per class."""
    img_dir, lbl_dir, files = get_originals("train")

    # find first image per class
    class_examples = {}
    for img_path in files:
        lbl = lbl_dir / (img_path.stem + ".txt")
        cls_list, boxes = read_label(lbl)
        for c in cls_list:
            if c not in class_examples:
                class_examples[c] = (img_path, cls_list, boxes)
        if len(class_examples) == len(CLASSES):
            break

    fig = slide_fig()
    slide_title(fig, "Class Reference", "One example image per moisture level")

    gs = GridSpec(1, len(CLASSES), figure=fig,
                  left=0.03, right=0.97, top=0.85, bottom=0.05,
                  hspace=0.0, wspace=0.06)

    led_colors = ["red/orange", "intense blue-violet",
                  "pink-magenta", "bright blue-violet", "strong blue-violet"]
    wetness    = ["Very Dry", "Very Wet", "Dry", "Moist", "Wet"]

    for cls_id in range(len(CLASSES)):
        ax = fig.add_subplot(gs[0, cls_id])
        ax.set_facecolor(SLIDE_BG)
        if cls_id in class_examples:
            img_path, cls_list, boxes = class_examples[cls_id]
            img = load_image_rgb(img_path)
            draw_boxes(ax, img, cls_list, boxes, linewidth=2)
            ax.set_title(
                f"Level {CLASSES[cls_id].split('_')[-1]}\n"
                f"{wetness[cls_id]}\nLED: {led_colors[cls_id]}",
                color=CLASS_COLORS[cls_id], fontsize=8.5, fontweight="bold", pad=3,
            )
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "no example", ha="center", va="center",
                    color=BODY_CLR, fontsize=9, transform=ax.transAxes)

    pdf.savefig(fig, facecolor=SLIDE_BG)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Building slides → {OUTPUT_PDF}")

    with PdfPages(str(OUTPUT_PDF)) as pdf:
        print("  Slide 1: Cover")
        slide_cover(pdf)

        print("  Slide 2: Dataset overview")
        slide_dataset_overview(pdf)

        print("  Slide 3: Class legend")
        slide_class_legend(pdf)

        print("  Slide 4: All originals grid")
        slide_originals_grid(pdf)

        print("  Slides 5-43: Augmentation showcase (39 originals × 13 augments each)")
        slide_augmentation_showcase(pdf)

        print("  Slide 8: Class distribution before/after")
        slide_class_distribution(pdf)

        print("  Slide 9: Pipeline summary")
        slide_pipeline_summary(pdf)

        meta = pdf.infodict()
        meta["Title"]   = "soil_moisture_stir_september — Augmentation Slides"
        meta["Subject"] = "YOLOv5 data augmentation overview"

    print(f"\nDone — {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
