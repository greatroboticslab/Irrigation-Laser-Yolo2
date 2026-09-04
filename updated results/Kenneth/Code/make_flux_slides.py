"""
make_flux_slides.py

Generates a PDF slide deck showcasing FLUX.1-schnell synthetic images
for soil_moisture_stir_september across all 5 moisture classes.

Slides:
  1. Cover
  2. Dataset overview (real vs synthetic counts)
  3. Per-class: real originals grid
  4. Per-class: FLUX synthetic grid (paginated)
  5. Per-class: side-by-side real vs FLUX (8 pairs)
  6. Summary stats slide

Output: plots/flux_synthetic_slides.pdf
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

DATASET_ROOT = Path("data/training-data/downloaded/soil_moisture_stir_september")
OUTPUT_PDF   = Path("plots/flux_synthetic_slides.pdf")

CLASSES = {
    0: {"name": "moisture_level_1",  "label": "Level 1 — Very Dry",       "led": "Red/Orange LED",      "color": (1.0, 0.35, 0.0)},
    1: {"name": "moisture_level_10", "label": "Level 10 — Very Wet",       "led": "Blue-Violet LED",     "color": (0.45, 0.0, 0.9)},
    2: {"name": "moisture_level_3",  "label": "Level 3 — Slightly Dry",    "led": "Pink-Magenta LED",    "color": (1.0, 0.2, 0.6)},
    3: {"name": "moisture_level_7",  "label": "Level 7 — Slightly Wet",    "led": "Blue-Violet LED",     "color": (0.2, 0.4, 1.0)},
    4: {"name": "moisture_level_8",  "label": "Level 8 — Wet",             "led": "Strong Violet-Blue LED", "color": (0.6, 0.0, 1.0)},
}

SLIDE_BG  = "#0d1117"
TITLE_CLR = "#e6edf3"
BODY_CLR  = "#8b949e"
ACCENT    = "#58a6ff"
YELLOW    = "#d29922"
GREEN     = "#3fb950"
RED       = "#f85149"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_rgb(path):
    img = cv2.imread(str(path))
    if img is None:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def read_label(lbl_path):
    classes, boxes = [], []
    p = Path(lbl_path)
    if p.exists():
        for line in p.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) == 5:
                classes.append(int(float(parts[0])))
                boxes.append(list(map(float, parts[1:])))
    return classes, boxes

def draw_boxes(ax, img, classes, boxes, color, lw=2):
    h, w = img.shape[:2]
    ax.imshow(img)
    for cls, (cx, cy, bw, bh) in zip(classes, boxes):
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        rect = mpatches.FancyBboxPatch(
            (x1, y1), bw * w, bh * h,
            boxstyle="square,pad=0",
            linewidth=lw, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
    ax.axis("off")

def slide_fig(figsize=(16, 9)):
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(SLIDE_BG)
    return fig

def get_real_images(split, cls_idx):
    """Real captured images — no aug/flux/synthetic prefix."""
    img_dir = DATASET_ROOT / split / "images"
    lbl_dir = DATASET_ROOT / split / "labels"
    files = sorted([
        p for p in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        if not any(p.name.startswith(pfx) for pfx in ("flux_", "synthetic_", "augx"))
        and "_aug" not in p.name
    ])
    result = []
    for f in files:
        lbl = lbl_dir / (f.stem + ".txt")
        cls_list, _ = read_label(lbl)
        if cls_idx in cls_list:
            result.append(f)
    return img_dir, lbl_dir, result

def get_flux_images(split, cls_name):
    img_dir = DATASET_ROOT / split / "images"
    lbl_dir = DATASET_ROOT / split / "labels"
    files = sorted(img_dir.glob(f"flux_{cls_name}_{split}_*.jpg"))
    return img_dir, lbl_dir, list(files)

def get_all_flux_images(cls_name):
    all_files = []
    for split in ["train", "valid", "test"]:
        img_dir = DATASET_ROOT / split / "images"
        lbl_dir = DATASET_ROOT / split / "labels"
        files = sorted(img_dir.glob(f"flux_{cls_name}_{split}_*.jpg"))
        all_files.extend([(f, lbl_dir, split) for f in files])
    return all_files

# ── Slide builders ────────────────────────────────────────────────────────────

def slide_cover(pdf):
    fig = slide_fig()
    fig.text(0.5, 0.70, "FLUX.1-schnell Synthetic Data Generation",
             ha="center", color=TITLE_CLR, fontsize=32, fontweight="bold")
    fig.text(0.5, 0.58, "soil_moisture_stir_september",
             ha="center", color=ACCENT, fontsize=20, fontfamily="monospace")
    fig.text(0.5, 0.47, "5 moisture classes  ·  375 synthetic images  ·  train / valid / test splits",
             ha="center", color=BODY_CLR, fontsize=14)
    fig.text(0.5, 0.38, "Model: black-forest-labs/FLUX.1-schnell  ·  4 inference steps  ·  bfloat16",
             ha="center", color=BODY_CLR, fontsize=12)
    ax = fig.add_axes([0.08, 0.535, 0.84, 0.003])
    ax.set_facecolor(ACCENT)
    ax.axis("off")
    pdf.savefig(fig, facecolor=SLIDE_BG)
    plt.close(fig)


def slide_overview(pdf):
    """Dataset counts: real vs augmented vs FLUX per class."""
    fig = slide_fig()
    fig.text(0.5, 0.94, "Dataset Overview — Real vs Synthetic Image Counts",
             ha="center", color=TITLE_CLR, fontsize=18, fontweight="bold")
    fig.text(0.5, 0.875, "soil_moisture_stir_september  ·  train split",
             ha="center", color=BODY_CLR, fontsize=12)

    cls_names  = []
    real_counts  = []
    aug_counts   = []
    flux_counts  = []

    img_dir = DATASET_ROOT / "train" / "images"
    lbl_dir = DATASET_ROOT / "train" / "labels"

    for cls_idx, meta in CLASSES.items():
        cls_names.append(meta["label"])
        _, _, real = get_real_images("train", cls_idx)
        real_counts.append(len(real))

        aug = sorted([
            p for p in list(img_dir.glob("*.jpg"))
            if "augx" in p.name or "_aug" in p.name
        ])
        aug_cls = []
        for f in aug:
            lbl = lbl_dir / (f.stem + ".txt")
            cl, _ = read_label(lbl)
            if cls_idx in cl:
                aug_cls.append(f)
        aug_counts.append(len(aug_cls))

        _, _, flux = get_flux_images("train", meta["name"])
        flux_counts.append(len(flux))

    x = np.arange(len(cls_names))
    w = 0.25
    ax = fig.add_axes([0.08, 0.12, 0.88, 0.70])
    ax.set_facecolor(SLIDE_BG)
    ax.spines[:].set_color("#30363d")
    ax.tick_params(colors=BODY_CLR)
    ax.yaxis.label.set_color(BODY_CLR)

    b1 = ax.bar(x - w, real_counts,  w, label="Real originals", color=YELLOW,  alpha=0.9)
    b2 = ax.bar(x,     aug_counts,   w, label="Augmented (cv2)", color=ACCENT,  alpha=0.9)
    b3 = ax.bar(x + w, flux_counts,  w, label="FLUX synthetic",  color=GREEN,   alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([c.split("—")[0].strip() for c in cls_names], color=TITLE_CLR, fontsize=11)
    ax.set_ylabel("Image count", color=BODY_CLR)
    ax.legend(facecolor="#161b22", labelcolor=TITLE_CLR, fontsize=11)

    for bar in list(b1) + list(b2) + list(b3):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, str(int(h)),
                    ha="center", color=TITLE_CLR, fontsize=9)

    pdf.savefig(fig, facecolor=SLIDE_BG)
    plt.close(fig)


def slide_grid(pdf, files, lbl_dir, title, subtitle, box_color, ncols=8):
    n = len(files)
    if n == 0:
        return
    nrows = max(1, int(np.ceil(n / ncols)))
    fig = slide_fig()
    fig.text(0.5, 0.975, title,    ha="center", color=TITLE_CLR, fontsize=15, fontweight="bold")
    fig.text(0.5, 0.945, subtitle, ha="center", color=BODY_CLR,  fontsize=11)
    gs = GridSpec(nrows, ncols, figure=fig,
                  left=0.01, right=0.99, top=0.93, bottom=0.01,
                  hspace=0.05, wspace=0.03)
    for idx, img_path in enumerate(files):
        r, c = divmod(idx, ncols)
        ax = fig.add_subplot(gs[r, c])
        ax.set_facecolor(SLIDE_BG)
        img = load_rgb(img_path)
        lbl = Path(lbl_dir) / (img_path.stem + ".txt")
        cls_list, boxes = read_label(lbl)
        draw_boxes(ax, img, cls_list, boxes, box_color, lw=1.5)
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = fig.add_subplot(gs[r, c])
        ax.set_facecolor(SLIDE_BG)
        ax.axis("off")
    pdf.savefig(fig, facecolor=SLIDE_BG)
    plt.close(fig)


def slide_flux_grid(pdf, cls_info):
    all_flux = get_all_flux_images(cls_info["name"])
    if not all_flux:
        print(f"    No FLUX images found for {cls_info['name']}")
        return

    color     = cls_info["color"]
    page_size = 16
    ncols, nrows = 8, 2

    for page_start in range(0, len(all_flux), page_size):
        page      = all_flux[page_start:page_start + page_size]
        page_num  = page_start // page_size + 1
        total_pgs = int(np.ceil(len(all_flux) / page_size))

        fig = slide_fig()
        fig.text(0.5, 0.975,
                 f"FLUX Synthetic Images — {cls_info['label']}  (page {page_num}/{total_pgs})",
                 ha="center", color=TITLE_CLR, fontsize=14, fontweight="bold")
        fig.text(0.5, 0.945,
                 f"{cls_info['led']}  ·  {len(all_flux)} total FLUX images  ·  with predicted bboxes",
                 ha="center", color=BODY_CLR, fontsize=11)

        gs = GridSpec(nrows, ncols, figure=fig,
                      left=0.01, right=0.99, top=0.93, bottom=0.01,
                      hspace=0.05, wspace=0.03)

        for idx, (img_path, lbl_dir, split) in enumerate(page):
            r, c = divmod(idx, ncols)
            ax = fig.add_subplot(gs[r, c])
            ax.set_facecolor(SLIDE_BG)
            img = load_rgb(img_path)
            lbl = lbl_dir / (img_path.stem + ".txt")
            cls_list, boxes = read_label(lbl)
            draw_boxes(ax, img, cls_list, boxes, color, lw=1.5)
            ax.set_title(split, color=ACCENT, fontsize=7, pad=1)

        for idx in range(len(page), nrows * ncols):
            r, c = divmod(idx, ncols)
            ax = fig.add_subplot(gs[r, c])
            ax.set_facecolor(SLIDE_BG)
            ax.axis("off")

        pdf.savefig(fig, facecolor=SLIDE_BG)
        plt.close(fig)


def slide_side_by_side(pdf, cls_info, n_pairs=8):
    _, orig_lbl_dir, orig_files = get_real_images("train", list(CLASSES.keys())[
        [v["name"] for v in CLASSES.values()].index(cls_info["name"])
    ])
    _, flux_lbl_dir, flux_files = get_flux_images("train", cls_info["name"])

    pairs = min(n_pairs, len(orig_files), len(flux_files))
    if pairs == 0:
        return

    color = cls_info["color"]
    fig = slide_fig()
    fig.text(0.5, 0.975, f"Side-by-Side — {cls_info['label']}",
             ha="center", color=TITLE_CLR, fontsize=15, fontweight="bold")
    fig.text(0.5, 0.945,
             f"{cls_info['led']}  ·  top = real captured  ·  bottom = FLUX synthetic",
             ha="center", color=BODY_CLR, fontsize=11)

    gs = GridSpec(2, pairs, figure=fig,
                  left=0.01, right=0.99, top=0.93, bottom=0.01,
                  hspace=0.06, wspace=0.03)

    for i in range(pairs):
        ax_o = fig.add_subplot(gs[0, i])
        ax_o.set_facecolor(SLIDE_BG)
        img_o = load_rgb(orig_files[i])
        cls_o, box_o = read_label(orig_lbl_dir / (orig_files[i].stem + ".txt"))
        draw_boxes(ax_o, img_o, cls_o, box_o, color, lw=1.5)
        if i == 0:
            ax_o.set_title("REAL", color=YELLOW, fontsize=9, fontweight="bold", pad=2)

        ax_s = fig.add_subplot(gs[1, i])
        ax_s.set_facecolor(SLIDE_BG)
        img_s = load_rgb(flux_files[i])
        cls_s, box_s = read_label(flux_lbl_dir / (flux_files[i].stem + ".txt"))
        draw_boxes(ax_s, img_s, cls_s, box_s, color, lw=1.5)
        if i == 0:
            ax_s.set_title("FLUX", color=GREEN, fontsize=9, fontweight="bold", pad=2)

    pdf.savefig(fig, facecolor=SLIDE_BG)
    plt.close(fig)


def slide_summary(pdf):
    fig = slide_fig()
    fig.text(0.5, 0.88, "Summary", ha="center", color=TITLE_CLR, fontsize=28, fontweight="bold")

    lines = [
        ("Model",         "FLUX.1-schnell  (black-forest-labs)",          ACCENT),
        ("Inference",     "4 steps · bfloat16 · A100 GPU",                BODY_CLR),
        ("Target",        "375 images  (5 classes × 75,  split 50/15/10)", BODY_CLR),
        ("Generated",     "375 images  ·  0 rejected after QC revision",   GREEN),
        ("Bbox method",   "HSV colour masking → fallback per-class range",  BODY_CLR),
        ("Training result", "mAP@0.5 degraded with synthetic data included", RED),
        ("Next step",     "Remove synthetics · fine-tune from v4 weights",  YELLOW),
    ]

    y = 0.74
    for label, value, color in lines:
        fig.text(0.12, y, f"{label}:", color=BODY_CLR,  fontsize=13, fontweight="bold")
        fig.text(0.35, y, value,       color=color,      fontsize=13)
        y -= 0.09

    pdf.savefig(fig, facecolor=SLIDE_BG)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    print(f"Building slides → {OUTPUT_PDF}")

    with PdfPages(str(OUTPUT_PDF)) as pdf:

        print("  Cover")
        slide_cover(pdf)

        print("  Overview")
        slide_overview(pdf)

        for cls_idx, cls_info in CLASSES.items():
            print(f"\n  ── {cls_info['label']} ──")

            _, lbl_dir, real_files = get_real_images("train", cls_idx)
            print(f"    Real grid ({len(real_files)} images)")
            slide_grid(
                pdf, real_files, lbl_dir,
                title=f"Real Captured Images — {cls_info['label']}",
                subtitle=f"{cls_info['led']}  ·  train split  ·  {len(real_files)} original images with ground-truth bboxes",
                box_color=cls_info["color"],
            )

            print(f"    FLUX synthetic grid")
            slide_flux_grid(pdf, cls_info)

            print(f"    Side-by-side (8 pairs)")
            slide_side_by_side(pdf, cls_info, n_pairs=8)

        print("\n  Summary")
        slide_summary(pdf)

        meta = pdf.infodict()
        meta["Title"] = "FLUX.1-schnell Synthetic Data — soil_moisture_stir_september"
        meta["Author"] = "Irrigation Laser YOLO2"

    print(f"\nDone → {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
