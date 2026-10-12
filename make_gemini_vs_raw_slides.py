"""
make_gemini_vs_raw_slides.py
Generates a PDF slide deck comparing raw captured images vs Gemini-generated
synthetic images for the soil moisture dataset.

Output: documents/gemini_vs_raw_slides.pdf
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

RAW_ROOT    = Path("data/soil_september")
SYNTH_ROOT  = Path("data/training-data/downloaded/soil_moisture_stir_september")
OUTPUT_PDF  = Path("documents/gemini_vs_raw_slides.pdf")

SYNTH_CLASSES = {
    0: {"name": "moisture_level_1",  "label": "Level 1 — Very Dry",  "led": "Red/Orange LED",  "color": (1.0, 0.35, 0.0)},
    1: {"name": "moisture_level_10", "label": "Level 10 — Very Wet", "led": "Blue-Violet LED", "color": (0.45, 0.0, 0.9)},
}

RAW_COLOR  = (0.2, 0.8, 0.4)   # green for raw bboxes
SLIDE_BG   = "#0d1117"
TITLE_CLR  = "#e6edf3"
BODY_CLR   = "#8b949e"
ACCENT     = "#58a6ff"
YELLOW     = "#d29922"
GREEN      = "#3fb950"

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

def get_raw_images(split="train"):
    img_dir = RAW_ROOT / split / "images"
    lbl_dir = RAW_ROOT / split / "labels"
    files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    return img_dir, lbl_dir, files

def get_synth_images(split, cls_name):
    img_dir = SYNTH_ROOT / split / "images"
    lbl_dir = SYNTH_ROOT / split / "labels"
    files = sorted(img_dir.glob(f"synthetic_{cls_name}_{split}_*.jpg"))
    return img_dir, lbl_dir, list(files)

# ── Slide builders ────────────────────────────────────────────────────────────

def slide_cover(pdf):
    fig = slide_fig()
    fig.text(0.5, 0.68, "soil_moisture_stir_september  ·  soil_september",
             ha="center", color=ACCENT, fontsize=16, fontweight="bold", fontfamily="monospace")
    fig.text(0.5, 0.55, "Raw Captured vs Gemini-Generated Images",
             ha="center", color=TITLE_CLR, fontsize=34, fontweight="bold")
    fig.text(0.5, 0.43, "moisture_level_1  (Very Dry)   ·   moisture_level_10  (Very Wet)",
             ha="center", color=BODY_CLR, fontsize=14)
    fig.text(0.5, 0.34, "Synthetic images generated with Google Imagen 4  ·  imagen-4.0-generate-001",
             ha="center", color=BODY_CLR, fontsize=12)
    ax = fig.add_axes([0.08, 0.48, 0.84, 0.003])
    ax.set_facecolor(ACCENT)
    ax.axis("off")
    pdf.savefig(fig, facecolor=SLIDE_BG)
    plt.close(fig)


def slide_grid(pdf, files, lbl_dir, title, subtitle, box_color, ncols=8):
    n = len(files)
    if n == 0:
        return
    nrows = int(np.ceil(n / ncols))

    fig = slide_fig(figsize=(16, 9))
    fig.text(0.5, 0.975, title, ha="center", color=TITLE_CLR,
             fontsize=16, fontweight="bold")
    fig.text(0.5, 0.945, subtitle, ha="center", color=BODY_CLR, fontsize=11)

    gs = GridSpec(nrows, ncols, figure=fig,
                  left=0.01, right=0.99, top=0.93, bottom=0.01,
                  hspace=0.05, wspace=0.03)

    for idx, img_path in enumerate(files):
        r, c = divmod(idx, ncols)
        ax = fig.add_subplot(gs[r, c])
        ax.set_facecolor(SLIDE_BG)
        img = load_rgb(img_path)
        lbl = lbl_dir / (img_path.stem + ".txt")
        cls_list, boxes = read_label(lbl)
        draw_boxes(ax, img, cls_list, boxes, box_color, lw=1.5)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = fig.add_subplot(gs[r, c])
        ax.set_facecolor(SLIDE_BG)
        ax.axis("off")

    pdf.savefig(fig, facecolor=SLIDE_BG)
    plt.close(fig)


def slide_side_by_side(pdf, raw_files, raw_lbl_dir, synth_files, synth_lbl_dir,
                       cls_info, n_pairs=8):
    pairs = min(n_pairs, len(raw_files), len(synth_files))
    if pairs == 0:
        return
    synth_color = cls_info["color"]

    fig = slide_fig()
    fig.text(0.5, 0.975, f"Side-by-Side Comparison — {cls_info['label']}",
             ha="center", color=TITLE_CLR, fontsize=15, fontweight="bold")
    fig.text(0.5, 0.945,
             f"{cls_info['led']}  ·  top row = raw captured  ·  bottom row = Gemini synthetic",
             ha="center", color=BODY_CLR, fontsize=11)

    gs = GridSpec(2, pairs, figure=fig,
                  left=0.01, right=0.99, top=0.93, bottom=0.01,
                  hspace=0.06, wspace=0.03)

    for i in range(pairs):
        ax_r = fig.add_subplot(gs[0, i])
        ax_r.set_facecolor(SLIDE_BG)
        img_r = load_rgb(raw_files[i])
        lbl_r = raw_lbl_dir / (raw_files[i].stem + ".txt")
        cls_r, box_r = read_label(lbl_r)
        draw_boxes(ax_r, img_r, cls_r, box_r, RAW_COLOR, lw=1.5)
        if i == 0:
            ax_r.set_title("RAW", color=YELLOW, fontsize=9, fontweight="bold", pad=2)

        ax_s = fig.add_subplot(gs[1, i])
        ax_s.set_facecolor(SLIDE_BG)
        img_s = load_rgb(synth_files[i])
        lbl_s = synth_lbl_dir / (synth_files[i].stem + ".txt")
        cls_s, box_s = read_label(lbl_s)
        draw_boxes(ax_s, img_s, cls_s, box_s, synth_color, lw=1.5)
        if i == 0:
            ax_s.set_title("GEMINI", color=GREEN, fontsize=9, fontweight="bold", pad=2)

    pdf.savefig(fig, facecolor=SLIDE_BG)
    plt.close(fig)


def slide_synth_pages(pdf, cls_info):
    cls_name = cls_info["name"]
    color    = cls_info["color"]

    all_synth = []
    for split in ["train", "valid", "test"]:
        img_dir = SYNTH_ROOT / split / "images"
        lbl_dir = SYNTH_ROOT / split / "labels"
        files   = sorted(img_dir.glob(f"synthetic_{cls_name}_{split}_*.jpg"))
        all_synth.extend([(f, lbl_dir) for f in files])

    if not all_synth:
        return

    page_size = 16
    ncols, nrows = 8, 2

    for page_start in range(0, len(all_synth), page_size):
        page = all_synth[page_start:page_start + page_size]
        page_num    = page_start // page_size + 1
        total_pages = int(np.ceil(len(all_synth) / page_size))

        fig = slide_fig()
        fig.text(0.5, 0.975,
                 f"All Gemini Synthetic — {cls_info['label']}  (page {page_num}/{total_pages})",
                 ha="center", color=TITLE_CLR, fontsize=14, fontweight="bold")
        fig.text(0.5, 0.945,
                 f"{cls_info['led']}  ·  {len(all_synth)} total synthetic images",
                 ha="center", color=BODY_CLR, fontsize=11)

        gs = GridSpec(nrows, ncols, figure=fig,
                      left=0.01, right=0.99, top=0.93, bottom=0.01,
                      hspace=0.05, wspace=0.03)

        for idx, (img_path, lbl_dir) in enumerate(page):
            r, c = divmod(idx, ncols)
            ax = fig.add_subplot(gs[r, c])
            ax.set_facecolor(SLIDE_BG)
            img = load_rgb(img_path)
            lbl = lbl_dir / (img_path.stem + ".txt")
            cls_list, boxes = read_label(lbl)
            draw_boxes(ax, img, cls_list, boxes, color, lw=1.5)
            parts = img_path.stem.split("_")
            ax.set_title(f"{parts[-2]} {parts[-1]}", color=ACCENT, fontsize=7, pad=1)

        for idx in range(len(page), nrows * ncols):
            r, c = divmod(idx, ncols)
            ax = fig.add_subplot(gs[r, c])
            ax.set_facecolor(SLIDE_BG)
            ax.axis("off")

        pdf.savefig(fig, facecolor=SLIDE_BG)
        plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    print(f"Building slides → {OUTPUT_PDF}")

    _, raw_lbl_dir, raw_files = get_raw_images("train")
    print(f"  Raw images (train): {len(raw_files)}")

    with PdfPages(str(OUTPUT_PDF)) as pdf:

        print("  Slide: Cover")
        slide_cover(pdf)

        print("  Slide: Raw images grid")
        slide_grid(
            pdf, raw_files, raw_lbl_dir,
            title="Raw Captured Images — soil_september (train split)",
            subtitle=f"Ground-truth bboxes  ·  {len(raw_files)} images",
            box_color=RAW_COLOR,
        )

        for cls_idx, cls_info in SYNTH_CLASSES.items():
            cls_name = cls_info["name"]
            label    = cls_info["label"]
            print(f"\n  ── {label} ──")

            print(f"  Gemini synthetic grids")
            slide_synth_pages(pdf, cls_info)

            _, synth_lbl_dir, synth_files = get_synth_images("train", cls_name)
            print(f"  Side-by-side ({min(8, len(raw_files), len(synth_files))} pairs)")
            slide_side_by_side(
                pdf,
                raw_files, raw_lbl_dir,
                synth_files, synth_lbl_dir,
                cls_info, n_pairs=8,
            )

        meta = pdf.infodict()
        meta["Title"] = "Raw Captured vs Gemini-Generated — soil moisture"

    print(f"\nDone — {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
