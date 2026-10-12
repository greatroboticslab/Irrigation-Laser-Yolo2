"""
Generate a side-by-side figure showing LED appearances for all 5 moisture
levels in the soil_moisture_stir_september dataset, for the Overleaf paper.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

BASE = "data/training-data/downloaded/soil_moisture_stir_september/valid"

# (moisture_label, image_path, yolo_label [cx, cy, w, h])
# Classes (alphabetical): 0='1', 1='10', 2='3', 3='7', 4='8'
ENTRIES = [
    (
        "1",
        f"{BASE}/images/6_png.rf.73af7efb84c2e082b226d66b338ab3a8.jpg",
        (0.490625, 0.615625, 0.11328125, 0.2578125),
    ),
    (
        "3",
        f"{BASE}/images/1_png.rf.98855c58e3c337c2139e39873b590b77.jpg",
        (0.59765625, 0.62265625, 0.0390625, 0.05234375),
    ),
    (
        "7",
        f"{BASE}/images/16_png.rf.327d1ff75f9fbe31f276e0f2507a9021.jpg",
        (0.48515625, 0.59921875, 0.0484375, 0.06796875),
    ),
    (
        "8",
        f"{BASE}/images/18_png.rf.158f059f3b9821784c92480ae6ed4233.jpg",
        (0.484375, 0.640625, 0.13671875, 0.30703125),
    ),
    (
        "10",
        f"{BASE}/images/24_png.rf.dbe350635c572acd77788d16278721cf.jpg",
        (0.5640625, 0.50390625, 0.0546875, 0.06015625),
    ),
]

# Padding factor around the bounding box (multiplier on the larger dimension)
PAD = 3.0
CROP_SIZE = 200  # pixels: square output crop for each panel


def crop_led(img_path, cx_n, cy_n, bw_n, bh_n, pad=PAD, out_size=CROP_SIZE):
    """Crop a square region around the LED centre with generous padding."""
    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    # Convert normalised YOLO coords to pixels
    cx = cx_n * W
    cy = cy_n * H
    bw = bw_n * W
    bh = bh_n * H

    # Half-side of the padded square crop
    half = max(bw, bh) * pad / 2.0
    half = max(half, 30)          # minimum 30px half-side

    x0 = int(max(0, cx - half))
    y0 = int(max(0, cy - half))
    x1 = int(min(W, cx + half))
    y1 = int(min(H, cy + half))

    crop = img.crop((x0, y0, x1, y1))
    crop = crop.resize((out_size, out_size), Image.LANCZOS)
    return np.array(crop)


# ── Figure layout ─────────────────────────────────────────────────────────────
n = len(ENTRIES)
fig_w = 2.5 * n          # inches
fig_h = 3.4               # inches  (panel + label row)

fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h),
                         gridspec_kw={"wspace": 0.06})

for ax, (moisture, img_path, (cx, cy, bw, bh)) in zip(axes, ENTRIES):
    crop = crop_led(img_path, cx, cy, bw, bh)

    ax.imshow(crop)
    ax.set_xticks([])
    ax.set_yticks([])

    # Thin white border for visual separation
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(1.2)

    ax.set_xlabel(
        f"Moisture: {moisture}",
        fontsize=11,
        fontweight="bold",
        color="white",
        labelpad=4,
    )

fig.patch.set_facecolor("#111111")   # dark background looks good for LED images

# Overall title
fig.suptitle(
    "Soil Moisture Sensor LED Appearance by Moisture Level\n"
    "soil_moisture_stir_september dataset",
    fontsize=11,
    color="white",
    y=1.01,
)

out_path = "plots/led_appearance_soil_moisture_stir_september.pdf"
os.makedirs("plots", exist_ok=True)
fig.savefig(
    out_path,
    dpi=300,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)

# Also save a PNG preview
png_path = out_path.replace(".pdf", ".png")
fig.savefig(
    png_path,
    dpi=200,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)

print(f"Saved: {out_path}")
print(f"Saved: {png_path}")
plt.close(fig)
