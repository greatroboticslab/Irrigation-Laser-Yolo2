"""
Generate a clean side-by-side figure of one representative image per
supported dataset (no title, no labels — for the Overleaf paper).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

BASE = "data/training-data/downloaded"

IMAGES = [
    f"{BASE}/soil-moisture-v4/valid/images/14_png.rf.33a5a3de8868670bbeec7e8cf5cf541d.jpg",
    f"{BASE}/soil-moisture-v4-ir/valid/images/25_png.rf.fccde6b533ea4cd3458841a8cd7491aa.jpg",
    f"{BASE}/soil-moisture-v4-uv/valid/images/110_png.rf.57ab60faad9f75db8772c4d1fc720fac.jpg",
    f"{BASE}/soil-moisture-ir/valid/images/WIN_20231111_14_37_25_Pro_jpg.rf.2745faae8dcaa5cfa1900648c698512a.jpg",
    f"{BASE}/soil-moisture-5sagf/valid/images/WIN_20231111_14_41_03_Pro_jpg.rf.6e4c746515d5ecbad658ad2456b57560.jpg",
    f"{BASE}/soil_moisture_september/valid/images/1_png.rf.77df9dfb0267f612a88016fd8db56d78.jpg",
    f"{BASE}/soil_moisture_stir_september/valid/images/16_png.rf.327d1ff75f9fbe31f276e0f2507a9021.jpg",
]

# ── Layout: 4 on top row, 3 on bottom row ────────────────────────────────────
n_top = 4
n_bot = 3
cell_size = 2.5   # inches per cell
gap = 0.04        # fraction of cell as gap

fig_w = n_top * cell_size
fig_h = 2 * cell_size

fig = plt.figure(figsize=(fig_w, fig_h))
fig.patch.set_facecolor("white")

def add_image(fig, img_path, left, bottom, width, height):
    ax = fig.add_axes([left, bottom, width, height])
    img = Image.open(img_path).convert("RGB")
    ax.imshow(img)
    ax.axis("off")

cell_w = 1.0 / n_top
cell_h = 0.5   # each row = half the figure height

pad_x = gap * cell_w
pad_y = gap * cell_h

# Top row: 4 images
for i, img_path in enumerate(IMAGES[:n_top]):
    left   = i * cell_w + pad_x
    bottom = cell_h + pad_y
    width  = cell_w - 2 * pad_x
    height = cell_h - 2 * pad_y
    add_image(fig, img_path, left, bottom, width, height)

# Bottom row: 3 images, centred under the top row
offset = (n_top - n_bot) / 2 * cell_w
for i, img_path in enumerate(IMAGES[n_top:]):
    left   = offset + i * cell_w + pad_x
    bottom = pad_y
    width  = cell_w - 2 * pad_x
    height = cell_h - 2 * pad_y
    add_image(fig, img_path, left, bottom, width, height)

os.makedirs("plots", exist_ok=True)
out_png = "plots/dataset_overview.png"
out_pdf = "plots/dataset_overview.pdf"

fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
