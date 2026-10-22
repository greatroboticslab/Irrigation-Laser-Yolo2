#!/usr/bin/env python3
"""
Training Pipeline Scientific Paper Figure — YOLOv5 Soil Moisture Detection
Chromatic Succession design philosophy: severe palette, geometric precision,
monospace evidence, center-axis vertical descent.

Drawn 4.6 in wide for a full-column (3.5 in) include, so 9 pt chip text
prints at about 7 pt.

Alternative Fig. 3 (navy design), not used by the paper; Fig. 3 is make_flowchart_figure.py. Dataset names are the
paper's; the parenthetical is each dataset's nc in its data.yaml. Stages 1 and 2
start from COCO weights (yolov5s.pt); Stage 3 fine-tunes from the IR-Pot-Background
Stage 1 checkpoint.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties, fontManager
import os

FONT_DIR = "/Users/kennethpieterson/.claude/skills/canvas-design/canvas-fonts"
for fn in sorted(os.listdir(FONT_DIR)):
    if fn.endswith('.ttf'):
        fontManager.addfont(os.path.join(FONT_DIR, fn))

# ── Palette ──────────────────────────────────────────────────────────────────
NAVY    = '#1B2A4A'
GRAY_BG = '#EDF0F5'
CHIP_BG = '#CDD5E2'
WHITE   = '#FFFFFF'
ARROW_C = '#1B2A4A'
LABEL_C = '#7B8FA6'
CAPT_C  = '#A0AABB'

def fp(name, bold=False, size=10):
    s = 'Bold' if bold else 'Regular'
    p = os.path.join(FONT_DIR, f'{name}-{s}.ttf')
    return FontProperties(fname=p, size=size) if os.path.exists(p) else FontProperties(size=size)

# ── Canvas ────────────────────────────────────────────────────────────────────
FW, DPI = 4.6, 300
LX  = 0.20          # stage box left x
RW  = FW - 2 * LX   # stage box width
CX  = FW / 2        # center x
NW  = 3.30          # node width
NH  = 0.42          # node height
SHH = 0.34          # stage header height
AGH = 0.36          # arrow gap height
CH, CGAP, RGAP, PAD = 0.28, 0.10, 0.10, 0.18   # chip height, gaps, body padding
BLW = 1.0           # border line width

STAGES = [
    ('Stage I  —  Baseline (COCO init)',
     [['UV-Soil-Only-Baseline (11)', 'UV-Soil-Only-Target (11)'],
      ['IR-Soil-Only (7)', 'UV-Strawberry-Pot (9)'],
      ['IR-Pot-Background (5)', 'UV-Pot-Background (5)'],
      ['Mixed Color and Background (5)']],
     'Pool label-compatible datasets'),
    ('Stage II  —  Pooled Training (COCO init)',
     [['UV-Combined (11): UV-Soil-Only-Baseline + UV-Strawberry-Pot'],
      ['IR-Combined (5): IR-Pot-Background + UV-Pot-Background']],
     'From IR-Pot-Background best.pt'),
    ('Stage III  —  Domain-Specific Fine-Tuning',
     [['UV-Combined (11)', 'IR-Combined (5)'],
      ['IR-Soil-Only (7)', 'UV-Soil-Only-Target (11)'],
      ['Mixed Color and Background (5)']],
     None),
]

def body_h(rows):
    return 2 * PAD + len(rows) * CH + (len(rows) - 1) * RGAP

FH = 0.15 + 2 * (NH + AGH) + sum(SHH + body_h(r) for _, r, _ in STAGES) + AGH * (len(STAGES) - 1) + 0.15
fig = plt.figure(figsize=(FW, FH))
ax = fig.add_axes([0, 0, 1, 1])   # 1 data unit = 1 inch
fig.patch.set_facecolor(WHITE)
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis('off')
renderer = fig.canvas.get_renderer()

# ── Primitives ────────────────────────────────────────────────────────────────
def R(x, y, w, h, fc=WHITE, ec='none', lw=0, z=2):
    ax.add_patch(Rectangle((x, y), w, h,
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))

def node(ytop, label):
    R(CX - NW/2, ytop - NH, NW, NH, fc=NAVY, z=4)
    ax.text(CX, ytop - NH/2, label, ha='center', va='center',
            color=WHITE, fontproperties=fp('WorkSans', True, 10), zorder=5)
    return ytop - NH

def arrow_dn(ytop, label=None):
    """Downward arrow filling one gap below ytop; returns the gap's bottom."""
    yb = ytop - AGH
    ax.annotate('', xy=(CX, yb + 0.03), xytext=(CX, ytop - 0.03),
        arrowprops=dict(arrowstyle='->', color=ARROW_C, lw=1.2, mutation_scale=11), zorder=3)
    if label:
        ax.text(CX + 0.12, (ytop + yb) / 2, label, ha='left', va='center', color=LABEL_C,
                fontproperties=fp('WorkSans', False, 8), style='italic', zorder=3)
    return yb

def text_w(t, prop):
    """Rendered width of t in inches."""
    tx = ax.text(0, 0, t, fontproperties=prop)
    w = tx.get_window_extent(renderer).width / fig.dpi
    tx.remove()
    return w

def chip_row(labels, cy):
    """Centered row of dataset chips, each sized to its rendered label."""
    prop = fp('WorkSans', False, 9)
    ws = [text_w(t, prop) + 0.20 for t in labels]
    x = CX - (sum(ws) + CGAP * (len(ws) - 1)) / 2
    assert x >= LX + 0.05, f'row too wide: {labels}'
    for t, w in zip(labels, ws):
        R(x, cy - CH/2, w, CH, fc=CHIP_BG, z=5)
        ax.text(x + w/2, cy, t, ha='center', va='center',
                color=NAVY, fontproperties=prop, zorder=6)
        x += w + CGAP

def stage(ytop, title, rows):
    """Full stage: navy header over a gray body of chip rows; returns its bottom."""
    bh = body_h(rows)
    yb = ytop - SHH - bh
    R(LX, yb, RW, SHH + bh, fc='none', ec=NAVY, lw=BLW, z=6)   # outline
    R(LX, yb, RW, bh, fc=GRAY_BG, z=1)                          # body fill
    R(LX, yb + bh, RW, SHH, fc=NAVY, z=2)                       # header fill
    ax.text(CX, ytop - SHH/2, title, ha='center', va='center',
            color=WHITE, fontproperties=fp('WorkSans', True, 9.5), zorder=5)
    for i, row in enumerate(rows):
        chip_row(row, yb + bh - PAD - CH/2 - i * (CH + RGAP))
    return yb

# ── Draw (top-down) ───────────────────────────────────────────────────────────
y = node(FH - 0.15, 'COCO Pre-trained Weights (yolov5s.pt)')
y = arrow_dn(y)
for title, rows, link in STAGES:
    y = stage(y, title, rows)
    y = arrow_dn(y, link)
node(y, 'Final Detector Model')

# ── Save ──────────────────────────────────────────────────────────────────────
OUT = '/Volumes/MacOS/Downloads/Irrigation-Laser-Yolo2/results/training_pipeline_figure.png'   # not used by the paper; Fig. 3 is make_flowchart_figure.py
plt.savefig(OUT, dpi=DPI, facecolor=WHITE)
plt.close()
print(f'Saved: {OUT}')
