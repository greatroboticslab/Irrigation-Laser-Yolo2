#!/usr/bin/env python3
"""
Training Pipeline Scientific Paper Figure — YOLOv5 Soil Moisture Detection
Chromatic Succession design philosophy: severe palette, geometric precision,
monospace evidence, center-axis vertical descent.
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
FW, FH, DPI = 7.0, 9.2, 200
fig, ax = plt.subplots(figsize=(FW, FH))
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(WHITE)
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis('off')

# ── Layout constants ──────────────────────────────────────────────────────────
LX  = 0.55      # stage box left x
RW  = 5.90      # stage box width
CX  = LX + RW / 2   # center x = 3.50
NW  = 3.70      # node width
NX  = CX - NW / 2   # node left x = 1.65
NH  = 0.50      # node height
SHH = 0.42      # stage header height
SBH = 1.52      # stage body height
AGH = 0.40      # arrow gap height
BLW = 1.2       # border line width

# ── Y positions (built bottom-up) ─────────────────────────────────────────────
Y_CAP  = 0.15
Y_BN   = 0.45               # Final Model node bottom
Y_BN_T = Y_BN  + NH         # 0.95  Final Model node top
Y_S3   = Y_BN_T + AGH       # 1.35  Stage III body bottom
Y_S3_T = Y_S3  + SBH + SHH  # 3.19  Stage III header top
Y_S2   = Y_S3_T + AGH       # 3.59  Stage II body bottom
Y_S2_T = Y_S2  + SBH + SHH  # 5.43  Stage II header top
Y_S1   = Y_S2_T + AGH       # 5.83  Stage I body bottom
Y_S1_T = Y_S1  + SBH + SHH  # 7.67  Stage I header top
Y_TN   = Y_S1_T + AGH       # 8.07  ImageNet node bottom
Y_TN_T = Y_TN  + NH         # 8.57  ImageNet node top

# ── Primitives ────────────────────────────────────────────────────────────────
def R(x, y, w, h, fc=WHITE, ec='none', lw=0, z=2):
    ax.add_patch(Rectangle((x, y), w, h,
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))

def node(y, label):
    R(NX, y, NW, NH, fc=NAVY, z=4)
    ax.text(CX, y + NH/2, label, ha='center', va='center',
            color=WHITE, fontproperties=fp('WorkSans', True, 10.5), zorder=5)

def arrow_dn(yf, yt, label=None):
    """Downward arrow: tail at yf (higher), head at yt (lower)."""
    ax.annotate('',
        xy=(CX, yt + 0.04),
        xytext=(CX, yf - 0.04),
        arrowprops=dict(arrowstyle='->', color=ARROW_C,
                        lw=1.5, mutation_scale=14), zorder=3)
    if label:
        ax.text(CX + 0.22, (yf + yt) / 2, label,
                ha='left', va='center', color=LABEL_C,
                fontproperties=fp('WorkSans', False, 7.5),
                style='italic', zorder=3)

def chip_row(labels, cy):
    """Centered row of monospace-labelled dataset chips."""
    CW = 0.057; HP = 0.15; GAP = 0.13; CH = 0.30
    ws = [len(t) * CW + 2 * HP for t in labels]
    tot = sum(ws) + GAP * (len(labels) - 1)
    avail = RW - 0.50
    if tot > avail:
        scale = avail / tot
        ws = [w * scale for w in ws]
        tot = avail
    x = CX - tot / 2
    for t, w in zip(labels, ws):
        R(x, cy - CH/2, w, CH, fc=CHIP_BG, z=5)
        ax.text(x + w/2, cy, t, ha='center', va='center',
                color=NAVY, fontproperties=fp('IBMPlexMono', False, 7.5), zorder=6)
        x += w + GAP

def stage(yb, title, row1, row2=None):
    """Full stage: outline → gray body → navy header → chips."""
    R(LX, yb, RW, SBH + SHH, fc='none', ec=NAVY, lw=BLW, z=6)   # outline
    R(LX, yb, RW, SBH, fc=GRAY_BG, z=1)                           # body fill
    R(LX, yb + SBH, RW, SHH, fc=NAVY, z=2)                        # header fill
    ax.text(CX, yb + SBH + SHH/2, title, ha='center', va='center',
            color=WHITE, fontproperties=fp('WorkSans', True, 9.0), zorder=5)
    CH = 0.30; GAP = 0.18
    if row2:
        two_h = 2 * CH + GAP
        base = yb + (SBH - two_h) / 2
        chip_row(row2, base + CH/2)
        chip_row(row1, base + CH + GAP + CH/2)
    else:
        chip_row(row1, yb + SBH/2)

# ── Draw ──────────────────────────────────────────────────────────────────────
# Caption (below everything)
ax.text(CX, Y_CAP,
    'Fig. X.  Three-stage multi-spectral training pipeline for YOLOv5-based soil moisture detection.',
    ha='center', va='bottom', color=CAPT_C,
    fontproperties=fp('IBMPlexMono', False, 6.5), zorder=3)

# Final Model node
node(Y_BN, 'Final Detector Model')

# Stage III → Final Model
arrow_dn(Y_S3, Y_BN_T)

# Stage III
stage(Y_S3,
    'Stage III  —  Combined Fine-Tuning',
    ['v4-sep combined', 'ir-combined', 'stir-sep (5c)'],
    ['v4-ir (11c)', 'v4-uv (11c)'])

# Stage II → Stage III
arrow_dn(Y_S2, Y_S3_T, 'Best Weights')

# Stage II
stage(Y_S2,
    'Stage II  —  Domain Adaptation',
    ['September (11c)', 'ir (5c)', 'v4 (11c)', '5sagf (11c)'],
    ['v4-sep combined', 'ir-combined'])

# Stage I → Stage II
arrow_dn(Y_S1, Y_S2_T, 'Best Weights  (ir / best.pt)')

# Stage I
stage(Y_S1,
    'Stage I  —  Multi-Spectral Base Training',
    ['v4 (11c)', 'v4-ir (11c)', 'v4-uv (11c)', 'ir (5c)'],
    ['5sagf (11c)', 'September (11c)', 'stir-sep (5c)'])

# ImageNet → Stage I
arrow_dn(Y_TN, Y_S1_T)

# ImageNet node
node(Y_TN, 'ImageNet Pre-trained Weights')

# ── Save ──────────────────────────────────────────────────────────────────────
OUT = '/Volumes/MacOS/Downloads/Irrigation-Laser-Yolo2/training_pipeline_figure.png'
plt.savefig(OUT, dpi=DPI, bbox_inches='tight', facecolor=WHITE, pad_inches=0.18)
plt.close()
print(f'Saved: {OUT}')
