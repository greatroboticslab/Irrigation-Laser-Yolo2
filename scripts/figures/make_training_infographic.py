"""
Generate an Excalidraw infographic for the 3-stage YOLOv5m training protocol.
Open plots/training_protocol.excalidraw at https://excalidraw.com
"""

import json, os, random
random.seed(42)

elems = []

# ── helpers ───────────────────────────────────────────────────────────────────

def _uid():
    return format(random.randint(0, 0xFFFFFFFF), "08x")

def _seed():
    return random.randint(0, 999999)

def _base(etype, x, y, w, h):
    return dict(
        id=_uid(), type=etype,
        x=x, y=y, width=w, height=h, angle=0,
        strokeColor="#1e293b", backgroundColor="transparent",
        fillStyle="solid", strokeWidth=2, strokeStyle="solid",
        roughness=0, opacity=100,
        groupIds=[], frameId=None, roundness=None,
        seed=_seed(), version=1, versionNonce=_seed(),
        isDeleted=False, boundElements=[], updated=1700000000000,
        link=None, locked=False,
    )

def box(x, y, w, h, bg="transparent", stroke="#1e293b", sw=2, r=0):
    e = _base("rectangle", x, y, w, h)
    e.update(backgroundColor=bg, strokeColor=stroke, strokeWidth=sw,
              roundness={"type": 3, "value": r} if r else None)
    elems.append(e); return e

def txt(x, y, w, content, size=14, color="#1e293b", align="left"):
    lines = content.count("\n") + 1
    h = max(lines * size * 1.4, size * 1.4)
    e = _base("text", x, y, w, h)
    e.update(strokeColor=color, text=content, originalText=content,
              fontSize=size, fontFamily=2,
              textAlign=align, verticalAlign="top",
              containerId=None, lineHeight=1.35)
    elems.append(e); return e

def hline(x, y, length, color="#cbd5e1", sw=1):
    e = _base("line", x, y, length, 1)
    e.update(strokeColor=color, strokeWidth=sw,
              points=[[0, 0], [length, 0]])
    elems.append(e); return e

def arrow_h(x1, y, x2, color="#374151", sw=3):
    dx = x2 - x1
    e = _base("arrow", x1, y, abs(dx), 1)
    e.update(strokeColor=color, strokeWidth=sw,
              roundness={"type": 2},
              points=[[0, 0], [dx, 0]],
              lastCommittedPoint=None,
              startBinding=None, endBinding=None,
              startArrowhead=None, endArrowhead="arrow")
    elems.append(e); return e

def arrow_elbow(x1, y1, x2, y2, color="#6366f1", sw=2, dashed=True):
    """Elbow arrow: right then down (or any L-shape)."""
    dx, dy = x2 - x1, y2 - y1
    e = _base("arrow", x1, y1, abs(dx), abs(dy))
    e.update(strokeColor=color, strokeWidth=sw,
              strokeStyle="dashed" if dashed else "solid",
              roundness={"type": 2},
              points=[[0, 0], [dx, 0], [dx, dy]],
              lastCommittedPoint=None,
              startBinding=None, endBinding=None,
              startArrowhead=None, endArrowhead="arrow")
    elems.append(e); return e

def pill(x, y, label, bg, stroke, tc, pill_h=28):
    """Pill badge. Returns consumed width."""
    w = int(len(label) * 7.8 + 20)
    box(x, y, w, pill_h, bg=bg, stroke=stroke, sw=1.5, r=pill_h // 2)
    txt(x + 10, y + (pill_h - 14) // 2, w - 20, label, size=12, color=tc)
    return w

def section_hdr(x, y, w, label, color):
    txt(x, y, w, label, size=11, color=color, align="left")
    hline(x, y + 17, w, color=color + "88" if len(color) == 7 else color, sw=1)
    return y + 24

def stage_header(sx, sy, sw, sh, num_label, title, subtitle, hdr_color, txt_light):
    """Draw stage box + colored header band."""
    # main container
    box(sx, sy, sw, sh, bg=STAGE_BG[num_label - 1],
        stroke=STAGE_STROKE[num_label - 1], sw=2.5, r=14)
    # header band (simulate by overlapping rects: top-rounded main rect + flat bottom rect)
    HDRH = 72
    box(sx, sy, sw, HDRH, bg=hdr_color, stroke=hdr_color, sw=0, r=14)
    box(sx, sy + HDRH // 2, sw, HDRH // 2, bg=hdr_color, stroke=hdr_color, sw=0)
    # Stage badge circle
    R = 22
    badge = _base("ellipse", sx + 18, sy + 14, R * 2, R * 2)
    badge.update(backgroundColor="#ffffff33", strokeColor="#ffffff",
                 strokeWidth=2, roughness=0)
    elems.append(badge)
    txt(sx + 18, sy + 19, R * 2, str(num_label), size=18, color="#ffffff", align="center")
    # Title + subtitle
    txt(sx + 18 + R * 2 + 10, sy + 12, sw - 80, title, size=18, color="#ffffff")
    txt(sx + 18 + R * 2 + 10, sy + 36, sw - 80, subtitle, size=11, color=txt_light)
    return sy + HDRH + 16  # return y cursor after header

# ── design tokens ─────────────────────────────────────────────────────────────

STAGE_BG     = ["#eff6ff", "#f0fdf4", "#fff7ed"]
STAGE_STROKE = ["#3b82f6", "#22c55e", "#f97316"]
HDR_COLOR    = ["#1d4ed8", "#15803d", "#c2410c"]
HDR_LIGHT    = ["#93c5fd", "#86efac", "#fdba74"]
SECTION_CLR  = ["#1e3a8a", "#14532d", "#7c2d12"]
SUB_CLR      = ["#2563eb", "#16a34a", "#ea580c"]

SW, SH = 580, 550        # stage box size
GAP    = 110             # gap between stages
SX     = [60, 60 + SW + GAP, 60 + 2 * (SW + GAP)]
SY     = 130

SETTINGS = "YOLOv5m (medium)  ·  AdamW  ·  Cosine LR\n200 epochs  ·  patience 100  ·  img_size 640"

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — Baseline
# ─────────────────────────────────────────────────────────────────────────────
s, hc, hl, sc, subc = 0, HDR_COLOR[0], HDR_LIGHT[0], SECTION_CLR[0], SUB_CLR[0]
x1 = SX[s]
cy = stage_header(x1, SY, SW, SH, 1, "Baseline Training", "slurm/run_tr.slurm",
                  hc, hl)

# ImageNet source badge above Stage 1
IW, IH = 220, 38
IX, IY = x1 + SW // 2 - IW // 2, SY - 68
box(IX, IY, IW, IH, bg="#f8fafc", stroke="#94a3b8", sw=2, r=8)
txt(IX, IY + 10, IW, "ImageNet Pretrained Weights", size=13, color="#475569", align="center")
arrow_h(x1 + SW // 2, IY + IH, x1 + SW // 2, color="#94a3b8", sw=2)

# ← fix: arrow for badge should be vertical, use elbow helper differently
#    Just draw a line element
line_e = _base("line", x1 + SW // 2, IY + IH, 1, SY - IY - IH)
line_e.update(strokeColor="#94a3b8", strokeWidth=2,
              points=[[0, 0], [0, SY - IY - IH + 2]],
              startArrowhead=None, endArrowhead="arrow")
elems.append(line_e)

cy = section_hdr(x1 + 20, cy, SW - 40, "INPUT — 7 individual datasets", sc)
cy += 4

# Dataset pills in 2 columns
datasets_1 = [
    ("v4  (11c)",          "#dbeafe", "#1d4ed8", "#1e3a8a"),
    ("v4-ir  (11c)",       "#ede9fe", "#7c3aed", "#4c1d95"),
    ("v4-uv  (11c)",       "#fce7f3", "#be185d", "#831843"),
    ("ir  (5c)",           "#e0f2fe", "#0369a1", "#0c4a6e"),
    ("5sagf  (5c)",        "#ecfeff", "#0e7490", "#164e63"),
    ("september  (11c)",   "#f0fdf4", "#15803d", "#14532d"),
    ("stir-sep  (5c)",     "#fef3c7", "#b45309", "#78350f"),
]
col_w = (SW - 40) // 2 - 6
for i, (name, pbg, pstk, ptc) in enumerate(datasets_1):
    col = i % 2
    row = i // 2
    pill(x1 + 20 + col * (col_w + 12), cy + row * 34, name, pbg, pstk, ptc, pill_h=26)

cy += ((len(datasets_1) - 1) // 2 + 1) * 34 + 8

hline(x1 + 20, cy, SW - 40, color="#bfdbfe"); cy += 14

cy = section_hdr(x1 + 20, cy, SW - 40, "HYPERPARAMETERS", sc)
txt(x1 + 20, cy, SW - 40, SETTINGS, size=12, color=subc); cy += 46

hline(x1 + 20, cy, SW - 40, color="#bfdbfe"); cy += 14

cy = section_hdr(x1 + 20, cy, SW - 40, "OUTPUT", sc)
txt(x1 + 20, cy, SW - 40, "7 × best.pt", size=16, color=hc); cy += 28
txt(x1 + 20, cy, SW - 40, "runs/train/downloaded/<dataset>/weights/",
    size=11, color="#64748b")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — Merged Dataset Training
# ─────────────────────────────────────────────────────────────────────────────
s, hc, hl, sc, subc = 1, HDR_COLOR[1], HDR_LIGHT[1], SECTION_CLR[1], SUB_CLR[1]
x2 = SX[s]
cy = stage_header(x2, SY, SW, SH, 2, "Merged Dataset Training",
                  "scripts/data_prep/merge_datasets.py  →  slurm/run_tr_merged.slurm", hc, hl)

cy = section_hdr(x2 + 20, cy, SW - 40, "DATASET MERGES", sc)
cy += 4

# Merge block helper
def merge_block(x, y, src_a, src_b, result, clr_a, clr_b, clr_r, nc, note_color):
    wa = pill(x, y, src_a, clr_a[0], clr_a[1], clr_a[2], 26)
    txt(x + wa + 6, y + 5, 18, "+", size=15, color="#374151")
    pill(x + wa + 28, y, src_b, clr_b[0], clr_b[1], clr_b[2], 26)
    y2 = y + 34
    txt(x, y2, 20, "→", size=16, color=note_color)
    pill(x + 24, y2, result, clr_r[0], clr_r[1], clr_r[2], 26)
    txt(x + 24 + int(len(result) * 7.8 + 22), y2 + 6,
        60, f"({nc}c)", size=11, color="#64748b")
    return y2 + 34

cy = merge_block(
    x2 + 20, cy,
    "5sagf", "soil-ir",
    "ir-combined",
    ("#ecfeff", "#0e7490", "#164e63"),
    ("#e0f2fe", "#0369a1", "#0c4a6e"),
    ("#cffafe", "#0891b2", "#0e4458"),
    5, hc
)
cy += 6
hline(x2 + 20, cy, SW - 40, color="#bbf7d0"); cy += 10

cy = merge_block(
    x2 + 20, cy,
    "v4", "september",
    "v4-sep-combined",
    ("#dbeafe", "#1d4ed8", "#1e3a8a"),
    ("#f0fdf4", "#15803d", "#14532d"),
    ("#d1fae5", "#059669", "#064e3b"),
    11, hc
)
cy += 4

hline(x2 + 20, cy, SW - 40, color="#bbf7d0"); cy += 14

cy = section_hdr(x2 + 20, cy, SW - 40, "HYPERPARAMETERS", sc)
txt(x2 + 20, cy, SW - 40, SETTINGS, size=12, color=subc); cy += 46

hline(x2 + 20, cy, SW - 40, color="#bbf7d0"); cy += 14

cy = section_hdr(x2 + 20, cy, SW - 40, "OUTPUT", sc)
txt(x2 + 20, cy, SW - 40, "2 × merged best.pt", size=16, color=hc); cy += 28
txt(x2 + 20, cy, SW - 40, "runs/train/merged/<dataset>/weights/",
    size=11, color="#64748b")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — Fine-Tuning
# ─────────────────────────────────────────────────────────────────────────────
s, hc, hl, sc, subc = 2, HDR_COLOR[2], HDR_LIGHT[2], SECTION_CLR[2], SUB_CLR[2]
x3 = SX[s]
cy = stage_header(x3, SY, SW, SH, 3, "Fine-Tuning",
                  "slurm/run_fine_tuning.slurm", hc, hl)

cy = section_hdr(x3 + 20, cy, SW - 40, "BASE WEIGHTS  (from Stage 1)", sc)
# Highlighted base weights box
box(x3 + 20, cy, SW - 40, 38, bg="#ffedd5", stroke=hc, sw=2, r=8)
txt(x3 + 20, cy + 10, SW - 40,
    "soil-moisture-v4 / best.pt",
    size=14, color=hc, align="center")
cy += 48

hline(x3 + 20, cy, SW - 40, color="#fdba74"); cy += 14

cy = section_hdr(x3 + 20, cy, SW - 40, "APPLIED TO — 8 datasets", sc)
cy += 4

datasets_3 = [
    ("v4",              "#dbeafe", "#1d4ed8", "#1e3a8a"),
    ("v4-ir",           "#ede9fe", "#7c3aed", "#4c1d95"),
    ("v4-uv",           "#fce7f3", "#be185d", "#831843"),
    ("ir-combined",     "#cffafe", "#0891b2", "#0e4458"),
    ("5sagf",           "#ecfeff", "#0e7490", "#164e63"),
    ("september",       "#f0fdf4", "#15803d", "#14532d"),
    ("v4-sep-combined", "#d1fae5", "#059669", "#064e3b"),
    ("stir-sep",        "#fef3c7", "#b45309", "#78350f"),
]
col_w = (SW - 40) // 2 - 6
for i, (name, pbg, pstk, ptc) in enumerate(datasets_3):
    col = i % 2
    row = i // 2
    pill(x3 + 20 + col * (col_w + 12), cy + row * 32, name, pbg, pstk, ptc, pill_h=26)

cy += ((len(datasets_3) - 1) // 2 + 1) * 32 + 8

hline(x3 + 20, cy, SW - 40, color="#fdba74"); cy += 14

cy = section_hdr(x3 + 20, cy, SW - 40, "HYPERPARAMETERS", sc)
txt(x3 + 20, cy, SW - 40, SETTINGS, size=12, color=subc); cy += 46

hline(x3 + 20, cy, SW - 40, color="#fdba74"); cy += 14

cy = section_hdr(x3 + 20, cy, SW - 40, "OUTPUT", sc)
txt(x3 + 20, cy, SW - 40, "best.pt per dataset", size=16, color=hc); cy += 28
txt(x3 + 20, cy, SW - 40, "runs/fine-tuned/<dataset>/weights/",
    size=11, color="#64748b")

# ─────────────────────────────────────────────────────────────────────────────
# Flow arrows between stages
# ─────────────────────────────────────────────────────────────────────────────
MID_Y = SY + SH // 2

# Stage 1 → Stage 2
arrow_h(x1 + SW + 8, MID_Y, x2 - 8, color="#374151", sw=3)
# Stage 2 → Stage 3
arrow_h(x2 + SW + 8, MID_Y, x3 - 8, color="#374151", sw=3)

# Cross-stage: Stage 1 v4/best.pt → Stage 3 base weights
# Dashed arc going below the diagram
BELOW = SY + SH + 55
e = {
    "id": _uid(), "type": "arrow",
    "x": x1 + SW // 2, "y": SY + SH,
    "width": x3 + SW // 2 - (x1 + SW // 2),
    "height": BELOW - SY - SH,
    "angle": 0,
    "strokeColor": "#6366f1",
    "backgroundColor": "transparent",
    "fillStyle": "solid", "strokeWidth": 2.5,
    "strokeStyle": "dashed",
    "roughness": 0, "opacity": 100,
    "groupIds": [], "frameId": None,
    "roundness": {"type": 2},
    "seed": _seed(), "version": 1, "versionNonce": _seed(),
    "isDeleted": False, "boundElements": [], "updated": 1700000000000,
    "link": None, "locked": False,
    "points": [
        [0, 0],
        [0, BELOW - SY - SH],
        [x3 + SW // 2 - (x1 + SW // 2), BELOW - SY - SH],
        [x3 + SW // 2 - (x1 + SW // 2), 0],
    ],
    "lastCommittedPoint": None,
    "startBinding": None, "endBinding": None,
    "startArrowhead": None, "endArrowhead": "arrow",
}
elems.append(e)

# Label for cross-stage arrow
LABEL_X = x1 + SW // 2 + (x3 + SW // 2 - x1 - SW // 2) // 2 - 130
txt(LABEL_X, BELOW + 4, 260,
    "v4/best.pt  →  Stage 3 base weights",
    size=12, color="#6366f1", align="center")

# ─────────────────────────────────────────────────────────────────────────────
# Write file
# ─────────────────────────────────────────────────────────────────────────────
doc = {
    "type": "excalidraw",
    "version": 2,
    "source": "https://excalidraw.com",
    "elements": elems,
    "appState": {
        "viewBackgroundColor": "#ffffff",
        "gridSize": None,
        "currentItemFontFamily": 2,
    },
    "files": {},
}

os.makedirs("plots", exist_ok=True)
out = "plots/training_protocol.excalidraw"
with open(out, "w") as f:
    json.dump(doc, f, indent=2)
print(f"Saved: {out}")
print(f"Open at: https://excalidraw.com  (File → Open)")
