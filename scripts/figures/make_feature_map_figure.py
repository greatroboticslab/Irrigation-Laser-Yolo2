#!/usr/bin/env python3
"""Visualize backbone feature-map evolution of the fine-tuned detector.

Runs every image of the IR test set through the final (fine-tuned
ir-combined) model, records activations at the P1..P5 backbone stages via
forward hooks, keeps the image with the most confident detection, and renders
a figure showing how features evolve with depth:

  P1/P2 - high-resolution, low-level primitives (edges, contrast, color)
  P3/P4 - object components (spot core, halo, boundary)
  P5    - coarse, semantic, whole-object activations

For each stage the 8 channels with the strongest mean activation inside the
detected LED box are shown; the red outline marks the most spot-selective
channel (highest in-box vs. global activation ratio).

Usage:  python scripts/figures/make_feature_map_figure.py
Output: results/feature_map_evolution.png
"""
import glob
import os
from pathlib import Path
import sys

import numpy as np
import torch

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)

# our own trained checkpoints predate torch>=2.6 weights_only default
_orig_load = torch.load
def _load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)
torch.load = _load

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes

WEIGHTS = os.path.join(REPO, "results/runs/fine-tuned/soil-moisture-ir-combined/weights/best.pt")
IMGDIR = os.path.join(REPO, "data/training-data/downloaded/soil-moisture-ir/test/images")
OUT = os.path.join(REPO, "results/feature_map_evolution.png")
OUT_PDF = os.path.join(REPO, "results/feature_map_evolution.pdf")

# backbone stages to capture: layer index -> (stage, block, description)
LAYERS = [
    (0, "P1/2", "Conv", "edges · contrast ·\ncolor transitions"),
    (2, "P2/4", "C3", "fine textures ·\nlocal primitives"),
    (4, "P3/8", "C3", "spot components:\ncore · halo · rim"),
    (6, "P4/16", "C3", "part assemblies ·\nobject contours"),
    (9, "P5/32", "SPPF", "whole-object semantics ·\nbackground suppressed"),
]
N_SHOW = 8


def extract():
    device = torch.device("cpu")
    model = attempt_load(WEIGHTS, device=device, inplace=True, fuse=True)
    model.eval()
    names = model.names

    acts = {}
    for idx, *_ in LAYERS:
        def mk(i):
            def hook(m, inp, out):
                acts[i] = out.detach()
            return hook
        model.model[idx].register_forward_hook(mk(idx))

    best = None  # (conf, path, im0, dets, acts)
    for path in sorted(glob.glob(os.path.join(IMGDIR, "*.jpg"))):
        im0 = cv2.imread(path)
        im = letterbox(im0, 640, stride=32, auto=False)[0]
        im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])
        t = torch.from_numpy(im).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            pred = model(t)[0]
        det = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]
        conf = float(det[:, 4].max()) if len(det) else 0.0
        print(f"{os.path.basename(path)[:44]:46s} dets={len(det)} maxconf={conf:.3f}")
        if best is None or conf > best[0]:
            det2 = det.clone()
            if len(det2):
                det2[:, :4] = scale_boxes(t.shape[2:], det2[:, :4], im0.shape).round()
            best = (conf, path, im0.copy(), det2.cpu().numpy(),
                    {k: v.cpu().numpy() for k, v in acts.items()})
    return names, best


def render(names, best):
    conf, path, im0_bgr, dets, acts = best
    im0 = cv2.cvtColor(im0_bgr, cv2.COLOR_BGR2RGB)
    H0, W0 = im0.shape[:2]
    x1, y1, x2, y2, dconf, cls = dets[0]
    label = f"{names[int(cls)]}  {dconf:.2f}"
    bx = (x1 / W0, y1 / H0, x2 / W0, y2 / H0)

    fig = plt.figure(figsize=(13.2, 10.2), facecolor="white")
    gs = fig.add_gridspec(
        6, N_SHOW + 1,
        width_ratios=[1.55] + [1.0] * N_SHOW,
        height_ratios=[1.55, 1, 1, 1, 1, 1],
        left=0.015, right=0.985, top=0.94, bottom=0.045,
        wspace=0.06, hspace=0.18,
    )
    fig.suptitle(
        "Backbone feature-map evolution of the fine-tuned YOLOv5 detector on an IR-Pot-Background test image",
        fontsize=15, y=0.985,
    )

    ax_in = fig.add_subplot(gs[0, 0:2])
    ax_in.imshow(im0)
    ax_in.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                              fill=False, edgecolor="#00FF40", linewidth=2.2))
    ax_in.text(x1, max(y1 - 12, 20), label, color="#00FF40", fontsize=9,
               fontweight="bold", va="bottom",
               bbox=dict(facecolor="black", alpha=0.55, pad=1.5, edgecolor="none"))
    ax_in.set_title(f"Input  {W0}×{H0}", fontsize=11)
    ax_in.axis("off")

    ax_txt = fig.add_subplot(gs[0, 2:])
    ax_txt.axis("off")
    ax_txt.text(
        0.02, 0.5,
        "Each row: the 8 channels responding most strongly inside the detected LED region.\n"
        "Early layers (P1, P2) keep full spatial detail; generic edges, contrast and color cues fire everywhere.\n"
        "Middle layers (P3, P4) trade resolution for depth; activations organize around the bright spot\n"
        "core, halo and boundary. Deep layers (P5, SPPF) are coarse but semantic; nearly all remaining\n"
        "activation sits on the object region and background clutter is discarded.\n"
        "Red outline: most spot-selective channel (highest in-box vs. global activation ratio).",
        fontsize=10.5, va="center", ha="left", linespacing=1.7, color="#333333",
    )

    for row, (idx, stage, block, desc) in enumerate(LAYERS, start=1):
        fm = acts[idx][0]  # (C, h, w)
        C, h, w = fm.shape

        mask = np.zeros((h, w), bool)
        mx1, my1 = int(bx[0] * w), int(bx[1] * h)
        mx2, my2 = max(int(bx[2] * w), mx1 + 1), max(int(bx[3] * h), my1 + 1)
        mask[my1:my2, mx1:mx2] = True

        pos = np.maximum(fm, 0)
        inbox = pos[:, mask].mean(axis=1)
        globl = pos.reshape(C, -1).mean(axis=1)
        order = np.argsort(inbox)[::-1][:N_SHOW]
        ratio = inbox / (globl + 1e-6)
        star = order[np.argmax(ratio[order])]

        axl = fig.add_subplot(gs[row, 0])
        axl.axis("off")
        axl.text(0.04, 0.72, stage, fontsize=13, fontweight="bold", color="#1F4E79")
        axl.text(0.04, 0.52, f"{block} · {w}×{h}×{C}", fontsize=9.5, color="#555555")
        axl.text(0.04, 0.05, desc, fontsize=9, color="#777777", linespacing=1.35)

        for j, c in enumerate(order):
            ax = fig.add_subplot(gs[row, j + 1])
            m = fm[c]
            vmax = m.max() if m.max() > 0 else 1.0
            ax.imshow(m, cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            edge = "#E53935" if c == star else "#BBBBBB"
            lw = 2.4 if c == star else 0.6
            for s in ax.spines.values():
                s.set_edgecolor(edge); s.set_linewidth(lw)
            ax.set_title(f"ch {c}", fontsize=7, color="#888888", pad=1.5)

    fig.savefig(OUT, dpi=300)
    fig.savefig(OUT_PDF, dpi=300)
    print("saved", OUT)
    print("saved", OUT_PDF)


if __name__ == "__main__":
    names, best = extract()
    print(f"\nBEST: {os.path.basename(best[1])}  conf {best[0]:.3f}")
    render(names, best)
