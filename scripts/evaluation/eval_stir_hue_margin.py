"""How far apart the stir-probe levels are in hue, and how far the augmentation moves them.

The augmentation caps the explicit hue shift at +-16 deg (augment_stir_september.py), a
value chosen rather than measured. This measures, on the stir-probe build:

  per level   circular mean and s.d. of the glow hue over its originals, and the hue gap
              from each level's mean to the nearest other level's mean
  margin      per original, the smallest hue shift (either direction) that brings its glow
              hue closer to another level's mean than to its own; the share of originals
              whose margin is at most 16 deg
  copies      per augmented copy, the change in glow hue from its original, and whether
              the copy's glow hue lies closer to another level's mean than to its own while
              its original's did not (a crossing)

Glow hue is the circular mean hue of the brightest 10% of pixels in the annotated box, as
in eval_stir_color_baseline.py. Frames come from the union of the CV folds' validation
splits, which holds every original and copy exactly once.

Writes results/stir_hue_levels.csv (per level) and results/stir_hue_margin.csv (per original).

Usage (from the project root): venv/bin/python scripts/evaluation/eval_stir_hue_margin.py
"""

import csv
import os
import re
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CV = ROOT / "data" / "training-data" / "cv" / "stir"
NAMES = ["1", "10", "3", "7", "8"]
CAP = 16.0
COPY_RE = re.compile(r"_aug(x)?\d+$")


def glow_hue(img_path, label_path):
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]
    cls, x, y, w, h = open(label_path).read().split()[:5]
    x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
    x0, x1 = max(0, int(x - w / 2)), min(W, int(np.ceil(x + w / 2)))
    y0, y1 = max(0, int(y - h / 2)), min(H, int(np.ceil(y + h / 2)))
    hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV_FULL).reshape(-1, 3).astype(float)
    glow = hsv[:, 2] >= np.percentile(hsv[:, 2], 90)
    a = hsv[glow, 0] * 2 * np.pi / 256
    return np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360, int(NAMES[int(float(cls))])


def cdist(a, b):
    return np.abs((np.asarray(a) - b + 180) % 360 - 180)


def cmean(h):
    a = np.radians(h)
    return np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360


def csd(h):
    a = np.radians(h)
    r = np.hypot(np.sin(a).mean(), np.cos(a).mean())
    return np.degrees(np.sqrt(-2 * np.log(max(r, 1e-12))))


def margin(h, own, means):
    """Smallest |shift| that makes h + shift closer to another level's mean than to its own."""
    for s in np.arange(0, 180.5, 0.5):
        for hs in (h + s, h - s):
            d_own = cdist(hs, means[own])
            if any(cdist(hs, m) < d_own for lv, m in means.items() if lv != own):
                return s
    return 180.0


def nearest(h, means):
    return min(means, key=lambda lv: cdist(h, means[lv]))


def main():
    frames = {}
    for fold in sorted(d for d in os.listdir(CV) if d.startswith("fold")):
        d = CV / fold / "valid"
        for f in sorted(os.listdir(d / "images")):
            stem = f.rsplit(".", 1)[0]
            frames[stem] = glow_hue(d / "images" / f, d / "labels" / (stem + ".txt"))
    orig = {s: v for s, v in frames.items() if not COPY_RE.search(s)}
    levels = sorted({lv for _, lv in orig.values()})
    means = {lv: cmean([h for h, l in orig.values() if l == lv]) for lv in levels}

    rows = []
    for lv in levels:
        hs = [h for h, l in orig.values() if l == lv]
        gap = min(cdist(means[lv], means[o]) for o in levels if o != lv)
        rows.append({"level": lv, "n_originals": len(hs), "hue_mean": round(means[lv], 1),
                     "hue_sd": round(csd(hs), 1), "gap_to_nearest_mean": round(float(gap), 1),
                     "nearest_level": min((o for o in levels if o != lv),
                                          key=lambda o: cdist(means[lv], means[o]))})
        print(rows[-1])

    per = []
    for s, (h, lv) in sorted(orig.items()):
        per.append({"original": s, "level": lv, "glow_hue": round(h, 1),
                    "nearest_mean_level": nearest(h, means), "margin": margin(h, lv, means)})
    m = np.array([p["margin"] for p in per])
    wrong = sum(p["nearest_mean_level"] != p["level"] for p in per)
    print(f"originals: {len(per)}; nearer another level's mean already: {wrong}; "
          f"margin <= {CAP:.0f} deg: {(m <= CAP).sum()} ({(m <= CAP).mean():.2f}); "
          f"median margin {np.median(m):.1f} deg")

    shifts, cross, n_ok = [], 0, 0
    for s, (h, lv) in frames.items():
        if not COPY_RE.search(s):
            continue
        base = COPY_RE.sub("", s)
        h0, _ = orig[base]
        shifts.append(float(cdist(h, h0)))
        if nearest(h0, means) == lv:
            n_ok += 1
            cross += nearest(h, means) != lv
    shifts = np.array(shifts)
    print(f"copies: {len(shifts)}; |glow hue change| median {np.median(shifts):.1f}, "
          f"p90 {np.percentile(shifts, 90):.1f}, > {CAP:.0f} deg: {(shifts > CAP).mean():.3f}; "
          f"crossings: {cross}/{n_ok} ({cross / n_ok:.3f}) of copies whose original lies nearest its own mean")

    for name, data in (("stir_hue_levels.csv", rows), ("stir_hue_margin.csv", per)):
        with open(ROOT / "results" / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader(), w.writerows(data)


if __name__ == "__main__":
    main()
