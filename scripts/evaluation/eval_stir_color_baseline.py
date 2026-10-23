"""Hand-crafted colour baseline for the stir-probe reading task, on the grouped CV folds.

The paper's claim is that the class signal is the LED colour. If so, a colour classifier on
the LED crop is the obvious baseline for the detector. This one is given the annotated box
(oracle localization), so it is an upper bound for colour features that must also find the
LED; the detector's top-1 accuracy additionally requires IoU >= 0.5 with that box.

Features per frame, from the annotated box: the brightest 10% of its pixels (the glow) give
circular hue (cos, sin), mean saturation and value, and mean R, G, B; the share of box
pixels above half the glow's value gives spot extent; box width and height give spot size.
Classifiers: k-nearest neighbours (k = 5) and nearest centroid, on standardized features.

Folds are data/training-data/cv/stir/foldNN (scripts/data_prep/build_stir_cv.py): bursts
are dealt into five level-stratified folds, so no held-out frame has a burst sibling in
training. Each fold trains on its training frames, augmented copies included, as the
detector does, and is scored on its held-out originals. Every original is held out once.

Writes results/stir_color_baseline.csv (one row per classifier) and
results/stir_color_baseline_oof.csv (per original).

Usage (from the project root, with opencv and numpy, e.g. venv/bin/python):
  python scripts/evaluation/eval_stir_color_baseline.py
"""

import csv
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CV = ROOT / "data" / "training-data" / "cv" / "stir"
NAMES = ["1", "10", "3", "7", "8"]
LEVEL = np.array([int(n) for n in NAMES])
RANK = {lv: i for i, lv in enumerate(sorted(LEVEL))}


def is_original(name):
    return "_aug" not in name


def features(img_path, label_path):
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]
    cls, x, y, w, h = open(label_path).read().split()[:5]
    x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
    x0, x1 = max(0, int(x - w / 2)), min(W, int(np.ceil(x + w / 2)))
    y0, y1 = max(0, int(y - h / 2)), min(H, int(np.ceil(y + h / 2)))
    crop = img[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV_FULL).reshape(-1, 3).astype(float)
    bgr = crop.reshape(-1, 3).astype(float)
    v = hsv[:, 2]
    glow = v >= np.percentile(v, 90)
    hue = hsv[glow, 0] * 2 * np.pi / 256
    f = [np.cos(hue).mean(), np.sin(hue).mean(), hsv[glow, 1].mean() / 255, v[glow].mean() / 255,
         *(bgr[glow].mean(0)[::-1] / 255), (v > 0.5 * v[glow].mean()).mean(), w / W, h / H]
    return np.array(f), int(float(cls))


def load(split_dir, originals_only):
    X, y, names = [], [], []
    for f in sorted(os.listdir(split_dir / "images")):
        if originals_only and not is_original(f):
            continue
        fx, c = features(split_dir / "images" / f, split_dir / "labels" / (f.rsplit(".", 1)[0] + ".txt"))
        X.append(fx), y.append(c), names.append(f.rsplit(".", 1)[0])
    return np.array(X), np.array(y), names


def knn(Xtr, ytr, Xte, k=5):
    d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    idx = np.argsort(d, axis=1)[:, :k]
    return np.array([np.bincount(ytr[i], minlength=len(NAMES)).argmax() for i in idx])


def centroid(Xtr, ytr, Xte):
    C = np.stack([Xtr[ytr == c].mean(0) for c in range(len(NAMES))])
    return ((Xte[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)


def main():
    oof = []
    for fold in sorted(d for d in os.listdir(CV) if d.startswith("fold")):
        Xtr, ytr, _ = load(CV / fold / "train", originals_only=False)
        Xte, yte, names = load(CV / fold / "valid", originals_only=True)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        for clf, pred in (("knn5", knn(Xtr, ytr, Xte)), ("centroid", centroid(Xtr, ytr, Xte))):
            for n, g, p in zip(names, yte, pred):
                oof.append({"fold": fold, "classifier": clf, "original": n,
                            "level": LEVEL[g], "pred_level": LEVEL[p]})
    rows = []
    for clf in ("knn5", "centroid"):
        rs = [r for r in oof if r["classifier"] == clf]
        e = np.array([abs(int(r["pred_level"]) - int(r["level"])) for r in rs])
        re = np.array([abs(RANK[r["pred_level"]] - RANK[r["level"]]) for r in rs])
        per = {lv: np.mean([r["pred_level"] == lv for r in rs if r["level"] == lv]) for lv in sorted(LEVEL)}
        row = {"classifier": clf, "n_originals": len(rs), "acc": round(float((e == 0).mean()), 4),
               "mae": round(float(e.mean()), 4), "rank_mae": round(float(re.mean()), 4),
               "far": round(float((re >= 3).mean()), 4),
               **{f"recall_L{lv}": round(float(v), 3) for lv, v in per.items()}}
        rows.append(row)
        print(row)
    with open(ROOT / "results" / "stir_color_baseline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(), w.writerows(rows)
    with open(ROOT / "results" / "stir_color_baseline_oof.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(oof[0]))
        w.writeheader(), w.writerows(oof)


if __name__ == "__main__":
    main()
