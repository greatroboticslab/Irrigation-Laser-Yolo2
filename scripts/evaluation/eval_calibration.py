#!/usr/bin/env python3
"""Post-training calibration test for overfitting (no retraining).

Idea: an overfit model is overconfident on its mistakes. For every image we
keep the single top-confidence detection (each image has exactly one labelled
laser spot) and score it correct if class matches GT AND IoU >= 0.5. A missed
spot (no detection) counts as an error with confidence 0.

Per (dataset, checkpoint, split) we report accuracy, mean confidence on
correct vs. wrong predictions, fraction of errors made at conf > 0.9, ECE
(10 equal-width bins) with a bootstrap 95% CI, and the train-minus-test
confidence gap (a memorising model is far surer on train than on held-out).

Validator: stir_september is the one run known to overfit (val loss rises
after epoch ~55), and its last.pt is the post-collapse checkpoint. If this
test cannot single out stir/last.pt, it is not diagnostic here.

Usage:   python scripts/evaluation/eval_calibration.py
Outputs: results/posthoc_calibration.csv        (summary)
         results/posthoc_calibration_preds.csv  (per image)
         results/posthoc_calibration.png
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
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes, xywhn2xyxy
from utils.metrics import box_iou

RUNS = os.path.join(REPO, "results/runs/train/downloaded")
DATA = os.path.join(REPO, "data/training-data/downloaded")
OUT = os.path.join(REPO, "results")
DATASETS = [
    "soil-moisture-v4", "soil-moisture-v4-uv", "soil-moisture-v4-ir",
    "soil-moisture-ir", "soil-moisture-5sagf", "soil_moisture_september",
    "soil_moisture_stir_september",
]
# validation is burst-independent of train only for these (see burst-leakage memo)
INDEPENDENT_VAL = {"soil-moisture-ir", "soil-moisture-5sagf", "soil_moisture_stir_september"}
SPLITS = ["train", "valid", "test"]
CKPTS = ["best", "last"]
IMG = 640
IOU_OK = 0.5
N_BINS = 10
N_BOOT = 1000


def predict(model, device, img_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))):
        lbl = os.path.join(os.path.dirname(img_dir), "labels", os.path.splitext(os.path.basename(path))[0] + ".txt")
        gt = np.loadtxt(lbl, ndmin=2) if os.path.exists(lbl) and os.path.getsize(lbl) else np.zeros((0, 5))
        if len(gt) != 1:
            continue  # protocol assumes one labelled spot per image
        im0 = cv2.imread(path)
        h, w = im0.shape[:2]
        im = letterbox(im0, IMG, stride=32, auto=False)[0]
        im = torch.from_numpy(np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1))).to(device).float()[None] / 255
        with torch.no_grad():
            pred = model(im)[0]
        det = non_max_suppression(pred, conf_thres=0.001, iou_thres=0.6, agnostic=True, max_det=1)[0]
        gt_cls = int(gt[0, 0])
        gt_box = torch.tensor(xywhn2xyxy(gt[:, 1:], w, h), dtype=torch.float32)
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            det = det.cpu()
            conf, cls = float(det[0, 4]), int(det[0, 5])
            iou = float(box_iou(det[:1, :4], gt_box)[0, 0])
        else:
            conf, cls, iou = 0.0, -1, 0.0
        rows.append(dict(image=os.path.basename(path), gt_cls=gt_cls, pred_cls=cls, conf=conf, iou=iou,
                         correct=int(cls == gt_cls and iou >= IOU_OK)))
    return rows


def ece(conf, correct):
    bins = np.minimum((conf * N_BINS).astype(int), N_BINS - 1)
    return sum(abs(correct[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean()
               for b in range(N_BINS) if (bins == b).any())


def summarise(df):
    c, y = df.conf.values, df.correct.values
    rng = np.random.default_rng(0)
    boot = [ece(c[i], y[i]) for i in (rng.integers(0, len(c), len(c)) for _ in range(N_BOOT))]
    err = df[df.correct == 0]
    return dict(
        n=len(df), acc=y.mean(), n_err=len(err), miss=(df.pred_cls == -1).mean(),
        conf_correct=df[df.correct == 1].conf.mean(), conf_wrong=err.conf.mean(),
        err_conf_gt90=(err.conf > 0.9).mean() if len(err) else np.nan,
        mean_conf=c.mean(), ece=ece(c, y), ece_lo=np.percentile(boot, 2.5), ece_hi=np.percentile(boot, 97.5),
    )


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    preds = []
    for ds in DATASETS:
        for ck in CKPTS:
            model = attempt_load(os.path.join(RUNS, ds, "weights", f"{ck}.pt"), device=device, inplace=True, fuse=True).eval()
            for sp in SPLITS:
                rows = predict(model, device, os.path.join(DATA, ds, sp, "images"))
                preds += [dict(dataset=ds, ckpt=ck, split=sp, **r) for r in rows]
                print(f"{ds:32s} {ck:4s} {sp:5s} n={len(rows)}", flush=True)
    pdf = pd.DataFrame(preds)
    pdf.to_csv(os.path.join(OUT, "posthoc_calibration_preds.csv"), index=False)

    summ = pd.DataFrame([dict(dataset=ds, ckpt=ck, split=sp, **summarise(g))
                         for (ds, ck, sp), g in pdf.groupby(["dataset", "ckpt", "split"], sort=False)])
    # train-minus-heldout confidence gap: memorisation shows as much higher train confidence
    tr = summ[summ.split == "train"].set_index(["dataset", "ckpt"]).mean_conf
    summ["conf_gap_vs_train"] = [tr[(r.dataset, r.ckpt)] - r.mean_conf for r in summ.itertuples()]
    summ["independent_val"] = summ.dataset.isin(INDEPENDENT_VAL)
    summ.to_csv(os.path.join(OUT, "posthoc_calibration.csv"), index=False, float_format="%.4f")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summ.round(3).to_string(index=False))
    plot(pdf)


def plot(pdf):
    held = pdf[pdf.split == "test"]
    fig, axes = plt.subplots(2, len(DATASETS), figsize=(3 * len(DATASETS), 5.6), sharey="row")
    edges = np.linspace(0, 1, N_BINS + 1)
    for j, ds in enumerate(DATASETS):
        for i, ck in enumerate(CKPTS):
            ax = axes[i, j]
            g = held[(held.dataset == ds) & (held.ckpt == ck)]
            ax.hist([g[g.correct == 1].conf, g[g.correct == 0].conf], bins=edges, stacked=True,
                    color=["#4c9a6a", "#c8553d"], label=["correct", "wrong"])
            ax.axvline(0.9, ls=":", c="k", lw=0.8)
            ax.set_title(f"{ds.replace('soil-moisture-', '').replace('soil_moisture_', '')} · {ck}.pt", fontsize=9)
            if j == 0:
                ax.set_ylabel("test images")
            if i == 1:
                ax.set_xlabel("top-detection confidence")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Confidence of correct vs. wrong top detections (test split)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "posthoc_calibration.png"), dpi=200)


if __name__ == "__main__":
    main()
