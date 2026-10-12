"""Score every saved checkpoint of scripts/training/tr_stir_cv.py on its held-out fold.

Each run trained on four folds; its held-out fold's frames were never seen in
training, and neither was any burst sibling of them. Scoring every saved
checkpoint there and pooling across the five folds gives an out-of-fold (OOF)
prediction for each of the 56 originals at each epoch -- a learning curve
measured on independent scenes, from which the stopping epoch can be chosen
without touching any fold twice.

Metric: top-1 detection per image, correct when class matches and IoU >= 0.5
(the same protocol as scripts/evaluation/eval_calibration.py). Accuracy is reported on the 56
originals (independent scenes, the honest number) and on all held-out frames
(augmented copies included, comparable to the shipped validation protocol).
Validation mAP@0.5 per epoch is read from each run's results.csv alongside.

Outputs (runs/stir_cv/):
  stir_cv_oof.csv       one row per (run, checkpoint, image)
  stir_cv_summary.csv   per (arm, epoch): OOF accuracy mean/sd over seeds,
                        frame accuracy, mean fold mAP@0.5

Usage (from the project root, after scripts/training/tr_stir_cv.py):
  python scripts/evaluation/eval_stir_cv.py
"""

import csv
import glob
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = "/projects/kp9e/Irrigation-Laser-Yolo2"
if not os.path.isdir(PROJECT_DIR):
    PROJECT_DIR = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

import pandas as pd
import torch

from eval_calibration import predict  # loads models with weights_only=False
from models.experimental import attempt_load

RUNS = os.path.join(PROJECT_DIR, "runs", "stir_cv")
CV_ROOT = os.path.join(PROJECT_DIR, "data", "training-data", "cv", "stir")
COPY_RE = re.compile(r"_aug(x)?\d+$")
NAME_RE = re.compile(r"stir__(fold\d+)__(\w+?)__seed(\d+)$")


def checkpoints(run):
    w = os.path.join(run, "weights")
    out = {}
    last = os.path.join(w, "last.pt")
    if os.path.isfile(last):
        out[len(pd.read_csv(os.path.join(run, "results.csv"))) - 1] = last
    for p in glob.glob(os.path.join(w, "epoch*.pt")):  # a periodic save wins a tie with last.pt
        out[int(re.search(r"epoch(\d+)", p).group(1))] = p
    return sorted(out.items())


def fold_map50(run):
    df = pd.read_csv(os.path.join(run, "results.csv"))
    df.columns = df.columns.str.strip()
    return dict(zip(df["epoch"].astype(int), df["metrics/mAP_0.5"]))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    rows, maps = [], []
    runs = sorted(d for d in glob.glob(os.path.join(RUNS, "stir__*")) if os.path.isdir(d))
    if not runs:
        sys.exit(f"ERROR: no runs under {RUNS}")
    for run in runs:
        m = NAME_RE.search(os.path.basename(run))
        if not m:
            continue
        fold, arm, seed = m.group(1), m.group(2), int(m.group(3))
        img_dir = os.path.join(CV_ROOT, fold, "valid", "images")
        fm = fold_map50(run)
        for epoch, ck in checkpoints(run):
            model = attempt_load(ck, device=device, inplace=True, fuse=True).eval()
            for r in predict(model, device, img_dir):
                stem = os.path.splitext(r["image"])[0]
                rows.append(dict(fold=fold, arm=arm, seed=seed, epoch=epoch, **r,
                                 original=COPY_RE.sub("", stem),
                                 is_original=int(not COPY_RE.search(stem))))
            maps.append(dict(fold=fold, arm=arm, seed=seed, epoch=epoch, map50=fm.get(epoch)))
            print(f"{os.path.basename(run)} epoch {epoch}", flush=True)

    oof = pd.DataFrame(rows)
    oof.to_csv(os.path.join(RUNS, "stir_cv_oof.csv"), index=False)
    mp = pd.DataFrame(maps)

    summ = []
    for (arm, epoch), g in oof.groupby(["arm", "epoch"]):
        per_seed = g[g.is_original == 1].groupby("seed").correct.mean()
        per_seed_frames = g.groupby("seed").correct.mean()
        n_orig = g[g.is_original == 1].groupby("seed").size()
        fm = mp[(mp.arm == arm) & (mp.epoch == epoch)].map50.dropna()
        summ.append(dict(
            arm=arm, epoch=epoch, n_seeds=len(per_seed), n_originals=int(n_orig.min()),
            oof_acc_originals=per_seed.mean(),
            oof_acc_originals_sd=per_seed.std(ddof=1) if len(per_seed) > 1 else float("nan"),
            oof_acc_frames=per_seed_frames.mean(),
            fold_map50_mean=fm.mean() if len(fm) else float("nan"),
        ))
    s = pd.DataFrame(summ).sort_values(["arm", "epoch"])
    s.to_csv(os.path.join(RUNS, "stir_cv_summary.csv"), index=False, float_format="%.4f")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(s.round(3).to_string(index=False))

    print("\nCV-selected stopping epoch (max mean OOF accuracy on originals):")
    for arm, g in s.groupby("arm"):
        b = g.loc[g.oof_acc_originals.idxmax()]
        last = g.loc[g.epoch.idxmax()]
        print(f"  {arm:>7}: epoch {int(b.epoch):>3}  acc {b.oof_acc_originals:.3f} "
              f"(sd {b.oof_acc_originals_sd:.3f} over seeds)   final epoch {int(last.epoch)}: "
              f"{last.oof_acc_originals:.3f}")
    print("  Selecting the epoch on these same OOF scores is mildly optimistic; quote the")
    print("  curve, or the final-epoch figure, beside it.")


if __name__ == "__main__":
    main()
