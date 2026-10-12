"""Train the laser-usefulness ablation: 4 conditions x N seeds.

Conditions (datasets built by scripts/data_prep/laser_ablation_dataset.py under data/ablation_september):
  full_laser    detection on whole frames, laser present
  full_nolaser  detection on whole frames, laser inpainted away
  patch_laser   classification on 160x160 spot crops, laser present
  patch_nolaser classification on the same crops, laser inpainted away

Everything else is held identical across conditions: model, hyperparameters,
epochs, image size, splits, and seed. Detection mirrors the scripts/training/tr.py protocol
(yolov5s, AdamW, cos-lr); classification uses classify/train.py (yolov5s-cls).

Isolation from the existing pipeline: results go to runs/ablation (detection)
and runs/train-cls/ablation (classification). scripts/evaluation/analyze.py and scripts/figures/plot_results.py
only scan runs/train, so first-run results and their reports are untouched.

After training, a summary table (best val mAP50 / top-1 per run, mean +/- std
per condition) is printed and written to runs/ablation/ablation_summary.csv.
"""

import csv
import os
import statistics
import subprocess
from pathlib import Path

PROJECT_DIR = "/projects/kp9e/Irrigation-Laser-Yolo2"
if not os.path.isdir(PROJECT_DIR):  # allow local smoke tests
    PROJECT_DIR = str(Path(__file__).resolve().parents[2])
PYTHON = os.path.join(PROJECT_DIR, "venv", "bin", "python")
if not os.path.isfile(PYTHON):
    import sys
    PYTHON = sys.executable
os.chdir(PROJECT_DIR)

ABLATION_DATA = os.path.join(PROJECT_DIR, "data", "ablation_september")
DET_PROJECT = "runs/ablation"
CLS_PROJECT = "runs/train-cls/ablation"
SEEDS = [0, 1, 2]
DET_EPOCHS = 200
CLS_EPOCHS = 100


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    print(f"Return code: {result.returncode}", flush=True)
    return result.returncode


def train_detection(cond, seed):
    name = f"{cond}_seed{seed}"
    if os.path.isfile(os.path.join(DET_PROJECT, name, "results.csv")):
        print(f"skip {name}: results.csv exists", flush=True)
        return
    run([
        PYTHON, os.path.join(PROJECT_DIR, "train.py"),
        "--img", "640",
        "--epochs", str(DET_EPOCHS),
        "--batch-size", "32",
        "--data", os.path.join(ABLATION_DATA, cond, "data.yaml"),
        "--weights", "yolov5s.pt",
        "--device", "",
        "--optimizer", "AdamW",
        "--cos-lr",
        "--patience", "100",
        "--seed", str(seed),
        "--project", DET_PROJECT,
        "--name", name,
    ])


def train_classification(cond, seed):
    name = f"{cond}_seed{seed}"
    if os.path.isfile(os.path.join(CLS_PROJECT, name, "results.csv")):
        print(f"skip {name}: results.csv exists", flush=True)
        return
    run([
        PYTHON, os.path.join(PROJECT_DIR, "classify", "train.py"),
        "--model", "yolov5s-cls.pt",
        "--data", os.path.join(ABLATION_DATA, cond),
        "--epochs", str(CLS_EPOCHS),
        "--batch-size", "64",
        "--imgsz", "160",
        "--optimizer", "AdamW",
        "--seed", str(seed),
        "--project", CLS_PROJECT,
        "--name", name,
    ])


def best_metric(results_csv, column_substring):
    """Max value of the first column whose stripped name contains the substring."""
    try:
        with open(results_csv, newline="") as f:
            reader = csv.reader(f)
            header = [c.strip() for c in next(reader)]
            idx = next(i for i, c in enumerate(header) if column_substring in c)
            return max(float(row[idx]) for row in reader if row)
    except (OSError, StopIteration, ValueError):
        return None


def summarize():
    rows = []
    for cond in ("full_laser", "full_nolaser"):
        for seed in SEEDS:
            m = best_metric(os.path.join(DET_PROJECT, f"{cond}_seed{seed}", "results.csv"),
                            "mAP_0.5")
            rows.append([cond, seed, "best_val_mAP50", m])
    for cond in ("patch_laser", "patch_nolaser"):
        for seed in SEEDS:
            m = best_metric(os.path.join(CLS_PROJECT, f"{cond}_seed{seed}", "results.csv"),
                            "top1")
            rows.append([cond, seed, "best_top1_acc", m])

    os.makedirs(DET_PROJECT, exist_ok=True)
    out_csv = os.path.join(DET_PROJECT, "ablation_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "seed", "metric", "value"])
        w.writerows(rows)

    print("\n=== Laser ablation summary ===", flush=True)
    for cond in ("full_laser", "full_nolaser", "patch_laser", "patch_nolaser"):
        vals = [r[3] for r in rows if r[0] == cond and r[3] is not None]
        if not vals:
            print(f"{cond:14s}  no results", flush=True)
            continue
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        metric = "best_val_mAP50" if cond.startswith("full") else "best_top1_acc"
        print(f"{cond:14s}  {metric:>14s}  mean={mean:.4f}  std={std:.4f}"
              f"  n={len(vals)}", flush=True)
    print(f"summary written to {out_csv}", flush=True)


if __name__ == "__main__":
    for seed in SEEDS:
        for cond in ("full_laser", "full_nolaser"):
            train_detection(cond, seed)
        for cond in ("patch_laser", "patch_nolaser"):
            train_classification(cond, seed)
    summarize()
