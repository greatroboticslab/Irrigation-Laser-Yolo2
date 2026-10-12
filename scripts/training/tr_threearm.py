"""Three-arm retrain: is the laser useful, or merely the feature models latch onto?

The inference-time ablation showed that models trained with the laser collapse
when it is removed. That proves dependence, not usefulness: a salient feature
gets learned in preference to a subtler one, so the collapse is equally
consistent with "soil texture could have carried this information, but the
model never had to learn it". Only retraining from scratch separates the two.

Arms, trained independently on identical splits:
  laser    original frames
  nolaser  laser inpainted out of train, valid and test
  sham     inpainting scar of the same size placed elsewhere, laser intact

The sham arm is not optional. Without it, the nolaser arm is the only one
trained on edited imagery, and any gap can be blamed on the edit rather than
on the missing laser.

Reading the result:
  nolaser ~= laser            -> the laser is dispensable; soil appearance suffices
  nolaser << laser ~= sham    -> the laser carries information the soil does not
  nolaser ~= sham << laser    -> the inpainting itself is the problem, not the laser

Datasets are limited to those where laser localisation was verified accurate
(>=92% of frames, measured against an independent colour-blob detector). The
pot-in-frame captures -- soil-moisture-ir, -ir-combined, 5sagf -- localise at
only 60-85% because the bright metal pot rim outscores the laser, so their
"laser removed" images are partly "rim removed" and they are excluded here.

Results go to runs/threearm/, leaving runs/train and runs/ablation untouched.

Usage (from the project root):
  python scripts/training/tr_threearm.py
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

DATA_ROOT = os.path.join(PROJECT_DIR, "data", "training-data")
ARMS = {
    "laser": "downloaded",
    "nolaser": "nolaser",
    "sham": "sham",
}
DATASETS = [
    "soil-moisture-v4-uv",
    "soil-moisture-v4",
    "soil-moisture-v4-september-combined",
]
SEEDS = [0, 1, 2]
EPOCHS = 200
PROJECT_OUT = "runs/threearm"


def data_yaml(arm, dataset):
    # The untouched arm gets a generated yaml: the datasets' own files carry a
    # broken `test: ../test/images` path and a block-style class list, and the
    # three arms must be specified identically so that only pixels differ.
    if arm == "laser":
        return os.path.join(DATA_ROOT, "arm_yamls", f"{dataset}__laser.yaml")
    return os.path.join(DATA_ROOT, ARMS[arm], dataset, "data.yaml")


def train(arm, dataset, seed):
    name = f"{dataset}__{arm}__seed{seed}"
    if os.path.isfile(os.path.join(PROJECT_OUT, name, "results.csv")):
        print(f"skip {name}: already trained", flush=True)
        return
    yaml_path = data_yaml(arm, dataset)
    if not os.path.isfile(yaml_path):
        print(f"MISSING {yaml_path} -- skipping {name}", flush=True)
        return
    cmd = [
        PYTHON, os.path.join(PROJECT_DIR, "train.py"),
        "--img", "640",
        "--epochs", str(EPOCHS),
        "--batch-size", "32",
        "--data", yaml_path,
        "--weights", "yolov5s.pt",
        "--device", "",
        "--optimizer", "AdamW",
        "--cos-lr",
        "--patience", "100",
        "--seed", str(seed),
        "--project", PROJECT_OUT,
        "--name", name,
    ]
    print("+ " + " ".join(cmd), flush=True)
    print(f"Return code: {subprocess.run(cmd).returncode}", flush=True)


def best_map50(results_csv):
    try:
        with open(results_csv, newline="") as f:
            reader = csv.reader(f)
            header = [c.strip() for c in next(reader)]
            idx = next(i for i, c in enumerate(header) if "mAP_0.5" in c and "0.95" not in c)
            return max(float(row[idx]) for row in reader if row)
    except (OSError, StopIteration, ValueError):
        return None


def summarize():
    rows = []
    for dataset in DATASETS:
        for arm in ARMS:
            vals = []
            for seed in SEEDS:
                m = best_map50(os.path.join(
                    PROJECT_OUT, f"{dataset}__{arm}__seed{seed}", "results.csv"))
                if m is not None:
                    vals.append(m)
                rows.append([dataset, arm, seed, round(m, 4) if m is not None else ""])
            if vals:
                mean = statistics.mean(vals)
                std = statistics.stdev(vals) if len(vals) > 1 else 0.0
                print(f"{dataset:<38} {arm:<8} mean={mean:.4f} std={std:.4f} n={len(vals)}",
                      flush=True)

    os.makedirs(PROJECT_OUT, exist_ok=True)
    out = os.path.join(PROJECT_OUT, "threearm_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "arm", "seed", "best_val_mAP50"])
        w.writerows(rows)

    print("\n=== laser vs nolaser (the question) ===", flush=True)
    for dataset in DATASETS:
        got = {}
        for arm in ARMS:
            vals = [best_map50(os.path.join(
                PROJECT_OUT, f"{dataset}__{arm}__seed{s}", "results.csv")) for s in SEEDS]
            vals = [v for v in vals if v is not None]
            got[arm] = statistics.mean(vals) if vals else None
        if got["laser"] is None or got["nolaser"] is None:
            continue
        delta = got["nolaser"] - got["laser"]
        sham = f"{got['sham']:.4f}" if got["sham"] is not None else "n/a"
        print(f"{dataset:<38} laser={got['laser']:.4f} nolaser={got['nolaser']:.4f} "
              f"sham={sham} delta={delta:+.4f}", flush=True)
    print(f"\nsummary written to {out}", flush=True)


if __name__ == "__main__":
    for seed in SEEDS:
        for dataset in DATASETS:
            for arm in ARMS:
                train(arm, dataset, seed)
    summarize()
