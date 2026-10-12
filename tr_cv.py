"""Cross-validated retrain: does accuracy survive when the capture burst does not?

Every validation frame in the shipped soil-moisture-v4-uv split has a near-twin
in training drawn from the same burst -- 40 of 40, at every clustering threshold
tested. A plain k-fold reproduces that structure in each fold and reports a tight
mean near the single-run figure, which reads as evidence of stability and is not.

This runs the paired design built by build_cv_folds.py:

  grouped  one whole capture burst held out; no sibling frame in training
  random   the same number of frames of the same class, drawn at random, so
           siblings remain in training exactly as the shipped split has them

Both arms see identical frame counts and an identical label space, so the gap
between them is the leak, measured.

Reading the result:
  grouped ~= random     the split was not doing the work; accuracy is real
  grouped << random     the reported figures are carried by burst leakage and
                        the paper's numbers describe in-session readout only

The label space is the six levels holding two or more bursts. An mAP here is not
comparable to the eleven-level figures in the main tables and must not be quoted
beside them without saying so.

Results go to runs/cv/, leaving runs/train, runs/threearm and runs/ablation alone.

Usage (from the project root):
  python build_cv_groups.py --dataset soil-moisture-v4-uv
  python build_cv_folds.py  --dataset soil-moisture-v4-uv
  python tr_cv.py
"""

import csv
import os
import statistics
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = "/projects/kp9e/Irrigation-Laser-Yolo2"
if not os.path.isdir(PROJECT_DIR):  # allow local smoke tests
    PROJECT_DIR = str(Path(__file__).resolve().parent)
PYTHON = os.path.join(PROJECT_DIR, "venv", "bin", "python")
if not os.path.isfile(PYTHON):
    PYTHON = sys.executable
os.chdir(PROJECT_DIR)

DATASET = os.environ.get("CV_DATASET", "soil-moisture-v4-uv")
CV_ROOT = os.path.join(PROJECT_DIR, "data", "training-data", "cv", DATASET)
# On a dataset with no burst redundancy the random control is identical to the
# grouped arm by construction, so there is nothing to compare and running it only
# doubles the cost. CV_REGIMES=grouped runs the cross-burst arm alone.
REGIMES = tuple(x.strip() for x in os.environ.get("CV_REGIMES", "grouped,random").split(",") if x.strip())
EPOCHS = int(os.environ.get("CV_EPOCHS", "200"))
# The thinnest eligible class holds two bursts, so two folds is the maximum that
# keeps every class in validation. Variance therefore comes from seeds, not folds.
SEEDS = [int(x) for x in os.environ.get("CV_SEEDS", "0,1,2").split(",") if x.strip() != ""]
PROJECT_OUT = "runs/cv"


def folds():
    if not os.path.isdir(CV_ROOT):
        sys.exit(f"ERROR: {CV_ROOT} not found -- run build_cv_folds.py first")
    return sorted(d for d in os.listdir(CV_ROOT) if d.startswith("fold"))


def train(fold, regime, seed):
    name = f"{DATASET}__{fold}__{regime}__seed{seed}"
    if os.path.isfile(os.path.join(PROJECT_OUT, name, "results.csv")):
        print(f"skip {name}: already trained", flush=True)
        return
    yaml_path = os.path.join(CV_ROOT, fold, regime, "data.yaml")
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
    manifest = {}
    man_path = os.path.join(PROJECT_DIR, "runs", "cv", f"{DATASET}_folds.csv")
    if os.path.isfile(man_path):
        for r in csv.DictReader(open(man_path)):
            manifest[f"fold{int(r['fold']):02d}"] = r

    rows, by_regime = [], {r: [] for r in REGIMES}
    for fold in folds():
        for seed in SEEDS:
            row = {"fold": fold, "seed": seed,
                   "n_val": manifest.get(fold, {}).get("n_val", ""),
                   "n_classes_val": manifest.get(fold, {}).get("n_classes_val", "")}
            for regime in REGIMES:
                m = best_map50(os.path.join(
                    PROJECT_OUT, f"{DATASET}__{fold}__{regime}__seed{seed}", "results.csv"))
                row[regime] = m
                if m is not None:
                    by_regime[regime].append(m)
            if row.get("grouped") is not None and row.get("random") is not None:
                row["delta"] = row["grouped"] - row["random"]
            rows.append(row)

    out = os.path.join(PROJECT_DIR, "runs", "cv", f"{DATASET}_cv_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "seed", "n_val", "n_classes_val",
                                          "grouped", "random", "delta"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})

    print("\n" + "=" * 72)
    print(f"{DATASET}  one burst per class held out, {len(folds())} folds x {len(SEEDS)} seeds")
    print("=" * 72)
    print(f"{'fold':>7} {'seed':>5} {'n_val':>6} {'levels':>7} {'grouped':>9} {'random':>9} {'delta':>9}")
    for r in rows:
        g = f"{r['grouped']:.4f}" if r.get("grouped") is not None else "--"
        n = f"{r['random']:.4f}" if r.get("random") is not None else "--"
        d = f"{r['delta']:+.4f}" if r.get("delta") is not None else "--"
        print(f"{r['fold']:>7} {r['seed']:>5} {str(r['n_val']):>6} "
              f"{str(r['n_classes_val']):>7} {g:>9} {n:>9} {d:>9}")

    print("-" * 72)
    for regime in REGIMES:
        v = by_regime[regime]
        if len(v) >= 2:
            print(f"{regime:>9}: mean={statistics.mean(v):.4f}  sd={statistics.stdev(v):.4f}  n={len(v)}")
        elif v:
            print(f"{regime:>9}: {v[0]:.4f}  n=1")
    g, n = by_regime.get("grouped", []), by_regime.get("random", [])
    if len(g) >= 2 and len(n) >= 2:
        gap = statistics.mean(g) - statistics.mean(n)
        pooled = ((statistics.stdev(g) ** 2 + statistics.stdev(n) ** 2) / 2) ** 0.5
        scale = f"  ({abs(gap) / pooled:.1f}x pooled sd)" if pooled else ""
        print(f"\n  cross-burst minus in-burst: {gap:+.4f}{scale}")
        print("  Re-encoding noise floor is 0.015; treat smaller gaps as nothing.")
    if g and n and all(abs(x - 0.995) < 1e-6 for x in g + n):
        print("\n  WARNING: every run returned exactly 0.9950, YOLOv5's ceiling for")
        print("  perfect detection. That is saturation, not agreement -- check that")
        print("  validation really spans more than one class before reading anything")
        print("  into the gap.")
    print(f"\nwrote {out}")


def main():
    for fold in folds():
        for regime in REGIMES:
            for seed in SEEDS:
                train(fold, regime, seed)
    summarize()


if __name__ == "__main__":
    main()
