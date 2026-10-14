"""Regularisation sweep for the Stage 3 Mixed Color and Background run.

The paper's Stage 3 stir run overfits: validation classification loss is lowest
at epoch 4 and is 2.5x higher by the end, while training loss falls 14x. The
training split is 39 photographs at 22 frames each, so the network can memorise
the scenes. The recipe has almost no explicit regularisation: AdamW with
weight_decay 0.0005 (decoupled decay, so effectively none), no label
smoothing, and a 200-epoch cosine schedule.

Each arm keeps the paper's recipe (IR-Pot-Background init, AdamW, cos-lr,
data/hyps/hyp.stir-finetune.yaml, batch 32, 640 px, shipped 858/204/95 split) and
changes only what its name says:

  ls         label smoothing 0.1
  wd         AdamW weight_decay 0.05 (100x the paper's)
  ls_wd      both
  ls_wd_e50  both, 50-epoch schedule (cosine reaches its floor at epoch 50)
  ls_wd_lr   both, lr0 0.0003 instead of 0.001
  ls_wd_lr_e60, ls_wd_lr_e100   ls_wd_lr on a 60- or 100-epoch schedule
  ls2_wd_lr_e100                the same at 100 epochs with label smoothing 0.2

Stage 1 arms (names starting s1_) retrain the Stage 1 Mixed Color run the same
way: COCO init (yolov5s.pt) and data/hyps/hyp.stir-soil.yaml, which is that run's
hyp.yaml, with AdamW at lr0 0.01 -- about ten times the usual AdamW rate:

  s1_ls_wd            label smoothing 0.1 + weight_decay 0.05, 200 epochs
  s1_ls_wd_lr         both + lr0 0.001, 200 epochs
  s1_ls_wd_lr_e100    the same on a 100-epoch schedule
  s1_ls_wd_lr_e60     the same on a 60-epoch schedule (the Stage 3 winner's length)
  s1_ls_wd_lr3_e100   both + lr0 0.003, 100 epochs

Patience is off and every run trains its full budget, so the loss curves show
whether the turn-up is gone rather than being cut off by early stopping.
Results go to runs/stir_reg/; nothing else under runs/ is touched.
YOLOv5 computes validation loss with the same smoothed targets it trains on,
so a smoothed run's logged val/cls_loss is damped and not comparable with the
paper's. Checkpoints are therefore saved every 5 epochs and
scripts/evaluation/eval_stir_regularize.py rescores each one with unsmoothed
loss.

Usage (from the project root):
  python scripts/training/tr_stir_regularize.py
Environment overrides: STIR_REG_ARMS=ls,wd  STIR_REG_SEEDS=0,1,2
"""

import glob
import os
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_DIR = "/projects/kp9e/Irrigation-Laser-Yolo2"
if not os.path.isdir(PROJECT_DIR):  # allow local smoke tests
    PROJECT_DIR = str(Path(__file__).resolve().parents[2])
PYTHON = os.path.join(PROJECT_DIR, "venv", "bin", "python")
if not os.path.isfile(PYTHON):
    PYTHON = sys.executable
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

from utils.general import strip_optimizer  # noqa: E402

DATA = os.path.join(PROJECT_DIR, "data", "training-data", "downloaded",
                    "soil_moisture_stir_september", "data.yaml")
INIT = os.path.join(PROJECT_DIR, "runs", "train", "downloaded", "soil-moisture-ir", "weights", "best.pt")
BASE_HYP = os.path.join(PROJECT_DIR, "data", "hyps", "hyp.stir-finetune.yaml")
# Stage 1: the paper's Stage 1 Mixed Color run (runs/train/downloaded/soil_moisture_stir_september)
S1_INIT = os.path.join(PROJECT_DIR, "yolov5s.pt")
S1_HYP = os.path.join(PROJECT_DIR, "data", "hyps", "hyp.stir-soil.yaml")
PROJECT_OUT = "runs/stir_reg"
HYP_DIR = os.path.join(PROJECT_OUT, "hyps")

# arm -> (hyp overrides, label smoothing, epochs)
ARMS = {
    "ls":        ({},                                     0.1, 200),
    "wd":        ({"weight_decay": 0.05},                 0.0, 200),
    "ls_wd":     ({"weight_decay": 0.05},                 0.1, 200),
    "ls_wd_e50": ({"weight_decay": 0.05},                 0.1, 50),
    "ls_wd_lr":  ({"weight_decay": 0.05, "lr0": 0.0003},  0.1, 200),
    # Round 2. Round 1 (2026-10-12): weight decay alone does nothing; label
    # smoothing halves the final val loss; ls_wd_lr stays flat to about epoch 90
    # and then climbs, so keep its recipe and end the cosine schedule earlier.
    "ls_wd_lr_e60":   ({"weight_decay": 0.05, "lr0": 0.0003}, 0.1, 60),
    "ls_wd_lr_e100":  ({"weight_decay": 0.05, "lr0": 0.0003}, 0.1, 100),
    "ls2_wd_lr_e100": ({"weight_decay": 0.05, "lr0": 0.0003}, 0.2, 100),
    # Stage 1 (COCO init, hyp.stir-soil.yaml base); see the docstring.
    "s1_ls_wd":          ({"weight_decay": 0.05},                0.1, 200),
    "s1_ls_wd_lr":       ({"weight_decay": 0.05, "lr0": 0.001},  0.1, 200),
    "s1_ls_wd_lr_e100":  ({"weight_decay": 0.05, "lr0": 0.001},  0.1, 100),
    "s1_ls_wd_lr_e60":   ({"weight_decay": 0.05, "lr0": 0.001},  0.1, 60),
    "s1_ls_wd_lr3_e100": ({"weight_decay": 0.05, "lr0": 0.003},  0.1, 100),
}
SELECTED = [x.strip() for x in os.environ.get("STIR_REG_ARMS", ",".join(ARMS)).split(",") if x.strip()]
SEEDS = [int(x) for x in os.environ.get("STIR_REG_SEEDS", "0,1,2").split(",") if x.strip()]


def stage_files(arm):
    """(init weights, base hyp) for an arm's stage."""
    return (S1_INIT, S1_HYP) if arm.startswith("s1_") else (INIT, BASE_HYP)


def hyp_for(arm):
    with open(stage_files(arm)[1]) as f:
        hyp = yaml.safe_load(f)
    hyp.update(ARMS[arm][0])
    os.makedirs(HYP_DIR, exist_ok=True)
    path = os.path.join(HYP_DIR, f"hyp.{arm}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    return path


def train(arm, seed):
    name = f"stir__{arm}__seed{seed}"
    if os.path.isfile(os.path.join(PROJECT_OUT, name, "weights", "last.pt")):
        print(f"skip {name}: already trained", flush=True)
        return
    _, smoothing, epochs = ARMS[arm]
    cmd = [
        PYTHON, os.path.join(PROJECT_DIR, "train.py"),
        "--img", "640",
        "--epochs", str(epochs),
        "--batch-size", "32",
        "--data", DATA,
        "--weights", stage_files(arm)[0],
        "--hyp", hyp_for(arm),
        "--device", "",
        "--optimizer", "AdamW",
        "--cos-lr",
        "--label-smoothing", str(smoothing),
        "--patience", "0",            # 0 = never stop early; the full curve is the evidence
        "--save-period", "5",         # rescored without smoothing by eval_stir_regularize.py
        "--seed", str(seed),
        "--project", PROJECT_OUT,
        "--name", name,
        "--exist-ok",
    ]
    print("+ " + " ".join(cmd), flush=True)
    print(f"Return code: {subprocess.run(cmd).returncode}", flush=True)
    # Periodic checkpoints carry EMA and AdamW state (~85 MB each, ~50 GB over
    # the sweep). Keep the FP16 EMA weights only, as train.py does for last.pt.
    for ckpt in sorted(glob.glob(os.path.join(PROJECT_OUT, name, "weights", "epoch*.pt"))):
        strip_optimizer(ckpt)


def main():
    unknown = [a for a in SELECTED if a not in ARMS]
    if unknown:
        sys.exit(f"ERROR: unknown arms {unknown}; choose from {list(ARMS)}")
    needed = {DATA} | {p for a in SELECTED for p in stage_files(a)}
    for path in sorted(needed):
        if not os.path.isfile(path):
            sys.exit(f"ERROR: {path} not found")
    for arm in SELECTED:
        for seed in SEEDS:
            train(arm, seed)


if __name__ == "__main__":
    main()
