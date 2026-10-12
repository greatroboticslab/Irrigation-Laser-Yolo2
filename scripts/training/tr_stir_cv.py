"""Stir-probe overfitting remedy, measured by grouped cross-validation.

The paper's Stage-3 stir run overfits: validation classification loss bottoms at
epoch 57 and mAP@0.5 falls from 0.711 to 0.480 by the end. The training set is
39 photographs at 22 frames each, so augmentation multiplied frames, not scenes.
Two cheap remedies are compared here, on the folds built by scripts/data_prep/build_stir_cv.py:

  full    the paper's exact Stage-3 recipe: IR-Pot-Background init,
          data/hyps/hyp.stir-finetune.yaml (copied from that run), AdamW, cos-lr
  freeze  the same, with the 10 backbone layers frozen so only the neck and head
          can adapt to 45 scenes

Early stopping is evaluated without selection bias. Every run trains a fixed
EPOCHS budget with patience disabled and saves a checkpoint every SAVE_PERIOD
epochs. scripts/evaluation/eval_stir_cv.py then scores each saved checkpoint on its held-out fold,
so the stopping epoch is chosen from the cross-validated curve, not from the
same fold it is reported on.

Results go to runs/stir_cv/, leaving every other runs/ tree alone.

Usage (from the project root):
  python scripts/data_prep/build_stir_cv.py
  python scripts/training/tr_stir_cv.py
  python scripts/evaluation/eval_stir_cv.py
Environment overrides: STIR_CV_ARMS=full,freeze  STIR_CV_SEEDS=0,1,2
                       STIR_CV_EPOCHS=100  STIR_CV_SAVE_PERIOD=10
                       STIR_CV_FOLDS=fold00,fold01   (default: all folds)
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = "/projects/kp9e/Irrigation-Laser-Yolo2"
if not os.path.isdir(PROJECT_DIR):  # allow local smoke tests
    PROJECT_DIR = str(Path(__file__).resolve().parents[2])
PYTHON = os.path.join(PROJECT_DIR, "venv", "bin", "python")
if not os.path.isfile(PYTHON):
    PYTHON = sys.executable
os.chdir(PROJECT_DIR)

CV_ROOT = os.path.join(PROJECT_DIR, "data", "training-data", "cv", "stir")
INIT = os.path.join(PROJECT_DIR, "runs", "train", "downloaded", "soil-moisture-ir", "weights", "best.pt")
HYP = os.path.join(PROJECT_DIR, "data", "hyps", "hyp.stir-finetune.yaml")
PROJECT_OUT = "runs/stir_cv"

ARMS = tuple(x.strip() for x in os.environ.get("STIR_CV_ARMS", "full,freeze").split(",") if x.strip())
SEEDS = [int(x) for x in os.environ.get("STIR_CV_SEEDS", "0,1,2").split(",") if x.strip()]
EPOCHS = int(os.environ.get("STIR_CV_EPOCHS", "100"))
SAVE_PERIOD = int(os.environ.get("STIR_CV_SAVE_PERIOD", "10"))
ONLY_FOLDS = {x.strip() for x in os.environ.get("STIR_CV_FOLDS", "").split(",") if x.strip()}
FREEZE = {"full": "0", "freeze": "10"}


def folds():
    if not os.path.isdir(CV_ROOT):
        sys.exit(f"ERROR: {CV_ROOT} not found -- run scripts/data_prep/build_stir_cv.py first")
    return sorted(d for d in os.listdir(CV_ROOT)
                  if d.startswith("fold") and (not ONLY_FOLDS or d in ONLY_FOLDS))


def train(fold, arm, seed):
    name = f"stir__{fold}__{arm}__seed{seed}"
    if os.path.isfile(os.path.join(PROJECT_OUT, name, "weights", "last.pt")):
        print(f"skip {name}: already trained", flush=True)
        return
    cmd = [
        PYTHON, os.path.join(PROJECT_DIR, "train.py"),
        "--img", "640",
        "--epochs", str(EPOCHS),
        "--batch-size", "32",
        "--data", os.path.join(CV_ROOT, fold, "data.yaml"),
        "--weights", INIT,
        "--hyp", HYP,
        "--device", "",
        "--optimizer", "AdamW",
        "--cos-lr",
        "--patience", "0",            # 0 = never stop early; stopping is chosen from CV
        "--save-period", str(SAVE_PERIOD),
        "--freeze", FREEZE[arm],
        "--seed", str(seed),
        "--project", PROJECT_OUT,
        "--name", name,
        "--exist-ok",
    ]
    print("+ " + " ".join(cmd), flush=True)
    print(f"Return code: {subprocess.run(cmd).returncode}", flush=True)


def main():
    for path in (INIT, HYP):
        if not os.path.isfile(path):
            sys.exit(f"ERROR: {path} not found")
    for fold in folds():
        for arm in ARMS:
            for seed in SEEDS:
                train(fold, arm, seed)


if __name__ == "__main__":
    main()
