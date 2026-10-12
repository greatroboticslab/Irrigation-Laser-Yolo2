#!/bin/bash
#SBATCH --job-name=stir_cv
#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/projects/kp9e/Irrigation-Laser-Yolo2/logs/stir_cv_%j.out
#SBATCH --error=/projects/kp9e/Irrigation-Laser-Yolo2/logs/stir_cv_%j.err

# Stir-probe overfitting remedy: grouped 5-fold CV over the 56 originals,
# paper Stage-3 recipe ("full") vs the same with a frozen backbone ("freeze"),
# 3 seeds each = 30 runs of 100 epochs, checkpoint every 10 epochs, then every
# checkpoint scored out-of-fold. See scripts/data_prep/build_stir_cv.py / scripts/training/tr_stir_cv.py /
# scripts/evaluation/eval_stir_cv.py for the design.
#
# Partition and resources follow slurm/run_cv_v4uv.slurm (kp9e is not in a100_users).
# The original Stage-3 stir run took 0.26 h for 171 epochs on an A100, so the
# 30 runs here should need roughly 5 h on that class of card; 48 h leaves room
# for a slower research-partition GPU. Reruns skip finished runs, so a timed-out
# job can simply be resubmitted.
#
# Requires on the cluster (beyond the existing dataset and venv):
#   scripts/data_prep/build_stir_cv.py  scripts/training/tr_stir_cv.py  scripts/evaluation/eval_stir_cv.py  scripts/evaluation/eval_calibration.py
#   data/hyps/hyp.stir-finetune.yaml
#   runs/cv/soil_moisture_stir_september_groups.csv
#   runs/train/downloaded/soil-moisture-ir/weights/best.pt   (Stage-3 init)
# All but the last ship in stir_cv_sync.tar.gz; extract it at the project root.

echo "Job started on $(hostname) at $(date)"

mkdir -p /projects/kp9e/Irrigation-Laser-Yolo2/logs

# ==================================================
# Load CUDA module
# ==================================================
module load cuda/12.4
export LD_LIBRARY_PATH=/usr/lib64:/opt/ohpc/pub/apps/cuda/12.4/lib64:$LD_LIBRARY_PATH

# ==================================================
# Git executable for gitpython (not in PATH on compute nodes by default)
# ==================================================
export GIT_PYTHON_GIT_EXECUTABLE=/usr/bin/git
export GIT_PYTHON_REFRESH=quiet

# ==================================================
# GPU Info
# ==================================================
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# ==================================================
# Activate Virtual Environment
# ==================================================
ENV_PATH="/projects/kp9e/Irrigation-Laser-Yolo2/venv"
source $ENV_PATH/bin/activate
export PYTHONPATH=/projects/kp9e/Irrigation-Laser-Yolo2/cv2_pkg:$ENV_PATH/lib64/python3.9/site-packages:$ENV_PATH/lib/python3.9/site-packages:$PYTHONPATH
echo "Python: $(which python)"

python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

cd /projects/kp9e/Irrigation-Laser-Yolo2

# STIR_CV_ARMS, STIR_CV_SEEDS, STIR_CV_EPOCHS, STIR_CV_SAVE_PERIOD, STIR_CV_FOLDS
# pass through to scripts/training/tr_stir_cv.py if set, e.g.
#   STIR_CV_SEEDS=0 sbatch slurm/run_stir_cv.slurm        # 10-run pilot
export STIR_CV_ARMS STIR_CV_SEEDS STIR_CV_EPOCHS STIR_CV_SAVE_PERIOD STIR_CV_FOLDS

for f in scripts/data_prep/build_stir_cv.py scripts/training/tr_stir_cv.py scripts/evaluation/eval_stir_cv.py scripts/evaluation/eval_calibration.py \
         data/hyps/hyp.stir-finetune.yaml \
         runs/cv/soil_moisture_stir_september_groups.csv \
         runs/train/downloaded/soil-moisture-ir/weights/best.pt; do
    [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done

# ==================================================
# Build the folds. Deterministic; rebuilding here keeps them in step with the
# imagery actually on the cluster.
# ==================================================
echo "Building grouped stir folds ..."
python scripts/data_prep/build_stir_cv.py || exit 1

# ==================================================
# Preflight: no validation original may share a capture burst with training,
# every copy must sit with its original, and validation must span all five
# levels. A leaking fold would report the inflated number this job exists to
# replace.
# ==================================================
python - <<'PYCHECK' || exit 1
import csv, glob, os, re, sys
R = re.compile(r"_aug(x)?\d+$")
burst = {os.path.basename(r["path"]): r["group"]
         for r in csv.DictReader(open("runs/cv/soil_moisture_stir_september_groups.csv"))}
root = "data/training-data/cv/stir"
folds = sorted(d for d in os.listdir(root) if d.startswith("fold"))
if len(folds) < 2:
    sys.exit("ERROR: fewer than two folds built")
fail, seen = 0, set()
for f in folds:
    stems = {s: {R.sub("", os.path.splitext(b)[0]) for b in os.listdir(f"{root}/{f}/{s}/images")}
             for s in ("train", "valid")}
    if stems["train"] & stems["valid"]:
        print(f"ERROR: {f} has originals split across train and valid"); fail = 1
    bt = {burst[s + ".jpg"] for s in stems["train"]}
    leak = [s for s in stems["valid"] if burst[s + ".jpg"] in bt]
    if leak:
        print(f"ERROR: {f} leaks {len(leak)} validation originals by burst"); fail = 1
    lv = set()
    for p in glob.glob(f"{root}/{f}/valid/labels/*.txt"):
        t = open(p).read().split()
        if t:
            lv.add(int(float(t[0])))
    if len(lv) != 5:
        print(f"ERROR: {f} validation spans {len(lv)} of 5 levels"); fail = 1
    seen |= stems["valid"]
if len(seen) != 56:
    print(f"ERROR: folds hold out {len(seen)} originals, expected 56"); fail = 1
print("Preflight OK: no burst leak, copies with originals, 5 levels per fold, 56 originals covered."
      if not fail else "Preflight FAILED")
sys.exit(fail)
PYCHECK

# ==================================================
# Train (5 folds x 2 arms x 3 seeds = 30 runs), then score out-of-fold.
# ==================================================
echo "Running scripts/training/tr_stir_cv.py ..."
python scripts/training/tr_stir_cv.py

echo "Running scripts/evaluation/eval_stir_cv.py ..."
python scripts/evaluation/eval_stir_cv.py

echo "Job finished at $(date)"
