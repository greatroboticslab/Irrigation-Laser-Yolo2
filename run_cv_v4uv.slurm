#!/bin/bash
#SBATCH --job-name=cv_v4uv
#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/projects/kp9e/Irrigation-Laser-Yolo2/logs/cv_v4uv_%j.out
#SBATCH --error=/projects/kp9e/Irrigation-Laser-Yolo2/logs/cv_v4uv_%j.err

# Partition note: `research`, not `a100`. Job 53151 sat PENDING with "uid not in
# group permitted to use this partition" because kp9e is not in a100_users.
#
# Resources are deliberately modest. Every job in this repo that actually trained
# asked for 64G on a100; copying that onto a research node is what produced
# "Memory specification can not be satisfied". The only research-partition job
# known to run here (check_python.slurm) uses gpu:1 and 4G, so this follows it.
# Confirm the real ceiling with:
#     sinfo -p research -o "%n %c %m %G %t"
#     scontrol show partition research | grep -i -E "maxmem|defmem|totalnodes"
# and raise --mem toward that node's %m if the run is memory-starved.

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

DATASET="${CV_DATASET:-soil-moisture-v4-uv}"
export CV_DATASET="$DATASET"
# CV_MAX_FOLDS caps the fold count, CV_SEEDS sets the seeds (comma separated).
# Where a dataset holds many bursts per class, folds are the better source of
# variance and one seed is enough:
#   CV_DATASET=soil-moisture-ir CV_MAX_FOLDS=5 CV_SEEDS=0 sbatch run_cv_v4uv.slurm
# Where a class holds only two bursts, two folds is the ceiling and the seeds
# carry the variance instead (the v4 family).
# CV_REGIMES selects which arms to train (default both).
export CV_MAX_FOLDS CV_SEEDS CV_REGIMES

# ==================================================
# Build capture-burst groups and the paired folds.
# Both steps are deterministic and cheap; rebuilding here keeps the folds in
# step with whatever imagery is actually on the cluster rather than trusting an
# upload to have matched.
# ==================================================
echo "Building burst groups for $DATASET ..."
python build_cv_groups.py --dataset "$DATASET" || exit 1

echo "Building leave-one-burst-out folds ..."
python build_cv_folds.py --dataset "$DATASET" || exit 1

# ==================================================
# Preflight: the grouped arm is the whole point, so verify on disk that no
# validation frame shares a capture burst with training. A fold that leaks would
# quietly report the same inflated number the shipped split does, and the run
# would look like a successful negative result.
# ==================================================
python - <<'PYCHECK' || exit 1
import csv, glob, os, sys
ds = os.environ["CV_DATASET"]
REGIMES = tuple(x.strip() for x in os.environ.get("CV_REGIMES", "grouped,random").split(",") if x.strip())
groups = {os.path.basename(r["path"]): r["group"]
          for r in csv.DictReader(open(f"runs/cv/{ds}_groups.csv"))}
root = f"data/training-data/cv/{ds}"
folds = sorted(d for d in os.listdir(root) if d.startswith("fold"))
if not folds:
    sys.exit("ERROR: no folds were built")
fail = 0
for fold in folds:
    for regime in REGIMES:
        r = os.path.join(root, fold, regime)
        tr = [os.path.basename(p) for p in glob.glob(f"{r}/train/images/*")]
        va = [os.path.basename(p) for p in glob.glob(f"{r}/valid/images/*")]
        if not tr or not va:
            print(f"ERROR: {fold}/{regime} has an empty split ({len(tr)} train, {len(va)} val)")
            fail = 1
            continue
        trg = {groups[b] for b in tr if b in groups}
        leak = sum(1 for b in va if groups.get(b) in trg)

        # Validation must span more than one class. A single-class validation set
        # turns the task into "find the LED" with no hue discrimination, and both
        # arms saturate at YOLOv5's 0.9950 ceiling, which looks like agreement and
        # measures nothing. This is how the first version of these folds failed.
        def classes_in(split):
            out = set()
            for lp in glob.glob(f"{r}/{split}/labels/*.txt"):
                for line in open(lp):
                    if line.split():
                        out.add(int(line.split()[0]))
            return out
        val_cls, train_cls = classes_in("valid"), classes_in("train")
        if len(val_cls) < 2:
            print(f"ERROR: {fold}/{regime} validation holds {len(val_cls)} class(es); "
                  f"it must span at least two or the metric saturates")
            fail = 1
        if not val_cls <= train_cls:
            print(f"ERROR: {fold}/{regime} validation contains classes absent from "
                  f"training: {sorted(val_cls - train_cls)}")
            fail = 1

        if regime == "grouped" and leak:
            print(f"ERROR: {fold}/grouped leaks {leak}/{len(va)} validation frames")
            fail = 1
        if regime == "random" and leak == 0:
            # Only meaningful where bursts hold more than one frame. Where every
            # frame is its own burst (soil-moisture-ir and 5sagf are 99 bursts over
            # 100 frames), a random draw cannot share a burst with training, the two
            # arms coincide, and there is simply no leak to measure -- which is the
            # result, not a failure.
            redundant = len(groups) - len(set(groups.values()))
            if redundant > 0:
                print(f"ERROR: {fold}/random leaks nothing despite {redundant} redundant "
                      f"frame(s); it is meant to reproduce the shipped split")
                fail = 1
            else:
                print(f"NOTE: {fold}/random cannot leak -- every frame is its own burst. "
                      f"The two arms coincide on this dataset.")
print("Preflight OK: folds leak-free, controls leak, validation multi-class."
      if not fail else "Preflight FAILED")
sys.exit(fail)
PYCHECK

# ==================================================
# Run the paired cross-validation (2 folds x 2 regimes x 3 seeds = 12 runs)
# ==================================================
echo "Running tr_cv.py ..."
python tr_cv.py

echo "Job finished at $(date)"
