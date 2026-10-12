#!/bin/bash
#SBATCH --job-name=threearm
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/projects/kp9e/Irrigation-Laser-Yolo2/logs/threearm_%j.out
#SBATCH --error=/projects/kp9e/Irrigation-Laser-Yolo2/logs/threearm_%j.err

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

# ==================================================
# Preflight: all three arms must exist for every dataset, with matching
# image counts. A missing or short split would silently bias the comparison.
# ==================================================
BASE="data/training-data"
FAIL=0
for DS in soil-moisture-v4-uv soil-moisture-v4 soil-moisture-v4-september-combined; do
    if [ ! -f "$BASE/arm_yamls/${DS}__laser.yaml" ]; then
        echo "ERROR: missing $BASE/arm_yamls/${DS}__laser.yaml"
        FAIL=1
    fi
    for ARM in downloaded nolaser sham; do
        if [ "$ARM" != "downloaded" ] && [ ! -f "$BASE/$ARM/$DS/data.yaml" ]; then
            echo "ERROR: missing $BASE/$ARM/$DS/data.yaml"
            FAIL=1
            continue
        fi
        for SPLIT in train valid test; do
            N=$(ls "$BASE/$ARM/$DS/$SPLIT/images" 2>/dev/null | wc -l)
            if [ "$N" -eq 0 ]; then
                echo "ERROR: $BASE/$ARM/$DS/$SPLIT/images is empty"
                FAIL=1
            fi
        done
    done
    A=$(ls "$BASE/downloaded/$DS/train/images" 2>/dev/null | wc -l)
    B=$(ls "$BASE/nolaser/$DS/train/images" 2>/dev/null | wc -l)
    C=$(ls "$BASE/sham/$DS/train/images" 2>/dev/null | wc -l)
    if [ "$A" -ne "$B" ] || [ "$A" -ne "$C" ]; then
        echo "ERROR: $DS train counts differ across arms: $A/$B/$C"
        FAIL=1
    fi
done
if [ "$FAIL" -ne 0 ]; then
    echo "Preflight failed. The laser-removed and sham arms are built locally and"
    echo "uploaded; they are not regenerated here. Re-extract the threearm archive"
    echo "into the project root so that these exist:"
    echo "  $BASE/nolaser/<dataset>/{train,valid,test}"
    echo "  $BASE/sham/<dataset>/{train,valid,test}"
    echo "  $BASE/arm_yamls/<dataset>__laser.yaml"
    exit 1
fi
echo "Preflight OK: three arms present with matching counts."

# ==================================================
# Run the three-arm study (3 datasets x 3 arms x 3 seeds = 27 runs)
# ==================================================
echo "Running scripts/training/tr_threearm.py..."
python scripts/training/tr_threearm.py

echo "Job finished at $(date)"
