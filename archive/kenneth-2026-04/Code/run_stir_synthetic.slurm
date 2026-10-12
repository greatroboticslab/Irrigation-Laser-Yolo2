#!/bin/bash
#SBATCH --job-name=stir_synthetic
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/projects/kp9e/Irrigation-Laser-Yolo2/logs/stir_synthetic_%j.out
#SBATCH --error=/projects/kp9e/Irrigation-Laser-Yolo2/logs/stir_synthetic_%j.err

echo "Job started on $(hostname) at $(date)"

mkdir -p /projects/kp9e/Irrigation-Laser-Yolo2/logs

# ==================================================
# Load CUDA module
# ==================================================
module load cuda/12.4
export LD_LIBRARY_PATH=/usr/lib64:/opt/ohpc/pub/apps/cuda/12.4/lib64:$LD_LIBRARY_PATH

# ==================================================
# Git executable for gitpython (not in PATH on compute nodes)
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

# ==================================================
# Verify CUDA
# ==================================================
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"

# ==================================================
# Train — soil_moisture_stir_september only
# Changes vs stir_september2 (best so far, mAP 0.567):
#   - model: yolov5s (tr.py now trains all datasets with yolov5s, so v4 best.pt
#     is yolov5s-compatible — transfer learning restored)
#   - weights: yolov5s v4 best.pt (domain transfer from probe detection)
#   - batch-size: 32 (A100 headroom with yolov5s)
#   - patience: 30 (stir peaks early ~epoch 100 then degrades — stop sooner)
#   - mosaic/mixup: still disabled in hyp.stir-soil.yaml
#   - freeze: none (freeze 10 was catastrophic, 0.204)
# ==================================================
cd /projects/kp9e/Irrigation-Laser-Yolo2

PYTHON="/projects/kp9e/Irrigation-Laser-Yolo2/venv/bin/python"
DATA_YAML="/projects/kp9e/Irrigation-Laser-Yolo2/data/training-data/downloaded/soil_moisture_stir_september/data.yaml"
STIR_HYP="/projects/kp9e/Irrigation-Laser-Yolo2/data/hyps/hyp.stir-soil.yaml"
V4_WEIGHTS="/projects/kp9e/Irrigation-Laser-Yolo2/runs/train/downloaded/soil-moisture-v4/weights/best.pt"

echo "Training on: $DATA_YAML"
echo "Starting weights: $V4_WEIGHTS"

$PYTHON train.py \
    --img 640 \
    --epochs 200 \
    --batch-size 32 \
    --data "$DATA_YAML" \
    --weights "$V4_WEIGHTS" \
    --device "" \
    --optimizer AdamW \
    --cos-lr \
    --patience 30 \
    --hyp "$STIR_HYP" \
    --project runs/train/stir-finetune \
    --name soil_moisture_stir_september5

echo "Training done at $(date)"

# ==================================================
# Analyze results
# ==================================================
echo "Running analyze.py..."
python analyze.py \
    --runs-root runs/train/stir-finetune \
    --out-dir stir-finetune-results

echo "Running plot_results.py..."
python plot_results.py \
    --runs-root runs/train/stir-finetune \
    --out-dir stir-finetune-results

echo "Job finished at $(date)"
