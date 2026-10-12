#!/bin/bash
#SBATCH --job-name=tr_ablation
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/projects/kp9e/Irrigation-Laser-Yolo2/logs/tr_ablation_%j.out
#SBATCH --error=/projects/kp9e/Irrigation-Laser-Yolo2/logs/tr_ablation_%j.err

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

# ==================================================
# Verify CUDA is available
# ==================================================
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

# ==================================================
# Preflight: ablation datasets and classifier weights must exist
# (compute nodes have no internet; yolov5s-cls.pt cannot auto-download there)
# ==================================================
cd /projects/kp9e/Irrigation-Laser-Yolo2
if [ ! -f data/ablation_september/full_laser/data.yaml ]; then
    echo "ERROR: data/ablation_september missing. Run scripts/data_prep/laser_ablation_dataset.py first:"
    echo "  python scripts/data_prep/laser_ablation_dataset.py --data data/soil_september --out data/ablation_september --mask-overlay"
    exit 1
fi
if [ ! -f yolov5s-cls.pt ]; then
    echo "ERROR: yolov5s-cls.pt missing. Download on the login node first:"
    echo "  curl -L -o yolov5s-cls.pt https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s-cls.pt"
    exit 1
fi

# ==================================================
# Run Ablation Training (4 conditions x 3 seeds + summary)
# ==================================================
echo "Running scripts/training/tr_ablation.py..."
python scripts/training/tr_ablation.py

echo "Job finished at $(date)"
