#!/bin/bash
#SBATCH --job-name=gen_synthetic_gpu
#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --exclude=c2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/projects/kp9e/Irrigation-Laser-Yolo2/logs/gen_synthetic_%j.out
#SBATCH --error=/projects/kp9e/Irrigation-Laser-Yolo2/logs/gen_synthetic_%j.err

set -e   # abort immediately if any command fails
echo "Job started on $(hostname) at $(date)"

mkdir -p /projects/kp9e/Irrigation-Laser-Yolo2/logs

# ==================================================
# Load modules (Python must be loaded before activating the venv
# so the venv's python symlink resolves correctly on compute nodes)
# ==================================================
module load cuda/12.4
export LD_LIBRARY_PATH=/usr/lib64:/opt/ohpc/pub/apps/cuda/12.4/lib64:$LD_LIBRARY_PATH
# Python 3.9 is system-wide on compute nodes (/usr/bin/python3.9) — no module needed

# ==================================================
# GPU Info
# ==================================================
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# ==================================================
# Activate Virtual Environment
# ==================================================
source /projects/kp9e/Irrigation-Laser-Yolo2/venv/bin/activate
export PYTHONPATH=/projects/kp9e/Irrigation-Laser-Yolo2/cv2_pkg:/projects/kp9e/Irrigation-Laser-Yolo2/venv/lib64/python3.9/site-packages:/projects/kp9e/Irrigation-Laser-Yolo2/venv/lib/python3.9/site-packages:$PYTHONPATH
echo "Python: $(which python)  — $(python --version)"

# ==================================================
# Install dependencies (separate calls so failures are visible)
# ==================================================
pip install diffusers
pip install transformers
pip install accelerate
pip uninstall -y opencv-python 2>/dev/null || true
pip install opencv-python-headless

# ==================================================
# Verify CUDA and cv2 are available
# ==================================================
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
python -c "import cv2; print('cv2 version:', cv2.__version__)"

# ==================================================
# Run Synthetic Generation
# ==================================================
cd /projects/kp9e/Irrigation-Laser-Yolo2

echo "Running generate_stir_synthetic.py..."
python generate_stir_synthetic.py \
    --n-train    50  \
    --n-valid    15  \
    --n-test     10  \
    --model      runwayml/stable-diffusion-v1-5 \
    --steps      30  \
    --guidance   7.5 \
    --seed       42  \
    --require-gpu

echo "Job finished at $(date)"
