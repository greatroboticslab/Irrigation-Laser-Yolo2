#!/bin/bash
#SBATCH --job-name=run_fine_tuning
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/projects/kp9e/Irrigation-Laser-Yolo2/logs/run_fine_tuning_%j.out
#SBATCH --error=/projects/kp9e/Irrigation-Laser-Yolo2/logs/run_fine_tuning_%j.err

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
# Run Fine-tuning Script
# ==================================================
cd /projects/kp9e/Irrigation-Laser-Yolo2
echo "Running fine_tuning.py..."
python fine_tuning.py

# ==================================================
# Analyze and Plot Fine-Tuned Results
# ==================================================
echo "Running analyze.py..."
python analyze.py --runs-root runs/fine-tuned --out-dir fine-tuned-results

echo "Running plot_results.py..."
python plot_results.py --runs-root runs/fine-tuned --out-dir fine-tuned-results

echo "Job finished at $(date)"
