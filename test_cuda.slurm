#!/bin/bash
#SBATCH --job-name=test_cuda
#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --nodelist=c6
#SBATCH --time=00:05:00
#SBATCH --output=/projects/kp9e/Irrigation-Laser-Yolo2/logs/test_cuda_%j.out

module load cuda/12.4
export LD_LIBRARY_PATH=/usr/lib64:/opt/ohpc/pub/apps/cuda/12.4/lib64:$LD_LIBRARY_PATH

echo "=== Attempting to load nvidia-uvm ==="
nvidia-modprobe -u -c 0 2>&1 || echo "nvidia-modprobe failed"

echo "=== /dev/nvidia-uvm exists? ==="
ls -la /dev/nvidia-uvm 2>/dev/null || echo "/dev/nvidia-uvm NOT found"

source /projects/kp9e/Irrigation-Laser-Yolo2/venv/bin/activate
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
