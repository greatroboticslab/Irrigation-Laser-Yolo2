#!/bin/bash
#SBATCH --job-name=gen_synthetic
#SBATCH --partition=research
#SBATCH --gres=gpu:2080Ti:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=c5,c7
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/projects/kp9e/Irrigation-Laser-Yolo2/logs/gen_synthetic_%j.out
#SBATCH --error=/projects/kp9e/Irrigation-Laser-Yolo2/logs/gen_synthetic_%j.err

echo "Job started on $(hostname) at $(date)"

mkdir -p /projects/kp9e/Irrigation-Laser-Yolo2/logs

# ==================================================
# Load CUDA module
# ==================================================
module load cuda/12.4

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
echo "Python: $(which python)"

# ==================================================
# Install diffusers dependencies if not present
# ==================================================
pip install --quiet diffusers transformers accelerate opencv-python-headless

# ==================================================
# Verify CUDA is available
# ==================================================
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

# ==================================================
# Run Synthetic Generation
# ==================================================
cd /projects/kp9e/Irrigation-Laser-Yolo2

echo "Running generate_stir_synthetic.py..."
python generate_stir_synthetic.py \
    --n-train 50 \
    --n-valid 15 \
    --n-test  10 \
    --model   runwayml/stable-diffusion-v1-5 \
    --steps   30 \
    --guidance 7.5 \
    --seed    42

echo "Job finished at $(date)"
