#!/bin/bash
#SBATCH --job-name=run_fine_tuning
#SBATCH --partition=research
#SBATCH --gres=gpu:2080Ti:1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=c5,c7
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/projects/kp9e/Irrigation-Laser-Yolo2/logs/run_fine_tuning_%j.out
#SBATCH --error=/projects/kp9e/Irrigation-Laser-Yolo2/logs/run_fine_tuning_%j.err

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
