#!/bin/bash
#SBATCH --job-name=gen_synthetic
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/projects/kp9e/Irrigation-Laser-Yolo2/logs/gen_synthetic_%j.out
#SBATCH --error=/projects/kp9e/Irrigation-Laser-Yolo2/logs/gen_synthetic_%j.err

echo "Job started on $(hostname) at $(date)"

mkdir -p /projects/kp9e/Irrigation-Laser-Yolo2/logs

# ==================================================
# Environment — required for CUDA and cv2 on cluster
# ==================================================
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/projects/kp9e/Irrigation-Laser-Yolo2/cv2_pkg:/projects/kp9e/Irrigation-Laser-Yolo2/venv/lib64/python3.9/site-packages:/projects/kp9e/Irrigation-Laser-Yolo2/venv/lib/python3.9/site-packages:$PYTHONPATH
export GIT_PYTHON_GIT_EXECUTABLE=/usr/bin/git
# Hugging Face token is read from ~/.hf_token on the cluster (chmod 600),
# never hardcoded here. Create it once with:
#   printf %s "$YOUR_TOKEN" > ~/.hf_token && chmod 600 ~/.hf_token
if [ -r "$HOME/.hf_token" ]; then
    export HUGGING_FACE_HUB_TOKEN="$(cat "$HOME/.hf_token")"
fi
if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "ERROR: HUGGING_FACE_HUB_TOKEN is unset and ~/.hf_token is missing." >&2
    exit 1
fi

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
# Install / upgrade diffusers dependencies for FLUX.1-schnell
# ==================================================
pip install --quiet "diffusers==0.30.3" "transformers==4.44.2" "accelerate==0.33.0" sentencepiece protobuf opencv-python-headless

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
    --model   black-forest-labs/FLUX.1-schnell \
    --steps   4 \
    --seed    42 \
    --require-gpu

echo "Job finished at $(date)"
