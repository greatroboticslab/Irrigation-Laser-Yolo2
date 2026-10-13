#!/bin/bash
#SBATCH --job-name=stir_reg
#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/projects/kp9e/Irrigation-Laser-Yolo2/logs/stir_reg_%j.out
#SBATCH --error=/projects/kp9e/Irrigation-Laser-Yolo2/logs/stir_reg_%j.err

# Regularisation sweep for the overfitting Stage 3 Mixed Color and Background
# run: 5 arms (label smoothing, weight decay, both, both + 50 epochs, both +
# lower lr) x 3 seeds = 15 runs on the shipped 858/204/95 split, then every
# 5-epoch checkpoint rescored with unsmoothed loss. See
# scripts/training/tr_stir_regularize.py and scripts/evaluation/eval_stir_regularize.py.
#
# The paper's Stage 3 stir run took 0.26 h for 171 epochs on an A100, so the
# sweep is roughly 3 h of training there; 48 h leaves room for a slower
# research-partition GPU. Reruns skip finished runs, so a timed-out job can be
# resubmitted. Disk: ~40 stripped checkpoints x 14 MB per run, about 8 GB total.
#
# Pilot first if you like:  STIR_REG_SEEDS=0 sbatch slurm/run_stir_regularize.slurm
# Round 2 only:  STIR_REG_ARMS=ls_wd_lr_e60,ls_wd_lr_e100,ls2_wd_lr_e100 sbatch slurm/run_stir_regularize.slurm
# (scoring always rescores every run under runs/stir_reg, round 1 included)

echo "Job started on $(hostname) at $(date)"
mkdir -p /projects/kp9e/Irrigation-Laser-Yolo2/logs

module load cuda/12.4
export LD_LIBRARY_PATH=/usr/lib64:/opt/ohpc/pub/apps/cuda/12.4/lib64:$LD_LIBRARY_PATH
export GIT_PYTHON_GIT_EXECUTABLE=/usr/bin/git
export GIT_PYTHON_REFRESH=quiet

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

ENV_PATH="/projects/kp9e/Irrigation-Laser-Yolo2/venv"
source $ENV_PATH/bin/activate
export PYTHONPATH=/projects/kp9e/Irrigation-Laser-Yolo2/cv2_pkg:$ENV_PATH/lib64/python3.9/site-packages:$ENV_PATH/lib/python3.9/site-packages:$PYTHONPATH
echo "Python: $(which python)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

cd /projects/kp9e/Irrigation-Laser-Yolo2
export STIR_REG_ARMS STIR_REG_SEEDS

for f in scripts/training/tr_stir_regularize.py scripts/evaluation/eval_stir_regularize.py \
         data/hyps/hyp.stir-finetune.yaml \
         data/training-data/downloaded/soil_moisture_stir_september/data.yaml \
         data/training-data/downloaded/soil_moisture_stir_september/test/images \
         runs/train/downloaded/soil-moisture-ir/weights/best.pt; do
    [ -e "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done
[ -f runs/fine-tuned/soil_moisture_stir_september/weights/last.pt ] || \
    echo "NOTE: paper Stage 3 stir run not found; the summary will have no 'paper' reference row"

echo "Running scripts/training/tr_stir_regularize.py ..."
python scripts/training/tr_stir_regularize.py

echo "Running scripts/evaluation/eval_stir_regularize.py ..."
python scripts/evaluation/eval_stir_regularize.py

echo "Job finished at $(date)"
