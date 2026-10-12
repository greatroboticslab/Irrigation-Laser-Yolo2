#!/bin/bash
#SBATCH --job-name=check_python
#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --output=/projects/kp9e/Irrigation-Laser-Yolo2/logs/check_python_%j.out
#SBATCH --error=/projects/kp9e/Irrigation-Laser-Yolo2/logs/check_python_%j.err

echo "Node: $(hostname)"
echo "--- Python binaries in /usr/bin ---"
ls /usr/bin/python* 2>/dev/null || echo "none found"
echo "--- Python binaries in /usr/local/bin ---"
ls /usr/local/bin/python* 2>/dev/null || echo "none found"
echo "--- which python3 ---"
which python3 2>/dev/null || echo "not in PATH"
echo "--- python3 version ---"
python3 --version 2>/dev/null || echo "python3 not available"
