import os
import subprocess

PROJECT_DIR = "/projects/kp9e/Irrigation-Laser-Yolo2"
TRAIN_SCRIPT = os.path.join(PROJECT_DIR, "train.py")
PYTHON = "/projects/kp9e/Irrigation-Laser-Yolo2/venv/bin/python"
os.chdir(PROJECT_DIR)
print(f"Working directory: {os.getcwd()}", flush=True)
print(f"Python: {PYTHON}", flush=True)
print(f"train.py exists: {os.path.isfile(TRAIN_SCRIPT)}", flush=True)

BASE = "/projects/kp9e/Irrigation-Laser-Yolo2/data/training-data/"

for name in os.listdir(BASE):
    group_path = os.path.join(BASE, name)
    if not os.path.isdir(group_path):
        continue
    for dataset in os.listdir(group_path):
        dataset_path = os.path.join(group_path, dataset)
        yaml_path = os.path.join(dataset_path, "data.yaml")
        if not os.path.isfile(yaml_path):
            print(f"Skipping {dataset_path} (no data.yaml)")
            continue
        print(f"Training on: {yaml_path}", flush=True)
        result = subprocess.run([
            PYTHON, TRAIN_SCRIPT,
            "--img", "640",
            "--epochs", "200",      # increased from 100 to allow more training time
            "--batch-size", "-1",   # autobatch: YOLOv5 detects the safe batch size for your GPU
            "--data", yaml_path,
            "--weights", "runs/train/downloaded/soil-moisture-v4/weights/best.pt",  # fine-tune from best soil moisture model
            "--device", "",  # auto-detect GPU assigned by SLURM
            "--optimizer", "AdamW", # better convergence than SGD for fine-tuning
            "--cos-lr",             # cosine LR schedule for smooth decay
            "--patience", "100",    # increased to 100 for small datasets that converge slowly
            "--project", f"runs/train/{name}/",
            "--name", dataset
        ])
        print(f"Return code: {result.returncode}", flush=True)