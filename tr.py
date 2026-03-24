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

SKIP_DATASETS = {
    'soil-moisture-v3',                       # 87 raw sensor classes, unusable
    'Soil Moisture v3.v5i.yolov5pytorch',     # same dataset, sean version
    'soil-moisture-ir-combined',              # trained separately in tr_merged.py
    'soil-moisture-v4-september-combined',    # trained separately in tr_merged.py
}

SKIP_GROUPS = {'sean', 'john'}

for name in sorted(os.listdir(BASE)):
    if name in SKIP_GROUPS:
        print(f"Skipping group: {name}")
        continue
    group_path = os.path.join(BASE, name)
    if not os.path.isdir(group_path):
        continue
    for dataset in sorted(os.listdir(group_path)):
        if dataset in SKIP_DATASETS:
            print(f"Skipping {dataset} (excluded)")
            continue
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
            "--batch-size", "4",    # reduced to avoid OOM on 2080Ti (11GB VRAM)
            "--data", yaml_path,
            "--weights", "yolov5m.pt",
            "--device", "",  # auto-detect GPU assigned by SLURM
            "--optimizer", "AdamW",
            "--cos-lr",
            "--patience", "100",
            "--project", f"runs/train/{name}/",
            "--name", dataset
        ])
        print(f"Return code: {result.returncode}", flush=True)