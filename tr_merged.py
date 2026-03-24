import os
import subprocess

PROJECT_DIR = "/projects/kp9e/Irrigation-Laser-Yolo2"
TRAIN_SCRIPT = os.path.join(PROJECT_DIR, "train.py")
PYTHON = "/projects/kp9e/Irrigation-Laser-Yolo2/venv/bin/python"
os.chdir(PROJECT_DIR)
print(f"Working directory: {os.getcwd()}", flush=True)
print(f"Python: {PYTHON}", flush=True)
print(f"train.py exists: {os.path.isfile(TRAIN_SCRIPT)}", flush=True)

DATASETS = [
    "data/training-data/downloaded/soil-moisture-ir-combined/data.yaml",
    "data/training-data/downloaded/soil-moisture-v4-september-combined/data.yaml",
]

for yaml_path in DATASETS:
    dataset_name = os.path.basename(os.path.dirname(yaml_path))
    print(f"Training on: {yaml_path}", flush=True)
    result = subprocess.run([
        PYTHON, TRAIN_SCRIPT,
        "--img", "640",
        "--epochs", "200",
        "--batch-size", "2",    # reduced further for larger merged datasets on 2080Ti
        "--data", yaml_path,
        "--weights", "yolov5m.pt",
        "--device", "",
        "--optimizer", "AdamW",
        "--cos-lr",
        "--patience", "100",
        "--project", "runs/train/merged/",
        "--name", dataset_name,
    ])
    print(f"Return code: {result.returncode}", flush=True)
