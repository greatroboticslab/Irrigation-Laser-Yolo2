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
BEST_WEIGHTS = "runs/train/downloaded/soil-moisture-v4/weights/best.pt"

# Datasets that were merged into a combined dataset — skip the individual
# components so we only fine-tune on the merged version, not each part.
SKIP_DATASETS = {
    'soil-moisture-v3',                    # 87 raw sensor classes, unusable
    'Soil Moisture v3.v5i.yolov5pytorch',  # duplicate of v3
    'soil-moisture-5sagf',                 # merged into soil-moisture-ir-combined
    'soil-moisture-ir',                    # merged into soil-moisture-ir-combined
    'soil-moisture-v4',                    # merged into soil-moisture-v4-september-combined
                                           # (also source of best.pt weights)
    'soil_moisture_september',             # merged into soil-moisture-v4-september-combined
}

SKIP_GROUPS = {'sean', 'john'}

# Verify best weights exist before starting
if not os.path.isfile(BEST_WEIGHTS):
    print(f"ERROR: Best weights not found at {BEST_WEIGHTS}")
    print("Run tr.py first and wait for soil-moisture-v4 to finish training.")
    exit(1)

print(f"Fine-tuning from: {BEST_WEIGHTS}", flush=True)

for name in sorted(os.listdir(BASE)):
    if name in SKIP_GROUPS:
        print(f"Skipping group: {name}")
        continue
    group_path = os.path.join(BASE, name)
    if not os.path.isdir(group_path):
        continue
    for dataset in sorted(os.listdir(group_path)):
        if dataset in SKIP_DATASETS:
            print(f"Skipping {dataset} (component of a merged dataset or excluded)")
            continue
        dataset_path = os.path.join(group_path, dataset)
        yaml_path = os.path.join(dataset_path, "data.yaml")
        if not os.path.isfile(yaml_path):
            print(f"Skipping {dataset_path} (no data.yaml)")
            continue
        print(f"Fine-tuning on: {yaml_path}", flush=True)
        result = subprocess.run([
            PYTHON, TRAIN_SCRIPT,
            "--img", "640",
            "--epochs", "200",
            "--batch-size", "4",    # reduced to avoid OOM on 2080Ti (11GB VRAM)
            "--data", yaml_path,
            "--weights", BEST_WEIGHTS,
            "--device", "",
            "--optimizer", "AdamW",
            "--cos-lr",
            "--patience", "100",
            "--project", "runs/fine-tuned",
            "--name", dataset,
        ])
        print(f"Return code: {result.returncode}", flush=True)
