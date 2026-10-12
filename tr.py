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

# Datasets that benefit from STIR-specific augmentation (overhead pot images)
STIR_DATASETS = {
    'soil_moisture_stir_september',
    # soil_moisture_september removed — too few images (46 train) for aggressive
    # augmentation; use default hyp so it can learn the base pattern first
    'soil-moisture-v4',
    'soil-moisture-v4-uv',
}
STIR_HYP = "/projects/kp9e/Irrigation-Laser-Yolo2/data/hyps/hyp.stir-soil.yaml"

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
        use_stir_hyp = any(s in dataset for s in STIR_DATASETS)
        cmd = [
            PYTHON, TRAIN_SCRIPT,
            "--img", "640",
            "--epochs", "200",
            "--batch-size", "32",   # yolov5s on A100 (40GB); 32 is safe with headroom
            "--data", yaml_path,
            "--weights", "yolov5s.pt",  # smaller model — less overfitting on limited data
            "--device", "",  # auto-detect GPU assigned by SLURM
            "--optimizer", "AdamW",
            "--cos-lr",
            "--patience", "100",
            "--project", f"runs/train/{name}/",
            "--name", dataset,
        ]
        if use_stir_hyp:
            cmd += ["--hyp", STIR_HYP]
            print(f"  Using STIR augmentation hyp: {STIR_HYP}", flush=True)
        result = subprocess.run(cmd)
        print(f"Return code: {result.returncode}", flush=True)