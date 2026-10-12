import os
import csv
import subprocess

PROJECT_DIR = "/projects/kp9e/Irrigation-Laser-Yolo2"
TRAIN_SCRIPT = os.path.join(PROJECT_DIR, "train.py")
PYTHON = "/projects/kp9e/Irrigation-Laser-Yolo2/venv/bin/python"
os.chdir(PROJECT_DIR)
print(f"Working directory: {os.getcwd()}", flush=True)
print(f"Python: {PYTHON}", flush=True)
print(f"train.py exists: {os.path.isfile(TRAIN_SCRIPT)}", flush=True)

BASE = "/projects/kp9e/Irrigation-Laser-Yolo2/data/training-data/"
RUNS_DOWNLOADED = os.path.join(PROJECT_DIR, "runs/train/downloaded")

# Datasets that were merged into a combined dataset — skip the individual
# components so we only fine-tune on the merged version, not each part.
SKIP_DATASETS = {
    'soil-moisture-v3',                    # 87 raw sensor classes, unusable
    'Soil Moisture v3.v5i.yolov5pytorch',  # duplicate of v3
    'soil-moisture-5sagf',                 # merged into soil-moisture-ir-combined
    'soil-moisture-ir',                    # merged into soil-moisture-ir-combined
    'soil-moisture-v4',                    # merged into soil-moisture-v4-september-combined
                                           # (also candidate for best.pt weights)
    'soil_moisture_september',             # merged into soil-moisture-v4-september-combined
}

SKIP_GROUPS = {'sean', 'john'}


def find_best_weights(runs_downloaded):
    """
    Scan runs/train/downloaded/ and return the weights/best.pt with the highest
    peak mAP@0.5 across all completed baseline training runs.
    mAP@0.5 is column index 6 in YOLOv5 results.csv (0-indexed).
    """
    best_map = -1.0
    best_weights = None
    best_dataset = None

    if not os.path.isdir(runs_downloaded):
        return None, None, -1.0

    for dataset in sorted(os.listdir(runs_downloaded)):
        weights_path = os.path.join(runs_downloaded, dataset, "weights", "best.pt")
        results_path = os.path.join(runs_downloaded, dataset, "results.csv")

        if not os.path.isfile(weights_path) or not os.path.isfile(results_path):
            continue

        try:
            with open(results_path, newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
            # rows[0] is the header; data starts at rows[1]
            data_rows = [r for r in rows[1:] if len(r) > 6]
            if not data_rows:
                continue
            dataset_best = max(float(r[6]) for r in data_rows)
            print(f"  {dataset}: best mAP@0.5 = {dataset_best:.4f}", flush=True)
            if dataset_best > best_map:
                best_map = dataset_best
                best_weights = weights_path
                best_dataset = dataset
        except Exception as e:
            print(f"  Could not read {results_path}: {e}", flush=True)
            continue

    return best_weights, best_dataset, best_map


print("Scanning baseline runs to find best starting weights ...", flush=True)
BEST_WEIGHTS, best_source, best_map_val = find_best_weights(RUNS_DOWNLOADED)

if BEST_WEIGHTS is None or not os.path.isfile(BEST_WEIGHTS):
    print("ERROR: No valid baseline weights found in runs/train/downloaded/")
    print("Run tr.py first and wait for at least one dataset to finish training.")
    exit(1)

print(f"Selected weights : {BEST_WEIGHTS}", flush=True)
print(f"Source dataset   : {best_source}  (mAP@0.5 = {best_map_val:.4f})", flush=True)

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
            "--batch-size", "32",   # yolov5s on A100 (40GB); 32 is safe with headroom
            "--data", yaml_path,
            "--weights", BEST_WEIGHTS,  # auto-selected best yolov5s baseline weights
            "--device", "",
            "--optimizer", "AdamW",
            "--cos-lr",
            "--patience", "100",
            "--project", "runs/fine-tuned",
            "--name", dataset,
        ])
        print(f"Return code: {result.returncode}", flush=True)
