import os
import csv
import argparse

parser = argparse.ArgumentParser(description="Analyze YOLOv5 training results.")
parser.add_argument(
    "--runs-root",
    default="runs/train",
    help="Root directory to search for training runs (default: runs/train)",
)
parser.add_argument(
    "--out-dir",
    default=None,
    help="Directory to save summary.txt (default: print to stdout only)",
)
args = parser.parse_args()

results = []

def process_csv(csv_path, label):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)
    if len(rows) < 2:
        return
    average = 0.0
    averageSample = min(5, len(rows) - 1)
    for i in range(averageSample):
        average += float(rows[-averageSample + i][6])
    average /= averageSample
    line = f"{label}: {average:.6f}"
    print(line)
    results.append(line)


for name in os.listdir(args.runs_root):
    subdir = os.path.join(args.runs_root, name)
    if not os.path.isdir(subdir):
        continue

    # Flat structure: runs_root/dataset/results.csv
    flat_csv = os.path.join(subdir, "results.csv")
    if os.path.isfile(flat_csv):
        process_csv(flat_csv, name)
        continue

    # Two-level structure: runs_root/group/dataset/results.csv
    for subfile in os.listdir(subdir):
        csv_path = os.path.join(subdir, subfile, "results.csv")
        if os.path.isfile(csv_path):
            process_csv(csv_path, subfile)

if args.out_dir and results:
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "summary.txt")
    with open(out_path, "w") as f:
        f.write(f"runs_root: {args.runs_root}\n\n")
        f.write("\n".join(results) + "\n")
    print(f"\nSummary saved to {out_path}")
