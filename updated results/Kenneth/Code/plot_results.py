"""
Plot validation loss, validation accuracy (precision/recall), and mAP
for each soil moisture dataset from YOLOv5 results.csv files.

Usage:
    # After Step 1+2 (baseline runs):
    python plot_results.py --out-dir first-run

    # After Step 4 (fine-tuned runs):
    python plot_results.py --out-dir fine-tuned-results --runs-root runs/fine-tuned

Outputs:
    <out-dir>/  - one PNG per dataset showing training curves
    <out-dir>/comparison.png - bar chart comparing best mAP across datasets
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server environments

# ── CLI args ───────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Plot YOLOv5 training results.")
parser.add_argument(
    "--out-dir",
    default="plots",
    help="Directory to save output plots (default: plots)",
)
parser.add_argument(
    "--runs-root",
    default="runs/train",
    help="Root directory to search for training runs (default: runs/train)",
)
args = parser.parse_args()

# ── Dataset definitions ────────────────────────────────────────────────────────
# Maps each base dataset name to its moisture level classes and results dir.
# runs_dir is only used as a label hint; find_best_run searches args.runs_root.
DATASETS = {
    'soil-moisture-ir': {
        'classes': ['1.0', '2.0', '3.0', '5.0', '8.2'],
        'label': 'Soil Moisture IR\n(levels: 1.0, 2.0, 3.0, 5.0, 8.2)',
    },
    'soil-moisture-v4': {
        'classes': ['0','1','2','3','4','5','6','7','8','9','10'],
        'label': 'Soil Moisture v4\n(levels: 0–10)',
    },
    'soil-moisture-v4-ir': {
        'classes': ['2','4','5','6','8','9','10'],
        'label': 'Soil Moisture v4 IR\n(levels: 2,4,5,6,8,9,10)',
    },
    'soil-moisture-v4-uv': {
        'classes': ['0','1','2','3','4','5','6','7','8','9','10'],
        'label': 'Soil Moisture v4 UV\n(levels: 0–10)',
    },
    'soil-moisture-5sagf': {
        'classes': [],
        'label': 'Soil Moisture 5sagf',
    },
    'soil_moisture_september': {
        'classes': [],
        'label': 'Soil Moisture September',
    },
    'soil_moisture_stir_september': {
        'classes': [],
        'label': 'Soil Moisture Stir September',
    },
    'soil-moisture-ir-combined': {
        'classes': ['1.0', '2.0', '3.0', '5.0', '8.2'],
        'label': 'Soil Moisture IR Combined\n(5sagf + ir)',
    },
    'soil-moisture-v4-september-combined': {
        'classes': ['0','1','2','3','4','5','6','7','8','9','10'],
        'label': 'Soil Moisture v4 + September\n(levels: 0–10)',
    },
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def find_best_run(base_name, runs_root):
    """
    Search runs_root for all runs matching base_name (with optional numeric
    suffix), and return the DataFrame with the highest peak mAP@0.5.
    """
    best_df = None
    best_map = -1
    best_run = None

    if not os.path.isdir(runs_root):
        return None

    def try_csv(csv_path, run_label):
        nonlocal best_df, best_map, best_run
        if not os.path.isfile(csv_path):
            return
        try:
            df = pd.read_csv(csv_path, skipinitialspace=True)
            df.columns = df.columns.str.strip()
            peak = df['metrics/mAP_0.5'].max()
            if peak > best_map:
                best_map = peak
                best_df = df
                best_run = run_label
        except Exception as e:
            print(f"  Warning: could not read {csv_path}: {e}")

    for entry in sorted(os.listdir(runs_root)):
        entry_path = os.path.join(runs_root, entry)
        if not os.path.isdir(entry_path):
            continue

        suffix = entry[len(base_name):]
        if entry == base_name or (entry.startswith(base_name) and suffix.isdigit()):
            # Flat structure: runs_root/dataset/results.csv
            try_csv(os.path.join(entry_path, 'results.csv'), entry)
        else:
            # Two-level structure: runs_root/group/dataset/results.csv
            for sub in sorted(os.listdir(entry_path)):
                sub_suffix = sub[len(base_name):]
                if sub == base_name or (sub.startswith(base_name) and sub_suffix.isdigit()):
                    try_csv(os.path.join(entry_path, sub, 'results.csv'), os.path.join(entry, sub))

    if best_run:
        print(f"  Best run: {best_run} (mAP@0.5 = {best_map:.4f})")

    return best_df


def plot_dataset(name, info, df, out_dir):
    """Plot val loss, precision/recall, and mAP curves for one dataset."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(f'Training Metrics: {info["label"]}', fontsize=13, fontweight='bold')

    epochs = df['epoch']

    # ── 1. Validation Loss ─────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, df['val/box_loss'], label='Box Loss', color='royalblue')
    ax.plot(epochs, df['val/obj_loss'], label='Obj Loss', color='darkorange')
    ax.plot(epochs, df['val/cls_loss'], label='Cls Loss', color='green')
    total_val_loss = df['val/box_loss'] + df['val/obj_loss'] + df['val/cls_loss']
    ax.plot(epochs, total_val_loss, label='Total Val Loss', color='red', linewidth=2, linestyle='--')
    ax.set_title('Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 2. Validation Accuracy (Precision & Recall) ────────────────────────────
    ax = axes[1]
    ax.plot(epochs, df['metrics/precision'], label='Precision', color='royalblue')
    ax.plot(epochs, df['metrics/recall'],    label='Recall',    color='darkorange')
    ax.set_title('Validation Accuracy (Precision & Recall)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 3. mAP ────────────────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(epochs, df['metrics/mAP_0.5'],      label='mAP@0.5',      color='royalblue', linewidth=2)
    ax.plot(epochs, df['metrics/mAP_0.5:0.95'], label='mAP@0.5:0.95', color='darkorange', linewidth=2)

    best_map = df['metrics/mAP_0.5'].max()
    best_epoch = df.loc[df['metrics/mAP_0.5'].idxmax(), 'epoch']
    ax.axvline(x=best_epoch, color='gray', linestyle=':', alpha=0.7)
    ax.annotate(f'Best: {best_map:.4f}\n(epoch {int(best_epoch)})',
                xy=(best_epoch, best_map),
                xytext=(best_epoch + max(1, len(epochs) * 0.05), best_map - 0.05),
                fontsize=9, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray'))

    ax.set_title('mAP over Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'{name}.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")
    return best_map


def plot_comparison(results, out_dir):
    """Bar chart comparing best mAP@0.5 across all datasets."""
    labels = [info['label'] for _, info, _ in results]
    values = [best_map for _, _, best_map in results]
    colors = ['royalblue' if v > 0 else 'lightcoral' for v in values]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor='white', width=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Best mAP@0.5')
    ax.set_ylim(0, 1.1)
    ax.set_title('Best Validation mAP@0.5 per Soil Moisture Dataset', fontsize=13, fontweight='bold')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='0.5 threshold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'comparison.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    out_dir = args.out_dir
    runs_root = args.runs_root
    os.makedirs(out_dir, exist_ok=True)

    print(f"Searching runs in : {runs_root}")
    print(f"Saving plots to   : {out_dir}\n")

    comparison_results = []

    # Build the list of dataset names to process:
    # - Default (runs/train): iterate the full DATASETS dict.
    # - Custom runs-root: scan the directory and plot everything found there.
    #   Use DATASETS for metadata if available, otherwise fall back to the
    #   folder name as the label.
    if runs_root == "runs/train":
        names_to_process = list(DATASETS.keys())
    else:
        SKIP_RUNS = {'drone-training-sonxh', 'hard-hat-sample-w9zqx', 'yolonas'}
        names_to_process = [
            d for d in sorted(os.listdir(runs_root))
            if os.path.isdir(os.path.join(runs_root, d)) and d not in SKIP_RUNS
        ]

    for name in names_to_process:
        info = DATASETS.get(name, {'label': name, 'classes': []})
        print(f"Processing: {name}")
        df = find_best_run(name, runs_root)

        if df is None:
            print(f"  No results found, skipping.")
            continue

        print(f"  Found {len(df)} epochs, peak mAP@0.5 = {df['metrics/mAP_0.5'].max():.4f}")
        best_map = plot_dataset(name, info, df, out_dir)
        comparison_results.append((name, info, best_map))

    if comparison_results:
        print("\nGenerating comparison chart...")
        plot_comparison(comparison_results, out_dir)

    print(f"\nDone. All plots saved to '{out_dir}/'")


if __name__ == '__main__':
    main()
