"""
Comparison chart for selected soil moisture datasets:
  - Soil Moisture IR Combined (5sagf + ir)
  - Soil Moisture v4 + September
  - Soil Moisture v4 IR
  - Soil Moisture v4 UV
  - Soil Moisture Stir September

Usage:
    python compare_selected.py

Output:
    plots/compare_selected.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Selected datasets ──────────────────────────────────────────────────────────
SELECTED = {
    'soil-moisture-ir-combined': {
        'label': 'IR Combined\n(5sagf + ir)',
        'color': '#e6194b',
        'runs_dir': 'runs/train/merged',
    },
    'soil-moisture-v4-september-combined': {
        'label': 'v4 + September\n(combined)',
        'color': '#3cb44b',
        'runs_dir': 'runs/train/merged',
    },
    'soil-moisture-v4-ir': {
        'label': 'v4 IR\n(levels: 2,4,5,6,8,9,10)',
        'color': '#4363d8',
        'runs_dir': 'runs/train/downloaded',
    },
    'soil-moisture-v4-uv': {
        'label': 'v4 UV\n(levels: 0–10)',
        'color': '#f58231',
        'runs_dir': 'runs/train/downloaded',
    },
    'soil_moisture_stir_september': {
        'label': 'Stir September',
        'color': '#911eb4',
        'runs_dir': 'runs/train/downloaded',
    },
}

# ── Helper: find best run ──────────────────────────────────────────────────────
def find_best_run(base_name):
    """Return the DataFrame with the highest peak mAP@0.5 across all run groups."""
    best_df   = None
    best_map  = -1
    best_run  = None
    train_root = 'runs/train'

    if not os.path.isdir(train_root):
        print(f"  ERROR: '{train_root}' not found.")
        return None

    for group in os.listdir(train_root):
        group_dir = os.path.join(train_root, group)
        if not os.path.isdir(group_dir):
            continue

        for entry in sorted(os.listdir(group_dir)):
            suffix = entry[len(base_name):]
            if entry == base_name or (entry.startswith(base_name) and suffix.isdigit()):
                csv_path = os.path.join(group_dir, entry, 'results.csv')
                if not os.path.isfile(csv_path):
                    continue
                try:
                    df = pd.read_csv(csv_path, skipinitialspace=True)
                    df.columns = df.columns.str.strip()
                    peak = df['metrics/mAP_0.5'].max()
                    if peak > best_map:
                        best_map = peak
                        best_df  = df
                        best_run = os.path.join(group, entry)
                except Exception as e:
                    print(f"  Warning: could not read {csv_path}: {e}")

    if best_run:
        print(f"  Best run: {best_run}  →  peak mAP@0.5 = {best_map:.4f}")
    else:
        print(f"  No runs found for '{base_name}'")

    return best_df

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    out_dir = 'plots'
    os.makedirs(out_dir, exist_ok=True)

    loaded = {}   # name → (info, df, peak_map)

    for name, info in SELECTED.items():
        print(f"\nLoading: {name}")
        df = find_best_run(name)
        if df is None:
            continue
        peak = df['metrics/mAP_0.5'].max()
        loaded[name] = (info, df, peak)

    if not loaded:
        print("No data found. Check your runs/train/ directory.")
        return

    # ── Figure layout: 2 rows ──────────────────────────────────────────────────
    # Row 1: mAP@0.5 curves over epochs (all datasets on one chart)
    # Row 2: Bar chart of peak mAP values
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Selected Dataset Comparison', fontsize=15, fontweight='bold', y=0.98)

    ax_line = fig.add_axes([0.07, 0.38, 0.88, 0.52])   # top: line chart
    ax_bar  = fig.add_axes([0.07, 0.06, 0.88, 0.25])   # bottom: bar chart

    # ── Row 1: mAP@0.5 over Epochs ────────────────────────────────────────────
    for name, (info, df, peak) in loaded.items():
        epochs = df['epoch']
        map_vals = df['metrics/mAP_0.5']
        ax_line.plot(
            epochs, map_vals,
            label=f"{info['label'].replace(chr(10), ' ')}  (peak {peak:.4f})",
            color=info['color'],
            linewidth=2,
        )
        # Mark peak
        best_epoch = df.loc[df['metrics/mAP_0.5'].idxmax(), 'epoch']
        ax_line.scatter([best_epoch], [peak], color=info['color'], s=60, zorder=5)

    ax_line.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='0.5 threshold')
    ax_line.set_title('mAP@0.5 over Training Epochs', fontsize=12, fontweight='bold')
    ax_line.set_xlabel('Epoch', fontsize=10)
    ax_line.set_ylabel('mAP@0.5', fontsize=10)
    ax_line.set_ylim(0, 1.05)
    ax_line.legend(fontsize=8.5, loc='lower right')
    ax_line.grid(True, alpha=0.3)

    # ── Row 2: Peak mAP bar chart ─────────────────────────────────────────────
    names  = list(loaded.keys())
    labels = [loaded[n][0]['label'] for n in names]
    peaks  = [loaded[n][2] for n in names]
    colors = [loaded[n][0]['color'] for n in names]

    bars = ax_bar.bar(range(len(names)), peaks, color=colors, edgecolor='white', width=0.5)

    for bar, val in zip(bars, peaks):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold',
        )

    ax_bar.set_xticks(range(len(names)))
    ax_bar.set_xticklabels(labels, fontsize=8.5)
    ax_bar.set_ylabel('Best mAP@0.5', fontsize=10)
    ax_bar.set_ylim(0, 1.15)
    ax_bar.set_title('Peak mAP@0.5 per Dataset', fontsize=12, fontweight='bold')
    ax_bar.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax_bar.grid(axis='y', alpha=0.3)

    out_path = os.path.join(out_dir, 'compare_selected.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {out_path}")

if __name__ == '__main__':
    main()
