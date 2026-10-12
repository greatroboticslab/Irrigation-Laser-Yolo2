"""
plot_finetuned.py
Plots training results exclusively from runs/fine-tuned/.

Usage:
    python plot_finetuned.py

Outputs:
    fine-tuned-results/<dataset>.png  - training curves per dataset
    fine-tuned-results/comparison.png - bar chart comparing mAP across datasets
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

RUNS_DIR = "runs/fine-tuned"
OUT_DIR  = "fine-tuned-results"

SKIP = {'drone-training-sonxh', 'hard-hat-sample-w9zqx', 'yolonas'}


def plot_dataset(name, df, out_dir):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(f'Fine-Tuned: {name}', fontsize=13, fontweight='bold')

    epochs = df['epoch']

    # Validation Loss
    ax = axes[0]
    ax.plot(epochs, df['val/box_loss'], label='Box Loss',       color='royalblue')
    ax.plot(epochs, df['val/obj_loss'], label='Obj Loss',       color='darkorange')
    ax.plot(epochs, df['val/cls_loss'], label='Cls Loss',       color='green')
    total = df['val/box_loss'] + df['val/obj_loss'] + df['val/cls_loss']
    ax.plot(epochs, total,              label='Total Val Loss', color='red', linewidth=2, linestyle='--')
    ax.set_title('Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Precision & Recall
    ax = axes[1]
    ax.plot(epochs, df['metrics/precision'], label='Precision', color='royalblue')
    ax.plot(epochs, df['metrics/recall'],    label='Recall',    color='darkorange')
    ax.set_title('Precision & Recall')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # mAP
    ax = axes[2]
    ax.plot(epochs, df['metrics/mAP_0.5'],      label='mAP@0.5',      color='royalblue',  linewidth=2)
    ax.plot(epochs, df['metrics/mAP_0.5:0.95'], label='mAP@0.5:0.95', color='darkorange', linewidth=2)
    best_map   = df['metrics/mAP_0.5'].max()
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
    names  = [name for name, _ in results]
    values = [v    for _, v    in results]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(names)), values, color='royalblue', edgecolor='white', width=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9, rotation=15, ha='right')
    ax.set_ylabel('Best mAP@0.5')
    ax.set_ylim(0, 1.1)
    ax.set_title('Fine-Tuned Results — Best mAP@0.5 per Dataset', fontsize=13, fontweight='bold')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='0.5 threshold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'comparison.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    results = []

    for name in sorted(os.listdir(RUNS_DIR)):
        if name in SKIP:
            continue
        csv_path = os.path.join(RUNS_DIR, name, 'results.csv')
        if not os.path.isfile(csv_path):
            continue

        print(f"Processing: {name}")
        try:
            df = pd.read_csv(csv_path, skipinitialspace=True)
            df.columns = df.columns.str.strip()
            best_map = plot_dataset(name, df, OUT_DIR)
            results.append((name, best_map))
            print(f"  Peak mAP@0.5 = {best_map:.4f}")
        except Exception as e:
            print(f"  Error reading {csv_path}: {e}")

    if results:
        print("\nGenerating comparison chart...")
        plot_comparison(results, OUT_DIR)

    print(f"\nDone. {len(results)} datasets plotted → {OUT_DIR}/")


if __name__ == '__main__':
    main()
