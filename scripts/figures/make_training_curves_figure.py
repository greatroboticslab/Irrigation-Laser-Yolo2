"""Training curves for the runs the paper reports.

Reads the per-epoch `results.csv` that YOLOv5 writes and plots validation mAP
against epoch. Two column groups:

  Stage 1   `results/runs/train/downloaded/<ds>/results.csv`
            the seven single-dataset baselines. These are the LATER runs, which
            is the set the paper quotes -- `first-run/` holds an earlier, lower
            set and mixing the two has already put wrong numbers in a table.
  Stage 3   `results/runs/fine-tuned/<ds>/results.csv`
            fine-tuned from the UV-Soil-Only-Baseline checkpoint.

Every marked point is the best epoch of `metrics/mAP_0.5`, which is the statistic
the manuscript reports -- a maximum over 200 epochs of validation, not a test
score. Nothing in `runs/` ever scored a test split.

Display names come from NAME_MAP; edit there if the manuscript's names change.

Usage (from the project root):
  python scripts/figures/make_training_curves_figure.py
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
UPDATED = ROOT / "results"
STAGE1_DIR = UPDATED / "runs" / "train" / "downloaded"
STAGE3_DIR = UPDATED / "runs" / "fine-tuned"
OUT = [UPDATED / "training_curves.png", UPDATED / "training_curves.pdf"]

# Paper-facing dataset names (supervisor's naming, applied 2026-09-07).
NAME_MAP = {
    "soil-moisture-v4": "UV-Soil-Only-Baseline",
    "soil-moisture-v4-uv": "UV-Soil-Only-Target",
    "soil-moisture-v4-ir": "IR-Soil-Only",
    "soil-moisture-ir": "IR-Pot-Background",
    "soil-moisture-5sagf": "UV-Pot-Background",
    "soil_moisture_september": "UV-Strawberry-Pot",
    "soil_moisture_stir_september": "Mixed Color and Background",
    "soil-moisture-ir-combined": "IR-Combined",
    "soil-moisture-v4-september-combined": "UV-Combined",
}

# Which runs appear in each column. Legend order is descending best mAP, set at
# draw time; colour is fixed per dataset in COLORS so a dataset keeps its colour
# across all four panels -- ranking by panel would recolour the same run between
# the mAP@0.5 and mAP@0.5:0.95 rows.
STAGE1 = ["soil-moisture-ir", "soil-moisture-v4", "soil_moisture_september",
          "soil_moisture_stir_september", "soil-moisture-v4-uv",
          "soil-moisture-5sagf", "soil-moisture-v4-ir"]
STAGE3 = ["soil-moisture-ir-combined", "soil-moisture-v4-ir",
          "soil-moisture-v4-september-combined", "soil-moisture-v4-uv",
          "soil_moisture_stir_september"]

COLORS = {
    "soil-moisture-ir": "#1b5e9c",
    "soil-moisture-v4": "#c0392b",
    "soil_moisture_september": "#2e8b57",
    "soil_moisture_stir_september": "#d68910",
    "soil-moisture-v4-uv": "#8e44ad",
    "soil-moisture-5sagf": "#17a2b8",
    "soil-moisture-v4-ir": "#7f8c8d",
    "soil-moisture-ir-combined": "#1b5e9c",
    "soil-moisture-v4-september-combined": "#2e8b57",
}

MAP50 = "metrics/mAP_0.5"
MAP5095 = "metrics/mAP_0.5:0.95"


def read_curve(path):
    """-> {column: [value per epoch]}. Header fields are space-padded by YOLOv5."""
    cols = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = [c.strip() for c in next(reader)]
        rows = [r for r in reader if r and r[0].strip()]
    for i, name in enumerate(header):
        try:
            cols[name] = [float(r[i]) for r in rows if len(r) > i]
        except ValueError:
            continue
    return cols


def panel(ax, base_dir, datasets, metric, title, ylabel):
    curves = []
    for ds in datasets:
        path = base_dir / ds / "results.csv"
        if not path.is_file():
            print(f"  MISSING {path}")
            continue
        cols = read_curve(path)
        if metric not in cols:
            print(f"  {path}: no {metric} column")
            continue
        curves.append((ds, cols["epoch"], cols[metric]))

    curves.sort(key=lambda c: max(c[2]), reverse=True)
    for ds, epochs, ys in curves:
        color = COLORS.get(ds, "#333333")
        best = max(ys)
        best_ep = epochs[ys.index(best)]
        label = f"{NAME_MAP.get(ds, ds)}  ({best:.3f})"
        ax.plot(epochs, ys, color=color, lw=1.3, label=label)
        ax.plot([best_ep], [best], "o", color=color, ms=4.5,
                mec="white", mew=0.8, zorder=5)

    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)
    # loc="best" rather than a fixed corner: these curves are noisy and fill
    # different regions in each panel, and a pinned legend buried the Stage 1
    # traces it was sitting on.
    ax.legend(fontsize=7, loc="best", framealpha=0.92,
              title="best epoch in parentheses", title_fontsize=7)
    return curves


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    panel(axes[0][0], STAGE1_DIR, STAGE1, MAP50,
          "Stage 1: single-dataset baselines", "val mAP@0.5")
    panel(axes[0][1], STAGE3_DIR, STAGE3, MAP50,
          "Stage 3: fine-tuned from UV-Soil-Only-Baseline", "val mAP@0.5")
    panel(axes[1][0], STAGE1_DIR, STAGE1, MAP5095,
          "Stage 1: stricter IoU sweep", "val mAP@0.5:0.95")
    panel(axes[1][1], STAGE3_DIR, STAGE3, MAP5095,
          "Stage 3: stricter IoU sweep", "val mAP@0.5:0.95")

    fig.suptitle("YOLOv5s validation mAP per epoch (200 epochs, AdamW, cosine LR)",
                 fontsize=11.5, y=0.985)
    fig.text(0.5, 0.005,
             "Best-epoch validation mAP, not a test score. v4-family validation "
             "shares capture bursts with training (99-100% near-duplicate), so "
             "those curves read as in-session.",
             ha="center", fontsize=7.5, color="#444444")
    fig.tight_layout(rect=(0, 0.022, 1, 0.965))

    for out in OUT:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
