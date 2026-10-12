"""Publication figure for the three-arm laser retrain ablation.

Reads `updated results/threearm_summary.csv` (dataset, arm, seed, best_val_mAP50)
and writes a single-column IEEE figure.

Display names come from NAME_MAP, which is the one place to edit if the paper's
dataset names change. Keys are the on-disk dataset directory names.
"""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "updated results" / "threearm_summary.csv"
PAPER_OUT = [
    ROOT / "updated results" / "ieee" / "IEEE_LaTeX" / "figs" / "fig8_laser_ablation.png",
    ROOT / "updated results" / "ieee" / "IEEE_LaTeX" / "figs" / "fig8_laser_ablation.pdf",
]
# Wide two-panel version kept for slides and internal review.
REVIEW_OUT = [ROOT / "updated results" / "threearm_retrain_comparison.png"]

# Paper-facing dataset names. Edit here to match the manuscript.
# The two soil-only captures are the same illumination and taxonomy, so they are
# told apart by name only: -Baseline and -Target. (Neither is the Stage 3 transfer
# source; every Stage 3 run started from IR-Pot-Background, soil-moisture-ir.)
NAME_MAP = {
    "soil-moisture-v4-uv": "UV-Soil-Only-\nTarget",
    "soil-moisture-v4": "UV-Soil-Only-\nBaseline",
    "soil-moisture-v4-september-combined": "UV-Combined",
}

# Training-set size per dataset, printed under each group label.
TRAIN_IMAGES = {
    "soil-moisture-v4-uv": 142,
    "soil-moisture-v4": 246,
    "soil-moisture-v4-september-combined": 292,
}

ORDER = ["soil-moisture-v4-uv", "soil-moisture-v4", "soil-moisture-v4-september-combined"]
ARMS = [
    ("laser", "with laser", "#3b6ea5"),
    ("sham", "sham scar (laser kept)", "#9aa7b4"),
    ("nolaser", "laser inpainted out", "#c0442e"),
]


def load(path):
    runs = defaultdict(list)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            runs[(row["dataset"], row["arm"])].append(float(row["best_val_mAP50"]))
    return runs


def review_figure(runs):
    """Two-panel wide version: honest full scale next to a magnified view."""
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(20, 7), dpi=100)
    x = np.arange(len(ORDER))
    width = 0.26
    labels = [f"{NAME_MAP[ds].replace(chr(10), '')}\n({TRAIN_IMAGES[ds]} train imgs)" for ds in ORDER]

    for ax, (lo, hi), title in zip(
        axes, [(0.0, None), (0.85, 1.02)],
        ["Full scale \u2014 the honest view", "Zoomed (0.85\u20131.02) \u2014 differences magnified"],
    ):
        for i, (arm, label, color) in enumerate(ARMS):
            means = [mean(runs[(ds, arm)]) for ds in ORDER]
            sds = [stdev(runs[(ds, arm)]) for ds in ORDER]
            ax.bar(x + (i - 1) * width, means, width, yerr=sds, label=label,
                   color=color, error_kw={"capsize": 4})
        ax.set_ylim(lo, hi)
        ax.set_title(title)
        ax.set_ylabel("best val mAP@0.5")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
    axes[0].legend(loc="lower right")

    fig.suptitle(
        "Retrained from scratch on each condition (3 seeds, mean \u00b1 s.d.)\n"
        "Only " + NAME_MAP[ORDER[0]] + " \u2014 the smallest dataset, 12.9 images per class \u2014 shows a real cost"
    )
    fig.tight_layout()
    for path in REVIEW_OUT:
        fig.savefig(path, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def main():
    runs = load(CSV)
    review_figure(runs)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8,
        "axes.linewidth": 0.6,
    })

    fig, ax = plt.subplots(figsize=(3.5, 2.6), dpi=400)
    x = np.arange(len(ORDER))
    width = 0.26

    for i, (arm, label, color) in enumerate(ARMS):
        means = [mean(runs[(ds, arm)]) for ds in ORDER]
        sds = [stdev(runs[(ds, arm)]) for ds in ORDER]
        ax.bar(
            x + (i - 1) * width, means, width, yerr=sds, label=label,
            color=color, edgecolor="black", linewidth=0.4,
            error_kw={"elinewidth": 0.7, "capsize": 2, "capthick": 0.7},
        )

    ax.set_ylim(0.85, 1.02)
    ax.set_ylabel("best val mAP@0.5")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{NAME_MAP[ds]}\n({TRAIN_IMAGES[ds]} train imgs)" for ds in ORDER],
        fontsize=7,
    )
    ax.tick_params(axis="both", labelsize=7, width=0.6)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.5, loc="lower left", framealpha=0.95, borderpad=0.4)

    fig.tight_layout(pad=0.3)
    for path in PAPER_OUT:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        print(f"wrote {path}")

    print("\nmean +/- s.d. over 3 seeds")
    for ds in ORDER:
        parts = []
        for arm, label, _ in ARMS:
            v = runs[(ds, arm)]
            parts.append(f"{arm}={mean(v):.4f}+/-{stdev(v):.4f}")
        delta = mean(runs[(ds, "nolaser")]) - mean(runs[(ds, "laser")])
        print(f"  {ds:36} {'  '.join(parts)}  delta={delta:+.4f}")


if __name__ == "__main__":
    main()
