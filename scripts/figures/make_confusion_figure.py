"""Publication confusion matrix for the fine-tuned IR-Combined detector.

YOLOv5 writes a column-normalized matrix, which hides how small the validation
split is: every off-diagonal cell in this run is one or two images. This plots
raw counts instead and prints the per-class support in the axis labels.

Input is the raw (nc+1, nc+1) matrix dumped from utils.metrics.ConfusionMatrix,
indexed [predicted, true] with the last row/column being background.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = ROOT / "data/training-data/downloaded/soil-moisture-ir-combined/data.yaml"
OUT = [
    ROOT / "results/ieee/IEEE_LaTeX/figs/fig6_confusion_matrix.png",
    ROOT / "results/ieee/IEEE_LaTeX/figs/fig6_confusion_matrix.pdf",
]

# Class names are printed without the shared "soil-moisture-" prefix.
PREFIX = "soil-moisture-"

# Counts from the archived cluster run that the paper reports (runs/fine-tuned/
# soil-moisture-ir-combined). Recovered from that run's normalized figure: every
# published cell is a small-integer fraction over a known support, so the counts
# are exact (e.g. 0.86 = 6/7, 0.12 = 1/8, 0.71 = 5/7). Column sums reproduce the
# ground-truth counts 11/7/7/8/7 measured directly from the label files, and the
# background column sums to the 7 false positives implied by precision 0.985.
#
# A CPU re-run under torch 2.8 gives 2 detections differently at the 0.25
# confidence threshold (true 5.0: 6 correct + 2 to class 3.0; FPs 1/4/2). Headline
# metrics are unchanged (mAP@0.5 0.995, recall 1.00). The published run is used
# here so the figure agrees with the numbers quoted in the text.
PUBLISHED = [
    [11, 0, 0, 0, 0, 1],
    [0, 7, 0, 0, 0, 0],
    [0, 0, 6, 1, 0, 5],
    [0, 0, 1, 7, 0, 1],
    [0, 0, 0, 0, 7, 0],
    [0, 0, 0, 0, 0, 0],
]


def main(npy_path=None):
    m = np.load(npy_path) if npy_path else np.array(PUBLISHED, dtype=float)
    names = [n.replace(PREFIX, "") for n in yaml.safe_load(open(DATA_YAML))["names"]]
    labels = names + ["background"]
    support = m.sum(0)  # column totals = ground-truth count per class

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8,
        "axes.linewidth": 0.6,
    })
    fig, ax = plt.subplots(figsize=(3.4, 3.0), dpi=400)

    shown = np.ma.masked_where(m == 0, m)
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=m.max())

    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = int(m[i, j])
            if v == 0:
                continue
            ax.text(j, i, str(v), ha="center", va="center", fontsize=7.5,
                    color="white" if m[i, j] > 0.6 * m.max() else "black")

    xt = [f"{l}\n(n={int(support[k])})" if l != "background" else f"{l}\n(FP)"
          for k, l in enumerate(labels)]
    ax.set_xticks(range(len(labels)), xt, fontsize=6.5)
    ax.set_yticks(range(len(labels)), labels, fontsize=6.5)
    ax.set_xlabel("Ground truth", fontsize=8)
    ax.set_ylabel("Predicted", fontsize=8)
    ax.tick_params(width=0.6, length=2)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("detections", fontsize=7)
    cb.ax.tick_params(labelsize=6.5, width=0.6)

    fig.tight_layout(pad=0.3)
    for p in OUT:
        fig.savefig(p, bbox_inches="tight")
        print(f"wrote {p}")

    diag = np.diag(m)[:len(names)]
    print("\nper-class recall (diagonal / ground-truth count)")
    for k, n in enumerate(names):
        print(f"  {n:8} {int(diag[k]):2d}/{int(support[k]):2d} = {diag[k]/support[k]:.3f}")
    print(f"  total ground-truth objects: {int(support[:len(names)].sum())}")
    print(f"  false positives (background column): {int(support[len(names)])}")
    print(f"  missed detections (background row): {int(m[len(names)].sum())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", help="raw .npy from a fresh val run; default is the published run")
    main(ap.parse_args().matrix)
