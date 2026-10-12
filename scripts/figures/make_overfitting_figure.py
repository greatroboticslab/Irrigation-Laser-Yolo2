"""Train-versus-validation classification loss, the overfitting diagnostic.

Validation mAP alone cannot separate "converged" from "memorised": both look
like a high number. The separation is in the loss trajectories. A run that is
overfitting drives training loss toward zero while validation loss turns and
climbs; a run that is merely converged holds both flat, and a run that is still
improving has validation loss falling at the last epoch.

Reads the same `results.csv` files as scripts/figures/make_training_curves_figure.py and plots
`train/cls_loss` against `val/cls_loss`. Classification loss is the informative
channel here because the task is reading a moisture level off a probe -- box and
objectness losses measure only whether the probe was located.

Two cautions are built into how this is read:

  * Training loss is logged on augmented batches, validation loss on clean
    images, so the two curves are not on a common scale and their vertical gap
    is not a generalization gap. Only the *shape* of each curve is evidence.
  * Where validation frames share a capture burst with training, validation loss
    is measured on near-duplicates of the training data and cannot detect
    memorisation at all. That applies to the v4 family; see the burst section.

Usage (from the project root):
  python scripts/figures/make_overfitting_figure.py
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
PAPER_OUT = [ROOT / "results" / "ieee" / "IEEE_LaTeX" / "figs" / "fig9_overfitting.png",
             ROOT / "results" / "ieee" / "IEEE_LaTeX" / "figs" / "fig9_overfitting.pdf"]
REVIEW_OUT = [UPDATED / "overfitting_curves.png"]

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

STAGE1 = ["soil-moisture-ir", "soil-moisture-5sagf", "soil-moisture-v4",
          "soil-moisture-v4-uv", "soil-moisture-v4-ir", "soil_moisture_september",
          "soil_moisture_stir_september"]
STAGE3 = ["soil-moisture-ir-combined", "soil-moisture-v4-ir",
          "soil-moisture-v4-september-combined", "soil-moisture-v4-uv",
          "soil_moisture_stir_september"]

COLORS = {
    "soil-moisture-ir": "#1b5e9c",
    "soil-moisture-5sagf": "#17a2b8",
    "soil-moisture-v4": "#c0392b",
    "soil-moisture-v4-uv": "#8e44ad",
    "soil-moisture-v4-ir": "#7f8c8d",
    "soil_moisture_september": "#2e8b57",
    "soil_moisture_stir_september": "#d68910",
    "soil-moisture-ir-combined": "#1b5e9c",
    "soil-moisture-v4-september-combined": "#2e8b57",
}

# The one run family that turns. Drawn heavier so the exception is visible
# against six curves that do not move.
EMPHASIS = "soil_moisture_stir_september"

# Epochs before this are ignored when locating the validation minimum: the
# detector predicts almost nothing in the first few dozen epochs, which gives a
# spuriously low loss that is an artefact of silence, not of fit.
WARMUP = 50


def read_curve(path):
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


# The figure is drawn at its printed size (IEEE single column, 3.5 in) so that
# font sizes here are the sizes on the page. An earlier 11 in version was scaled
# to a third by \includegraphics[width=\columnwidth] and printed at about 2 pt.
FIG_W, FIG_H = 3.5, 4.1
FS = 7      # titles and axis labels, pt
FS_TICK = 6.5
FS_LEG = 6


def panel(ax, base_dir, datasets, metric, title, ylabel, mark_min):
    for ds in datasets:
        path = base_dir / ds / "results.csv"
        if not path.is_file():
            print(f"  MISSING {path}")
            continue
        cols = read_curve(path)
        if metric not in cols:
            continue
        epochs, ys = cols["epoch"], cols[metric]
        emph = ds == EMPHASIS
        ax.plot(epochs, ys, color=COLORS.get(ds, "#333333"),
                lw=1.4 if emph else 0.6, alpha=1.0 if emph else 0.85,
                zorder=4 if emph else 2, label=NAME_MAP.get(ds, ds))
        if mark_min and len(ys) > WARMUP:
            tail = ys[WARMUP:]
            mn = min(tail)
            mn_ep = ys.index(mn)
            ax.plot([mn_ep], [mn], "v", color=COLORS.get(ds, "#333333"),
                    ms=3.5 if emph else 2.5, mec="white", mew=0.4, zorder=5)

    if title:
        ax.set_title(title, fontsize=FS, pad=3, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS, labelpad=1)
    ax.set_yscale("log")
    ax.grid(alpha=0.25, lw=0.4, which="both")
    ax.tick_params(labelsize=FS_TICK, pad=1.5, length=2)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)


def main():
    plt.rcParams.update({"font.family": "serif"})
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, FIG_H), sharex=True)

    panel(axes[0][0], STAGE1_DIR, STAGE1, "val/cls_loss",
          "Stage 1", "Validation cls loss", True)
    panel(axes[0][1], STAGE3_DIR, STAGE3, "val/cls_loss",
          "Stage 3", None, True)
    panel(axes[1][0], STAGE1_DIR, STAGE1, "train/cls_loss",
          None, "Training cls loss", False)
    panel(axes[1][1], STAGE3_DIR, STAGE3, "train/cls_loss",
          None, None, False)
    for ax in axes[1]:
        ax.set_xlabel("Epoch", fontsize=FS, labelpad=1)
        ax.set_xticks([0, 50, 100, 150, 200])

    # The first few dozen epochs throw validation-loss spikes two to four orders
    # of magnitude above the settled range -- the detector is still predicting
    # almost nothing. Clipping keeps the readable band readable; the spikes are
    # outside the window the diagnostic uses anyway.
    axes[0][0].set_ylim(0.008, 0.5)
    axes[0][1].set_ylim(0.0015, 0.30)

    # One legend per column, under the panels, so no curve is hidden.
    fig.tight_layout(rect=(0, 0.2, 1, 1), h_pad=0.4, w_pad=0.6)
    for col, x in ((0, axes[1][0]), (1, axes[1][1])):
        h, l = x.get_legend_handles_labels()
        box = x.get_position()
        fig.legend(h, l, loc="upper left", bbox_to_anchor=(box.x0 - 0.06, 0.19),
                   fontsize=FS_LEG, frameon=False, handlelength=1.4,
                   handletextpad=0.4, labelspacing=0.25, borderaxespad=0)

    for out in PAPER_OUT + REVIEW_OUT:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
