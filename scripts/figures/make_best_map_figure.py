"""Publication bar charts: best validation mAP@0.5 per run, one figure per stage.

Reads `results/runs/<stage dir>/<dataset>/results.csv` and writes two paper
figures (the .png names are the ones the Overleaf manuscript includes):

  * fig:baseline  -- the seven Stage 1 (COCO-initialised) runs, figs/_fig4_view.png
  * fig:best      -- the five Stage 3 (fine-tuned) runs,        figs/_fig5_view.png

Bar heights are the max-mAP@0.5 epoch, the same "mAP peak" figure as Tables
IV-VI. Both are drawn at their printed size (IEEE single column, 3.5 in), so the
font sizes here are the sizes on the page. The dashed 0.5 line is explained in
the LaTeX caption rather than a legend, which collided with the bars.

Display names come from the per-figure name maps, the one place to edit if the
paper's dataset names change. Keys are the on-disk dataset directory names.

Usage (from the project root):
  python scripts/figures/make_best_map_figure.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "runs"
FIGS = ROOT / "results" / "ieee" / "IEEE_LaTeX" / "figs"

# name map in plotting order (matches Table IV), bar colours, output stem.
FIGURES = [
    {
        "runs": RUNS / "train" / "downloaded",
        "names": {
            "soil-moisture-ir": "IR-Pot-Background",
            "soil-moisture-5sagf": "UV-Pot-Background",
            "soil-moisture-v4": "UV-Soil-Only-Baseline",
            "soil-moisture-v4-uv": "UV-Soil-Only-Target",
            "soil-moisture-v4-ir": "IR-Soil-Only",
            "soil_moisture_september": "UV-Strawberry-Pot",
            "soil_moisture_stir_september": "Mixed Color and\nBackground",
        },
        "colors": ["#2f5f9e", "#17a2b8", "#c0392b", "#6a2c8a", "#7f8c8d", "#2e7d32", "#d4862a"],
        "out": "_fig4_view",
        "label_fs": 6.5,
        "rotate": 35,  # seven one-line names do not fit side by side at 3.5 in
    },
    {
        "runs": RUNS / "fine-tuned",
        "names": {
            "soil-moisture-ir-combined": "IR-\nCombined",
            "soil-moisture-v4-ir": "IR-Soil-\nOnly",
            "soil-moisture-v4-september-combined": "UV-\nCombined",
            "soil-moisture-v4-uv": "UV-Soil-\nOnly-\nTarget",
            "soil_moisture_stir_september": "Mixed\nColor and\nBackground",
        },
        "colors": ["#2f5f9e", "#a8262d", "#2e7d32", "#6a2c8a", "#d4862a"],
        "out": "_fig5_view",
        "label_fs": 6.5,
    },
]


def peak_map50(run_dir):
    df = pd.read_csv(run_dir / "results.csv")
    df.columns = df.columns.str.strip()
    return df["metrics/mAP_0.5"].max()


def draw(spec):
    names = list(spec["names"])
    values = [peak_map50(spec["runs"] / n) for n in names]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    bars = ax.bar(range(len(names)), values, color=spec["colors"], edgecolor="black",
                  linewidth=0.6, width=0.7)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.4f}",
                ha="center", va="bottom", fontsize=7 if len(names) > 5 else 7.5,
                fontweight="bold")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(len(names)))
    rot = spec.get("rotate", 0)
    ax.set_xticklabels([spec["names"][n] for n in names], fontsize=spec["label_fs"],
                       rotation=rot, ha="right" if rot else "center",
                       rotation_mode="anchor" if rot else "default")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Best mAP@0.5", fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        out = FIGS / (spec["out"] + ext)
        fig.savefig(out, dpi=300)
        print("wrote", out.relative_to(ROOT))
    plt.close(fig)
    for n, v in zip(names, values):
        print(f"  {spec['names'][n].replace(chr(10), ' '):32s} {v:.4f}")


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    for spec in FIGURES:
        draw(spec)


if __name__ == "__main__":
    main()
