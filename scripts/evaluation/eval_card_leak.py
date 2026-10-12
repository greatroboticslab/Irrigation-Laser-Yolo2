"""Score the card-dataset models with and without the in-scene paper labels.

soil-moisture-5sagf / -ir / -ir-combined photograph a sticky note reading
"No. 5 ... Moisture: 8.2" next to the pot, plus a numbered tag on the rim.
Their class names are the moisture values themselves (soil-moisture-8.2), so
the answer is legible in the frame. scripts/data_prep/build_card_masked.py blanks those labels
-- and only those, never the laser -- and this scores each model both ways.

Usage:
  python scripts/evaluation/eval_card_leak.py
"""

import csv
import functools
import sys
from pathlib import Path

import torch
torch.load = functools.partial(torch.load, weights_only=False)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RUNS = [
    ("first-run", "runs/train/downloaded", "soil-moisture-5sagf"),
    ("first-run", "runs/train/downloaded", "soil-moisture-ir"),
    ("merged", "runs/train/merged", "soil-moisture-ir-combined"),
    ("fine-tuned", "runs/fine-tuned", "soil-moisture-ir-combined"),
]


def run_val(weights, data_yaml, tmp):
    import val as validate
    results, _, _ = validate.run(
        str(data_yaml), weights=str(weights), batch_size=8, imgsz=640,
        device="cpu", task="val", workers=0, verbose=False, half=False,
        plots=False, project=tmp, name="v", exist_ok=True)
    return float(results[2])


def main():
    res = ROOT / "results"
    src = res / "data/training-data/downloaded"
    card = res / "data/training-data/card_masked"
    tmp = ROOT / ".card_eval_tmp"
    tmp.mkdir(exist_ok=True)

    rows = []
    for label, group, ds in RUNS:
        weights = res / group / ds / "weights/best.pt"
        card_yaml = card / ds / "data.yaml"
        if not weights.is_file() or not card_yaml.is_file():
            print(f"skip {label}/{ds}: missing weights or masked set")
            continue
        text = card_yaml.read_text()
        nc = [l for l in text.splitlines() if l.startswith("nc:")][0].split(":")[1].strip()
        names = [l for l in text.splitlines() if l.startswith("names:")][0].split(":", 1)[1].strip()
        imgs = (src / ds / "valid/images").resolve()
        orig_yaml = tmp / f"{label}_{ds}_orig.yaml"
        orig_yaml.write_text(f"train: {imgs}\nval: {imgs}\nnc: {nc}\nnames: {names}\n")

        m_orig = run_val(weights, orig_yaml, tmp)
        m_masked = run_val(weights, card_yaml, tmp)
        rows.append([ds, label, round(m_orig, 4), round(m_masked, 4),
                     round(m_masked - m_orig, 4)])
        print(f"{ds:<28} {label:<11} original={m_orig:.4f} "
              f"cards_masked={m_masked:.4f} delta={m_masked - m_orig:+.4f}", flush=True)

    out = ROOT / "results" / "card_leak.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "run", "mAP50_original", "mAP50_cards_masked", "delta"])
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
