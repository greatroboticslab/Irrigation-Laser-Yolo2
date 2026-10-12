"""Score each trained model with and without the burned-in moisture overlay.

The capture software prints the ground-truth value into a corner of every
frame. If a detector learned to read those digits, its mAP measures OCR, not
soil sensing. This scores every model twice on its own validation split --
once as captured, once with the overlay blanked (build_overlay_masked.py) --
so the gap between them bounds how much of the score is label leakage.

Datasets where no burned-in overlay exists are skipped rather than reported
as a zero change, since "no difference" there would only mean "nothing was
masked". Note that soil-moisture-5sagf / -ir / -ir-combined instead carry a
handwritten card in the physical scene; blanking a corner does not address
that, and those datasets need a separate card-masking pass.

Usage:
  python eval_overlay_leak.py
"""

import argparse
import csv
import functools
import sys
from pathlib import Path

import torch
torch.load = functools.partial(torch.load, weights_only=False)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

RUN_GROUPS = {
    "first-run": "runs/train/downloaded",
    "merged": "runs/train/merged",
    "fine-tuned": "runs/fine-tuned",
}


def run_val(weights, data_yaml, device, workdir):
    import val as validate
    results, _, _ = validate.run(
        str(data_yaml), weights=str(weights), batch_size=8, imgsz=640,
        device=device, task="val", workers=0, verbose=False, half=False,
        plots=False, project=workdir, name="val", exist_ok=True)
    return float(results[2])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", default="updated results")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="overlay_leak.csv")
    args = ap.parse_args()

    res = ROOT / args.results_dir
    src_root = res / "data/training-data/downloaded"
    masked_root = res / "data/training-data/overlay_masked"
    tmp = ROOT / ".overlay_eval_tmp"
    tmp.mkdir(exist_ok=True)

    dataset_names = {d.name for d in src_root.iterdir() if d.is_dir()}
    rows = []
    for label, group in RUN_GROUPS.items():
        gdir = res / group
        if not gdir.is_dir():
            continue
        for run in sorted(gdir.iterdir()):
            weights = run / "weights" / "best.pt"
            if not weights.is_file():
                continue
            name = run.name
            ds = name if name in dataset_names else name.rstrip("0123456789")
            masked_yaml = masked_root / ds / "data.yaml"
            src = src_root / ds
            if not masked_yaml.is_file() or not src.is_dir():
                print(f"skip {label}/{name}: no masked set (no burned-in overlay)")
                continue

            text = masked_yaml.read_text()
            nc = [l for l in text.splitlines() if l.startswith("nc:")][0].split(":")[1].strip()
            names = [l for l in text.splitlines() if l.startswith("names:")][0].split(":", 1)[1].strip()
            y_orig = tmp / f"{label}_{ds}_orig.yaml"
            imgs = (src / "valid/images").resolve()
            y_orig.write_text(f"train: {imgs}\nval: {imgs}\nnc: {nc}\nnames: {names}\n")

            try:
                m_orig = run_val(weights, y_orig, args.device, tmp)
                m_masked = run_val(weights, masked_yaml, args.device, tmp)
            except Exception as e:  # noqa: BLE001
                print(f"FAILED {label}/{name}: {e}")
                continue
            rows.append([ds, label, round(m_orig, 4), round(m_masked, 4),
                         round(m_masked - m_orig, 4)])
            print(f"{ds:<38} {label:<11} original={m_orig:.4f} "
                  f"overlay_masked={m_masked:.4f} delta={m_masked - m_orig:+.4f}",
                  flush=True)

    out = ROOT / args.out
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "run", "mAP50_original", "mAP50_overlay_masked", "delta"])
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
