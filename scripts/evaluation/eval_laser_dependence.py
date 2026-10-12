"""Evaluate each trained model on its own valid set, with and without the laser.

Every model in 'results' was trained on images that contain a laser
reflection. This re-runs validation on the identical images after the laser
has been inpainted away (scripts/data_prep/build_nolaser_datasets.py), so the drop in mAP is a
direct measure of how much the trained detector actually relies on the laser
return. No retraining is involved: same weights, same labels, same images,
one variable changed.

Writes laser_dependence.csv (dataset, run, mAP50 with laser, mAP50 without,
delta) and prints the table.

Usage:
  python scripts/evaluation/eval_laser_dependence.py --results-dir "results"
"""

import argparse
import csv
import functools
import sys
from pathlib import Path

import torch
# val.py loads full checkpoints; torch >= 2.6 defaults weights_only=True.
torch.load = functools.partial(torch.load, weights_only=False)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# dataset name -> list of (run label, path to run dir holding weights/best.pt)
RUN_GROUPS = {
    "first-run": "runs/train/downloaded",
    "merged": "runs/train/merged",
    "fine-tuned": "runs/fine-tuned",
}

# Datasets whose laser cannot be segmented reliably; see README note. Their
# laser return is too diffuse/low-saturation to separate from the soil, so a
# "removed" image is not trustworthy and the comparison would be meaningless.
EXCLUDED = {"soil-moisture-v4-ir"}


def read_nc_names(yaml_path):
    import re
    text = yaml_path.read_text()
    nc = re.search(r"^nc:\s*(\d+)", text, re.M)
    names = re.search(r"^names:\s*(\[.*\])", text, re.M)
    if not names:
        block = re.search(r"names:\n((?:- .*\n)+)", text)
        if block:
            items = [l[2:].strip() for l in block.group(1).splitlines()]
            names = "[" + ", ".join(items) + "]"
        else:
            return None, None
    else:
        names = names.group(1)
    return (nc.group(1) if nc else None), names


def make_yaml(dst, images_dir, nc, names):
    dst.write_text(f"train: {images_dir}\nval: {images_dir}\nnc: {nc}\nnames: {names}\n")
    return dst


def run_val(weights, data_yaml, device, workdir):
    import val as validate

    results, _, _ = validate.run(
        str(data_yaml), weights=str(weights), batch_size=8, imgsz=640,
        device=device, task="val", workers=0, verbose=False, half=False,
        plots=False, project=workdir, name="val", exist_ok=True)
    # results = (mp, mr, map50, map, *losses)
    return float(results[2])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/laser_dependence.csv")
    args = ap.parse_args()

    res = ROOT / args.results_dir
    src_root = res / "data/training-data/downloaded"
    nol_root = res / "data/training-data/nolaser"
    sham_root = res / "data/training-data/sham"
    tmp = ROOT / ".laser_eval_tmp"
    tmp.mkdir(exist_ok=True)

    rows = []
    for label, group in RUN_GROUPS.items():
        gdir = res / group
        if not gdir.is_dir():
            continue
        for run in sorted(gdir.iterdir()):
            weights = run / "weights" / "best.pt"
            name = run.name
            if not weights.is_file():
                continue
            # stir-finetune run dirs carry a numeric suffix
            ds = name.rstrip("0123456789") if name not in (d.name for d in src_root.iterdir()) else name
            src = src_root / ds
            nol = nol_root / ds
            if ds in EXCLUDED or not src.is_dir() or not nol.is_dir():
                print(f"skip {label}/{name} (excluded or missing dataset)")
                continue

            nc, names = read_nc_names(src / "data.yaml")
            if not nc:
                print(f"skip {label}/{name} (unreadable data.yaml)")
                continue

            y_with = make_yaml(tmp / f"{label}_{ds}_with.yaml",
                               (src / "valid/images").resolve(), nc, names)
            y_without = make_yaml(tmp / f"{label}_{ds}_without.yaml",
                                  (nol / "valid/images").resolve(), nc, names)
            sham_dir = sham_root / ds / "valid/images"
            y_sham = (make_yaml(tmp / f"{label}_{ds}_sham.yaml",
                                sham_dir.resolve(), nc, names)
                      if sham_dir.is_dir() else None)
            try:
                m_with = run_val(weights, y_with, args.device, tmp)
                m_without = run_val(weights, y_without, args.device, tmp)
                m_sham = run_val(weights, y_sham, args.device, tmp) if y_sham else float("nan")
            except Exception as e:  # noqa: BLE001 - report and continue
                print(f"FAILED {label}/{name}: {e}")
                continue
            rows.append([ds, label, round(m_with, 4), round(m_without, 4),
                         round(m_sham, 4), round(m_without - m_with, 4),
                         round(m_sham - m_with, 4)])
            print(f"{ds:<38} {label:<11} with={m_with:.4f} "
                  f"nolaser={m_without:.4f} sham={m_sham:.4f} "
                  f"d_laser={m_without - m_with:+.4f} d_sham={m_sham - m_with:+.4f}",
                  flush=True)

    out = ROOT / args.out
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "run", "mAP50_with_laser", "mAP50_laser_removed",
                    "mAP50_sham_control", "delta_laser_removed", "delta_sham"])
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
