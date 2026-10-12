"""Build laser-removed twins of the datasets behind the 'results' runs.

For each dataset under data/training-data/downloaded/, every image in every
split is copied twice-over: the laser spot is located (laser_ablation_dataset
.laser_mask) and inpainted from surrounding pixels, labels are copied verbatim,
and a data.yaml is emitted pointing at the new images. The result is a dataset
identical to the original except that the laser reflection is gone, so a model
retrained on it answers "does the laser contribute anything?".

Output: <out_root>/<dataset>/{train,valid,test}/{images,labels} + data.yaml,
plus qc/<dataset>.csv (per-image mask size / detection status) and
qc/preview_<dataset>.jpg (before/after strips for eyeballing).

Usage:
  python scripts/data_prep/build_nolaser_datasets.py --src "results/data/training-data/downloaded" \
      --out "results/data/training-data/nolaser"
"""

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from laser_ablation_dataset import COLOR_PRESETS, DILATE_KERNEL, INPAINT_RADIUS, laser_mask

# Laser color per dataset. Everything in this project is a violet/UV diode
# except the STIR September capture, which used a red pointer.
DATASET_COLOR = {
    "soil_moisture_stir_september": "red",
}
DEFAULT_COLOR = "violet"

SPLITS = ("train", "valid", "test")


def sham_mask(mask, bbox, shape):
    """Translate the laser mask somewhere else in the frame, without overlap.

    The sham condition is the control for this whole experiment: it applies an
    inpainting scar of the same size and shape as the real one, but over
    ordinary scene content, leaving the laser untouched. If accuracy falls as
    far under sham as it does under true removal, the fall is a reaction to
    the inpainting artifact rather than evidence that the laser carried
    information.
    """
    H, W = shape
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    if bw >= W or bh >= H:
        return None
    # Mirror the spot's position; fall back to the largest-gap direction.
    nx0, ny0 = W - x1 - 1, H - y1 - 1
    if abs(nx0 - x0) < bw and abs(ny0 - y0) < bh:
        nx0 = 0 if x0 > W // 2 else W - bw
        ny0 = 0 if y0 > H // 2 else H - bh
    nx0 = int(np.clip(nx0, 0, W - bw))
    ny0 = int(np.clip(ny0, 0, H - bh))
    if abs(nx0 - x0) < bw // 2 and abs(ny0 - y0) < bh // 2:
        return None  # cannot place it clear of the real spot

    patch = mask[y0:y1 + 1, x0:x1 + 1]
    out = np.zeros((H, W), np.uint8)
    out[ny0:ny0 + bh, nx0:nx0 + bw] = patch
    out[mask > 0] = 0  # never touch the real laser
    return out


def process_dataset(src, out, name, preset, preview_n=6, sham=False, splits=SPLITS):
    rows, previews = [], []
    n_ok = n_miss = 0

    for split in splits:
        img_dir = src / split / "images"
        lbl_dir = src / split / "labels"
        if not img_dir.is_dir():
            continue
        out_img = out / split / "images"
        out_lbl = out / split / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                rows.append([split, img_path.name, 0, "UNREADABLE"])
                continue

            mask, core_area, bbox = laser_mask(img, preset)
            if mask is None:
                # No laser found: copy through untouched so the split keeps the
                # same images in both conditions. Counted in the QC csv.
                removed = img
                n_miss += 1
                rows.append([split, img_path.name, 0, "NO_LASER_FOUND"])
            else:
                if sham:
                    shifted = sham_mask(mask, bbox, img.shape[:2])
                    if shifted is None:
                        removed = img
                        n_miss += 1
                        rows.append([split, img_path.name, 0, "SHAM_NO_ROOM"])
                        cv2.imwrite(str(out_img / img_path.name), removed)
                        lbl = lbl_dir / (img_path.stem + ".txt")
                        if lbl.is_file():
                            shutil.copy(lbl, out_lbl / lbl.name)
                        continue
                    mask = shifted
                removed = cv2.inpaint(img, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
                n_ok += 1
                rows.append([split, img_path.name, int((mask > 0).sum()), "ok"])
                if len(previews) < preview_n and split == "valid":
                    x0, y0, x1, y1 = bbox
                    cx = min(max((x0 + x1) // 2, 160), img.shape[1] - 160)
                    cy = min(max((y0 + y1) // 2, 160), img.shape[0] - 160)
                    crop = (slice(cy - 160, cy + 160), slice(cx - 160, cx + 160))
                    previews.append(np.hstack([img[crop], removed[crop]]))

            cv2.imwrite(str(out_img / img_path.name), removed)
            lbl = lbl_dir / (img_path.stem + ".txt")
            if lbl.is_file():
                shutil.copy(lbl, out_lbl / lbl.name)

    return n_ok, n_miss, rows, previews


def write_yaml(src, out, splits_present):
    """Copy nc/names from the source data.yaml, repoint the split paths."""
    src_yaml = src / "data.yaml"
    nc = names = None
    if src_yaml.is_file():
        for line in src_yaml.read_text().splitlines():
            if line.startswith("nc:"):
                nc = line
            # Only an inline list counts here. Roboflow writes the classes as a
            # block list, whose header line is a bare "names:" -- accepting that
            # as the value silently produced datasets with no class names.
            elif line.startswith("names:") and line.split(":", 1)[1].strip():
                names = line
    if names is None:  # roboflow block-style names list
        import re
        text = src_yaml.read_text() if src_yaml.is_file() else ""
        block = re.search(r"names:\n((?:- .*\n)+)", text)
        if block:
            items = [l[2:].strip() for l in block.group(1).splitlines()]
            names = "names: [" + ", ".join(items) + "]"
    if not (nc and names):
        print(f"  warning: could not read nc/names from {src_yaml}")
        return
    lines = [f"{'val' if s == 'valid' else s}: {(out / s / 'images').as_posix()}"
             for s in splits_present]
    (out / "data.yaml").write_text("\n".join(lines) + f"\n\n{nc}\n{names}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, help="dir holding the source datasets")
    ap.add_argument("--out", required=True, help="dir to write laser-removed twins")
    ap.add_argument("--only", nargs="*", help="limit to these dataset names")
    ap.add_argument("--sham", action="store_true",
                    help="control condition: inpaint an equally sized scar "
                         "elsewhere and leave the laser intact")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    args = ap.parse_args()

    src_root, out_root = Path(args.src), Path(args.out)
    qc_dir = out_root / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    for src in sorted(p for p in src_root.iterdir() if p.is_dir()):
        name = src.name
        if args.only and name not in args.only:
            continue
        color = DATASET_COLOR.get(name, DEFAULT_COLOR)
        preset = COLOR_PRESETS[color]
        out = out_root / name
        print(f"{name} (laser={color})", flush=True)

        n_ok, n_miss, rows, previews = process_dataset(
            src, out, name, preset, sham=args.sham, splits=args.splits)
        splits_present = [s for s in args.splits if (out / s / "images").is_dir()]
        write_yaml(src, out, splits_present)

        with open(qc_dir / f"{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["split", "image", "mask_px", "status"])
            w.writerows(rows)
        if previews:
            cv2.imwrite(str(qc_dir / f"preview_{name}.jpg"), np.vstack(previews))
        total = n_ok + n_miss
        pct = 100.0 * n_ok / total if total else 0.0
        print(f"  {n_ok}/{total} images had a laser removed ({pct:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
