"""Build laser-ablation datasets for the "is the laser useful?" experiment.

For every image in a YOLO-layout dataset (train/valid/test with images/ and
labels/), this script:

  1. Segments the laser spot (bright core seeded, halo grown only while
     connected to the core, clamped near the core bounding box).
  2. Writes a "laser-removed" copy where the spot is inpainted from
     surrounding soil pixels (cv2.INPAINT_TELEA).
  3. Writes a laser-only patch crop and a matched laser-removed patch crop
     (same location) for the patch-level classification conditions.
  4. Optionally blacks out the top strip in ALL outputs to hide the
     "Moisture: N" text overlay burned into the frames (--mask-overlay).
     Without this the whole-image comparison is invalid: a model can read
     the ground-truth label straight from the pixels.
  5. Copies label files unchanged and writes a per-image QC CSV
     (mask area, core area, bbox) so failures are easy to audit.

Output layout (under --out):
  full_laser/<split>/images,labels      original frames (overlay masked)
  full_nolaser/<split>/images,labels    inpainted frames (overlay masked)
  patch_laser/<split>/<class>/*.jpg     patch crops, laser present
  patch_nolaser/<split>/<class>/*.jpg   same patches, laser inpainted
  qc/<split>.csv, qc/preview_<split>.jpg

The four resulting datasets give a 2x2 ablation:
  full image  x {laser, no laser}   -> does the laser add anything at scene level?
  patch only  x {laser, no laser}   -> does the spot itself carry moisture info?

Usage:
  python scripts/data_prep/laser_ablation_dataset.py --data data/soil_september --out data/ablation_september
  python scripts/data_prep/laser_ablation_dataset.py --data <green laser dataset> --color green ...
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

# Hue ranges in OpenCV scale (0-179).
#
# min_s is the load-bearing threshold. A laser spot is a saturated light
# source; tinted scene content is not. Measured over these datasets, true
# spots sit at S 170-250 while the bluish soil of soil-moisture-v4-ir tops
# out near S 108 -- which is why that dataset yields no detections rather
# than false ones (its laser return is too diffuse to segment by color).
COLOR_PRESETS = {
    "violet": {"hue": (100, 170), "min_v": 120, "min_s": 130},
    "green": {"hue": (35, 85), "min_v": 120, "min_s": 130},
    # Red wraps around the end of the OpenCV hue circle, so it needs two ranges.
    "red": {"hue": [(0, 10), (168, 179)], "min_v": 100, "min_s": 130, "channel": 2},
}
# Channel that a spot of this color makes dominant in BGR (used for the halo).
COLOR_PRESETS["violet"]["channel"] = 0
COLOR_PRESETS["green"]["channel"] = 1

DILATE_KERNEL = 11        # final mask dilation to cover the dim fringe
SCORE_SIGMA = 4           # Gaussian sigma smoothing the laser-likeness map
REL_THRESH = 0.25         # grow while score >= this fraction of the peak
MAX_SPOT_RADIUS = 150     # hard cap on how far a mask may reach from the peak
WHITE_REACH = 15          # px within which an overexposed core joins the spot
MIN_PEAK_SCORE = 0.12     # below this peak score, the frame has no laser
INPAINT_RADIUS = 7
PATCH_SIZE = 160          # patch crop side, centered on the laser core
OVERLAY_ROWS = 55         # top rows blacked out by --mask-overlay


def laser_mask(img, preset, min_peak=None):
    """Return (mask, spot_area, spot_bbox) or (None, 0, None) if no laser found.

    Scale-adaptive by design. Rather than fixed thresholds (which need
    retuning per capture setup and silently mask the wrong thing when they
    drift), this scores every pixel on laser-likeness -- in the laser's hue,
    saturated, and bright -- smooths that score, takes its peak, and grows
    the mask outward while the score stays within a fraction of that peak.
    The peak is the anchor, so a small distant spot and a large close one are
    both captured, and a frame with no laser simply has no peak above
    `min_peak` and is reported as a miss instead of masking scenery.
    """
    if min_peak is None:
        min_peak = preset.get("min_peak", MIN_PEAK_SCORE)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    ranges = preset["hue"]
    if isinstance(ranges, tuple):
        ranges = [ranges]
    in_hue = np.zeros(h.shape, bool)
    for lo, hi in ranges:
        in_hue |= (h >= lo) & (h <= hi)

    score = (s.astype(np.float32) / 255.0) * (v.astype(np.float32) / 255.0)
    score[~in_hue] = 0.0
    smooth = cv2.GaussianBlur(score, (0, 0), SCORE_SIGMA)

    peak_val = float(smooth.max())
    if peak_val < min_peak:
        return None, 0, None
    py, px = np.unravel_index(int(np.argmax(smooth)), smooth.shape)

    # Grow from the peak while the smoothed score stays within REL_THRESH of
    # it, but never beyond MAX_SPOT_RADIUS -- a bound that keeps a bad frame
    # from turning into a whole-image mask.
    H, W = smooth.shape
    region = (smooth >= REL_THRESH * peak_val).astype(np.uint8)
    disk = np.zeros((H, W), np.uint8)
    cv2.circle(disk, (int(px), int(py)), MAX_SPOT_RADIUS, 255, -1)
    region = cv2.bitwise_and(region * 255, disk)

    nc, labels, stats, _ = cv2.connectedComponentsWithStats(region)
    if nc <= 1:
        return None, 0, None
    lab = labels[py, px]
    if lab == 0:
        return None, 0, None
    spot = (labels == lab).astype(np.uint8) * 255

    # The overexposed centre of a bright spot loses its hue (white, low S), so
    # it scores zero above and can survive as a bright blob in the middle of
    # an otherwise cleaned spot. Pull in whole white components that touch the
    # grown region, rather than a fixed reach, so the entire core comes out
    # however large it is.
    near = cv2.dilate(spot, np.ones((WHITE_REACH, WHITE_REACH), np.uint8))
    white = ((v >= 235) & (s <= 90)).astype(np.uint8) * 255
    wn, wlabels, _, _ = cv2.connectedComponentsWithStats(white)
    touching = np.unique(wlabels[near > 0])
    for wl in touching:
        if wl != 0:
            spot |= (wlabels == wl).astype(np.uint8) * 255

    mask = cv2.dilate(spot, np.ones((DILATE_KERNEL, DILATE_KERNEL), np.uint8))
    mask = cv2.bitwise_and(mask, disk)

    ys, xs = np.where(mask > 0)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return mask, int((mask > 0).sum()), bbox


def crop_patch(img, bbox):
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    H, W = img.shape[:2]
    half = PATCH_SIZE // 2
    cx = min(max(cx, half), W - half)
    cy = min(max(cy, half), H - half)
    return img[cy - half:cy + half, cx - half:cx + half]


def read_class(label_path):
    """First class id in the YOLO label file, or None."""
    try:
        line = label_path.read_text().strip().splitlines()[0]
        return line.split()[0]
    except (OSError, IndexError):
        return None


def process_split(data_dir, out_dir, split, preset, mask_overlay, preview_n=8):
    img_dir = data_dir / split / "images"
    lbl_dir = data_dir / split / "labels"
    if not img_dir.is_dir():
        return None

    dirs = {}
    for cond in ("full_laser", "full_nolaser"):
        for sub in ("images", "labels"):
            d = out_dir / cond / split / sub
            d.mkdir(parents=True, exist_ok=True)
            dirs[(cond, sub)] = d

    qc_dir = out_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    rows, previews = [], []
    n_ok = n_miss = 0

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        mask, core_area, bbox = laser_mask(img, preset)
        if mask is None:
            n_miss += 1
            rows.append([img_path.name, 0, 0, "", "NO_LASER_FOUND"])
            continue
        n_ok += 1

        removed = cv2.inpaint(img, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)

        patch_l = crop_patch(img, bbox)
        patch_n = crop_patch(removed, bbox)

        if mask_overlay:
            img = img.copy()
            img[:OVERLAY_ROWS, :] = 0
            removed[:OVERLAY_ROWS, :] = 0

        cv2.imwrite(str(dirs[("full_laser", "images")] / img_path.name), img)
        cv2.imwrite(str(dirs[("full_nolaser", "images")] / img_path.name), removed)

        lbl = lbl_dir / (img_path.stem + ".txt")
        cls = read_class(lbl) or "unknown"
        if lbl.is_file():
            for cond in ("full_laser", "full_nolaser"):
                (dirs[(cond, "labels")] / lbl.name).write_text(lbl.read_text())

        for cond, patch in (("patch_laser", patch_l), ("patch_nolaser", patch_n)):
            d = out_dir / cond / split / cls
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / img_path.name), patch)

        rows.append([img_path.name, int((mask > 0).sum()), core_area, cls, "ok"])
        if len(previews) < preview_n:
            previews.append(np.hstack([patch_l, patch_n]))

    with open(qc_dir / f"{split}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "mask_px", "core_px", "class", "status"])
        w.writerows(rows)
    if previews:
        cv2.imwrite(str(qc_dir / f"preview_{split}.jpg"), np.vstack(previews))
    return n_ok, n_miss


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="YOLO dataset root (train/valid/test)")
    ap.add_argument("--out", required=True, help="output root for ablation datasets")
    ap.add_argument("--color", choices=COLOR_PRESETS, default="violet")
    ap.add_argument("--mask-overlay", action="store_true",
                    help="black out the top strip hiding the Moisture text overlay")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data), Path(args.out)
    preset = COLOR_PRESETS[args.color]
    for split in args.splits:
        res = process_split(data_dir, out_dir, split, preset, args.mask_overlay)
        if res is None:
            print(f"{split}: skipped (no images dir)")
        else:
            print(f"{split}: {res[0]} processed, {res[1]} without detectable laser")

    write_detection_yamls(data_dir, out_dir, args.splits)


def write_detection_yamls(data_dir, out_dir, splits):
    """Emit data.yaml for the two full-image conditions, reusing the source
    dataset's nc/names. Split paths follow the repo convention of being
    relative to the project root (the cwd during training)."""
    src_yaml = data_dir / "data.yaml"
    nc_line = names_line = None
    if src_yaml.is_file():
        for line in src_yaml.read_text().splitlines():
            if line.startswith("nc:"):
                nc_line = line
            elif line.startswith("names:"):
                names_line = line
    if not (nc_line and names_line):
        print("warning: source data.yaml missing nc/names; skipping yaml emission")
        return
    for cond in ("full_laser", "full_nolaser"):
        lines = [f"{s if s != 'valid' else 'val'}: ./{out_dir}/{cond}/{s}/images"
                 for s in splits]
        (out_dir / cond / "data.yaml").write_text(
            "\n".join(lines) + f"\n\n{nc_line}\n{names_line}\n")
        print(f"wrote {out_dir / cond / 'data.yaml'}")


if __name__ == "__main__":
    main()
