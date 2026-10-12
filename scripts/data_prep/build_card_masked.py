"""Blank the handwritten moisture labels that sit inside the photographed scene.

soil-moisture-5sagf / -ir / -ir-combined carry no burned-in overlay. Instead
the moisture value is written on paper and left in frame: a yellow sticky note
("No. 5 ... Moisture: 8.2") beside the pot, and a small white tag clipped to
the pot rim with the level written on it. Both put the ground-truth answer in
the pixels, so a detector can score highly by reading paper rather than soil.

Two shapes are masked per image:
  * card -- a large yellow-cream rectangle anywhere in frame
  * tag  -- a small, bright, well-filled white rectangle (the rim label)

The rim itself is bright too, so a candidate is only accepted when it fills
most of its bounding box: the metal ring is a thin arc and fails that test,
while a paper rectangle passes it. Detections are unioned per dataset into a
stable region across the split, because the pot shifts slightly between
frames and a per-image box occasionally clips a corner of the paper.

Usage:
  python scripts/data_prep/build_card_masked.py --src "results/data/training-data/downloaded" \
      --out "results/data/training-data/card_masked" \
      --datasets soil-moisture-5sagf soil-moisture-ir soil-moisture-ir-combined
"""

import argparse
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

from laser_ablation_dataset import COLOR_PRESETS, laser_mask

PAPER_MIN_AREA = 500      # px; smaller than the rim tag is noise
PAPER_MAX_AREA = 40000    # px; larger than any label -- that is the pot
MIN_FILL = 0.55           # bbox fill: paper is solid, the metal rim is an arc
MAX_ASPECT = 5.0
MIN_INK = 0.05            # share of dark pixels: handwriting must be present
PAD = 8


def paper_boxes(img):
    """Bounding boxes of the paper labels (sticky note and rim tag) in a frame.

    Both labels are cream card stock, so one hue rule finds both; the rim tag
    is simply darker because it sits in the pot's shade, which is why the
    brightness floor is low. What separates paper from the similarly coloured
    olive pot is size (the pot is far larger than any label) and from plastic
    sheeting is hue. The handwriting test is the final filter: a label has
    dark ink strokes inside it, a blank bright surface does not.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Bounds measured from tight patches on the actual objects:
    #   label paper  H 26-39  S  78-98  V 101-149   <- wanted
    #   olive pot    H 37     S 165     V  66       <- excluded by saturation
    #   lit pot rim  H 81     S  39     V 110       <- excluded by hue
    #   plastic      H 96     S  56     V  51       <- excluded by hue
    #   wet soil     H 148    S 173     V  14       <- excluded by hue
    warm = ((h >= 15) & (h <= 45) & (s >= 60) & (s <= 140) & (v >= 80))
    # The rim tags photograph anywhere from cream to near-white depending on
    # how the pot shades them, and a white tag has no meaningful hue, so a
    # second low-saturation rule is needed to catch them. Plastic sheeting is
    # bright too, but it carries no writing and is dropped by the ink test.
    pale = (s <= 70) & (v >= 150)
    paper = warm | pale
    m = paper.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    boxes = []
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < PAPER_MIN_AREA or area > PAPER_MAX_AREA:
            continue  # smaller than a tag, or as large as the pot
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        w, hh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        if w == 0 or hh == 0:
            continue
        if max(w / hh, hh / w) > MAX_ASPECT:
            continue
        # Solidity against the *rotated* bounding rectangle. The labels lie at
        # an angle and their ink strokes punch holes in the colour mask, so an
        # axis-aligned fill ratio reads low (0.34 for the rim tag) and would
        # reject them; a rotated rect measures the shape itself.
        comp = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        rect_area = cv2.minAreaRect(max(cnts, key=cv2.contourArea))[1]
        rect_area = rect_area[0] * rect_area[1]
        if rect_area <= 0 or cv2.contourArea(max(cnts, key=cv2.contourArea)) / rect_area < MIN_FILL:
            continue
        patch = v[y:y + hh, x:x + w]
        if patch.size == 0:
            continue
        ink = float((patch < 0.55 * patch.mean()).mean())
        if ink < MIN_INK:
            continue  # a blank lit surface, not a written label
        boxes.append((x, y, x + w - 1, y + hh - 1))
    return boxes


def union_regions(all_boxes, shape):
    """Rasterise every detection over the split into one stable mask."""
    acc = np.zeros(shape, np.uint8)
    for x0, y0, x1, y1 in all_boxes:
        acc[max(0, y0 - PAD):y1 + 1 + PAD, max(0, x0 - PAD):x1 + 1 + PAD] = 255
    return acc


def read_classes(src_yaml):
    text = src_yaml.read_text()
    nc = re.search(r"^nc:\s*(\d+)", text, re.M)
    names = re.search(r"^names:\s*(\[.*\])", text, re.M)
    if names:
        names_val = names.group(1)
    else:
        blk = re.search(r"names:\n((?:- .*\n)+)", text)
        names_val = ("[" + ", ".join(l[2:].strip()
                     for l in blk.group(1).splitlines()) + "]") if blk else None
    return (nc.group(1) if nc else None), names_val


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--split", default="valid")
    args = ap.parse_args()

    src_root, out_root = Path(args.src), Path(args.out)
    qc = out_root / "qc"
    qc.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        src = src_root / name
        img_dir = src / args.split / "images"
        lbl_dir = src / args.split / "labels"
        if not img_dir.is_dir():
            print(f"skip {name}: no {args.split} split")
            continue
        paths = sorted(p for p in img_dir.iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))

        out_img = out_root / name / args.split / "images"
        out_lbl = out_root / name / args.split / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        # Mask per image rather than unioning every detection over the split.
        # The pot shifts between frames, so a union accumulates into a blob
        # that also covers soil -- the very signal under test.
        n_frames = n_empty = n_boxes = n_dropped = 0
        covered_pct = []
        first = None
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            n_frames += 1
            boxes = paper_boxes(img)
            # Never blank the laser. A label box that clips the spot would
            # confound this test with the laser-removal one, so drop it.
            lm, _, _ = laser_mask(img, COLOR_PRESETS["violet"])
            if lm is not None:
                kept = []
                for x0, y0, x1, y1 in boxes:
                    y0p, y1p = max(0, y0 - PAD), min(img.shape[0] - 1, y1 + PAD)
                    x0p, x1p = max(0, x0 - PAD), min(img.shape[1] - 1, x1 + PAD)
                    if (lm[y0p:y1p + 1, x0p:x1p + 1] > 0).any():
                        n_dropped += 1
                        continue
                    kept.append((x0, y0, x1, y1))
                boxes = kept
            n_boxes += len(boxes)
            if not boxes:
                n_empty += 1
            before = img.copy()
            covered = 0
            for x0, y0, x1, y1 in boxes:
                y0p, y1p = max(0, y0 - PAD), min(img.shape[0] - 1, y1 + PAD)
                x0p, x1p = max(0, x0 - PAD), min(img.shape[1] - 1, x1 + PAD)
                img[y0p:y1p + 1, x0p:x1p + 1] = 0
                covered += (y1p - y0p + 1) * (x1p - x0p + 1)
            covered_pct.append(100.0 * covered / (img.shape[0] * img.shape[1]))
            if first is None:
                first = np.hstack([before, img.copy()])
            cv2.imwrite(str(out_img / p.name), img)
            lbl = lbl_dir / (p.stem + ".txt")
            if lbl.is_file():
                shutil.copy(lbl, out_lbl / lbl.name)

        nc, names = read_classes(src / "data.yaml")
        if nc and names:
            (out_root / name / "data.yaml").write_text(
                f"train: {out_img.resolve()}\nval: {out_img.resolve()}\n"
                f"nc: {nc}\nnames: {names}\n")

        if first is not None:
            cv2.imwrite(str(qc / f"{name}.jpg"), first)
        print(f"{name:<34} frames={n_frames:<4} labels_found={n_boxes:<4} "
              f"frames_with_none={n_empty:<3} dropped_near_laser={n_dropped:<3} "
              f"mean_masked={np.mean(covered_pct):.1f}% of frame",
              flush=True)


if __name__ == "__main__":
    main()
