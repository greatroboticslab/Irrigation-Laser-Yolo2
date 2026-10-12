"""Blank the burned-in "Moisture: N" overlay and rebuild the validation images.

The capture software prints the ground-truth moisture value into a corner of
every frame. A detector can read those digits instead of the soil, so any
score measured on unmasked frames is open to a label-leakage objection. This
locates the overlay per dataset and writes a masked copy of the validation
split; scripts/evaluation/eval_overlay_leak.py then scores each model on both.

The overlay is not always in the same corner (top-left in some captures,
top-right in others), so its position is found from the data rather than
assumed: the text is rendered in saturated yellow-green, and unlike scene
content it sits in the *same pixels* in every frame. Pixels that are yellow
in a large fraction of the split are therefore overlay, and their bounding
box -- padded -- is what gets blanked.

Usage:
  python scripts/data_prep/build_overlay_masked.py --src "results/data/training-data/downloaded" \
      --out "results/data/training-data/overlay_masked"
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

PRESENCE_FRAC = 0.30   # a pixel is overlay if yellow in this share of frames
PAD = 6                # px of padding around the detected overlay box
SAMPLE = 60            # frames sampled when locating the overlay


MAX_TEXT_W = 300      # an overlay wider than this is scene content, not text
MAX_TEXT_H = 120


def text_mask(img):
    """Pixels of the overlay's yellow-green text.

    Matching vividness generally (any saturated bright pixel) pulled in olive
    plant pots and sunlit soil, so this stays on the specific yellow the
    capture software draws with.
    """
    b, g, r = cv2.split(img.astype(np.int16))
    return (r > 130) & (g > 130) & (b < 110) & (np.minimum(r, g) - b > 45)


def find_overlay_box(img_paths):
    """Bounding box of the burned-in overlay, or None if there isn't one.

    Restricted to the four corners: the overlay is always printed in one, and
    an unrestricted search happily returns an olive plant pot or a sunlit
    strip of soil, which would then be blanked out of the evaluation and
    quietly invalidate it. A candidate is also rejected when it fills much of
    its corner window, since real text is sparse.
    """
    acc = None
    n = 0
    for p in img_paths[:SAMPLE]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        m = text_mask(img).astype(np.float32)
        acc = m if acc is None else acc + m
        n += 1
    if not n:
        return None
    presence = acc / n
    stable = presence >= PRESENCE_FRAC
    H, W = stable.shape

    cw, ch = min(260, W // 2), min(90, H // 4)
    corners = {
        "tl": (0, 0, cw, ch),
        "tr": (W - cw, 0, cw, ch),
        "bl": (0, H - ch, cw, ch),
        "br": (W - cw, H - ch, cw, ch),
    }
    best = None
    for cx, cy, w, h in corners.values():
        sub = stable[cy:cy + h, cx:cx + w]
        count = int(sub.sum())
        if count < 20:
            continue
        if count > 0.40 * w * h:
            continue  # too dense to be text; scene content in the corner
        ys, xs = np.where(sub)
        box = (max(0, cx + int(xs.min()) - PAD), max(0, cy + int(ys.min()) - PAD),
               min(W - 1, cx + int(xs.max()) + PAD), min(H - 1, cy + int(ys.max()) + PAD))
        if box[2] - box[0] > MAX_TEXT_W or box[3] - box[1] > MAX_TEXT_H:
            continue  # a yellow pot or lit soil strip, not an overlay
        if best is None or count > best[0]:
            best = (count, box)
    return best[1] if best else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="valid")
    args = ap.parse_args()

    src_root, out_root = Path(args.src), Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    report = []

    for src in sorted(p for p in src_root.iterdir() if p.is_dir()):
        img_dir = src / args.split / "images"
        lbl_dir = src / args.split / "labels"
        if not img_dir.is_dir():
            continue
        paths = sorted(p for p in img_dir.iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        box = find_overlay_box(paths)
        out_img = out_root / src.name / args.split / "images"
        out_lbl = out_root / src.name / args.split / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            if box:
                x0, y0, x1, y1 = box
                img[y0:y1 + 1, x0:x1 + 1] = 0
            cv2.imwrite(str(out_img / p.name), img)
            lbl = lbl_dir / (p.stem + ".txt")
            if lbl.is_file():
                shutil.copy(lbl, out_lbl / lbl.name)

        # Reuse the source data.yaml's classes, repointed at the masked images.
        src_yaml = src / "data.yaml"
        if src_yaml.is_file():
            import re
            text = src_yaml.read_text()
            nc = re.search(r"^nc:\s*(\d+)", text, re.M)
            names = re.search(r"^names:\s*(\[.*\])", text, re.M)
            if not names:
                blk = re.search(r"names:\n((?:- .*\n)+)", text)
                names_val = ("[" + ", ".join(l[2:].strip()
                             for l in blk.group(1).splitlines()) + "]") if blk else None
            else:
                names_val = names.group(1)
            if nc and names_val:
                (out_root / src.name / "data.yaml").write_text(
                    f"train: {out_img.resolve()}\nval: {out_img.resolve()}\n"
                    f"nc: {nc.group(1)}\nnames: {names_val}\n")

        report.append((src.name, box, len(paths)))
        print(f"{src.name:<38} overlay_box={box} images={len(paths)}", flush=True)

    missing = [n for n, b, _ in report if b is None]
    if missing:
        print("\nNo overlay found (nothing masked) in: " + ", ".join(missing))


if __name__ == "__main__":
    main()
