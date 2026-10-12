"""Grouped, class-stratified k-fold over the 56 stir-probe originals.

The stir-probe build (soil_moisture_stir_september) holds 1157 frames, but only
56 of them are photographs; the rest are offline augmented copies (`_aug000`..
from augment_stir_september.py, `_augx00`.. from augment_stir_dataset.py). A copy
is not an independent sample, so every copy must travel with its original.

Originals are not independent of each other either. build_cv_groups.py found the
56 originals fall into 42 capture bursts, and the shipped split already lets five
of those bursts straddle train/valid/test. So the unit of assignment here is the
burst: a burst, its originals and all their copies land in one fold together.

Bursts are single-class (checked), so stratification is done at burst level:
bursts are dealt into folds class by class, largest first, each into the fold
currently holding the fewest originals of that class. Every fold's validation
then spans all five levels and every training split keeps all five.

The shipped train/valid/test boundaries are discarded; all 56 originals are
pooled. There is no untouched test set -- the 5 test originals were too few to
carry one, and one of them shares a burst with training anyway. Out-of-fold
predictions over all 56 originals take its place.

Writes, per fold:
  data/training-data/cv/stir/foldNN/{train,valid}/{images,labels}
  data/training-data/cv/stir/foldNN/data.yaml
and the manifest runs/cv/stir_cv_folds.csv (one row per original).

Usage (from the project root):
  python build_stir_cv.py            # 5 folds
  python build_stir_cv.py --folds 4
"""

import argparse
import collections
import csv
import glob
import os
import re
import shutil
import sys
from pathlib import Path

PROJECT_DIR = "/projects/kp9e/Irrigation-Laser-Yolo2"
if not os.path.isdir(PROJECT_DIR):
    PROJECT_DIR = str(Path(__file__).resolve().parent)

DATASET = "soil_moisture_stir_september"
SRC = os.path.join(PROJECT_DIR, "data", "training-data", "downloaded", DATASET)
OUT = os.path.join(PROJECT_DIR, "data", "training-data", "cv", "stir")
GROUPS = os.path.join(PROJECT_DIR, "runs", "cv", f"{DATASET}_groups.csv")
MANIFEST = os.path.join(PROJECT_DIR, "runs", "cv", "stir_cv_folds.csv")
COPY_RE = re.compile(r"_aug(x)?\d+$")


def read_names():
    names, in_names = [], False
    for line in open(os.path.join(SRC, "data.yaml")):
        s = line.strip()
        if s.startswith("names:"):
            rest = s[len("names:"):].strip()
            if rest.startswith("["):
                return [x.strip().strip("'\"") for x in rest[1:-1].split(",") if x.strip()]
            in_names = True
        elif in_names:
            if s.startswith("- "):
                names.append(s[2:].strip().strip("'\""))
            elif s and not line.startswith((" ", "\t", "-")):
                break
    if not names:
        sys.exit("ERROR: could not parse names from data.yaml")
    return names


def label_class(path):
    toks = open(path).read().split()
    if not toks:
        return None
    return int(float(toks[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    names = read_names()

    if not os.path.isfile(GROUPS):
        sys.exit(f"ERROR: {GROUPS} not found -- run build_cv_groups.py --dataset {DATASET}")
    burst_of = {os.path.basename(r["path"]): r["group"] for r in csv.DictReader(open(GROUPS))}

    # every frame, keyed to its original's stem
    frames = collections.defaultdict(list)  # stem -> [(img_path, lbl_path)]
    split_of = {}
    for split in ("train", "valid", "test"):
        for img in sorted(glob.glob(os.path.join(SRC, split, "images", "*.jpg"))):
            stem = COPY_RE.sub("", Path(img).stem)
            lbl = os.path.join(SRC, split, "labels", Path(img).stem + ".txt")
            frames[stem].append((img, lbl))
            if not COPY_RE.search(Path(img).stem):
                split_of[stem] = split

    originals = sorted(split_of)
    orphans = sorted(set(frames) - set(originals))
    if orphans:
        sys.exit(f"ERROR: {len(orphans)} copy stems have no original, e.g. {orphans[:3]}")

    cls_of, bursts = {}, collections.defaultdict(list)
    for stem in originals:
        lbl = os.path.join(SRC, split_of[stem], "labels", stem + ".txt")
        c = label_class(lbl)
        if c is None:
            sys.exit(f"ERROR: original {stem} has an empty label")
        cls_of[stem] = c
        b = burst_of.get(stem + ".jpg")
        if b is None:
            sys.exit(f"ERROR: original {stem}.jpg missing from {GROUPS}")
        bursts[b].append(stem)

    for b, stems in bursts.items():
        if len({cls_of[s] for s in stems}) != 1:
            sys.exit(f"ERROR: burst {b} mixes classes; stratifying by burst is invalid")
    per_class = collections.Counter(cls_of.values())
    if min(per_class.values()) < args.folds:
        sys.exit(f"ERROR: a class has only {min(per_class.values())} originals; "
                 f"cannot place one in each of {args.folds} folds")

    # deal bursts into folds, class by class, largest burst first
    fold_of_burst = {}
    load = [[0] * len(names) for _ in range(args.folds)]
    total = [0] * args.folds
    for c in sorted(per_class):
        bs = sorted((b for b in bursts if cls_of[bursts[b][0]] == c),
                    key=lambda b: (-len(bursts[b]), b))
        for b in bs:
            f = min(range(args.folds), key=lambda k: (load[k][c], total[k], k))
            fold_of_burst[b] = f
            load[f][c] += len(bursts[b])
            total[f] += len(bursts[b])

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    rows = []
    for f in range(args.folds):
        root = os.path.join(OUT, f"fold{f:02d}")
        for split in ("train", "valid"):
            for sub in ("images", "labels"):
                os.makedirs(os.path.join(root, split, sub), exist_ok=True)
        for b, stems in bursts.items():
            split = "valid" if fold_of_burst[b] == f else "train"
            for stem in stems:
                for img, lbl in frames[stem]:
                    shutil.copy2(img, os.path.join(root, split, "images", os.path.basename(img)))
                    shutil.copy2(lbl, os.path.join(root, split, "labels", os.path.basename(lbl)))
        with open(os.path.join(root, "data.yaml"), "w") as fh:
            fh.write(f"path: {root}\ntrain: train/images\nval: valid/images\n")
            fh.write(f"nc: {len(names)}\nnames: [{', '.join(repr(n) for n in names)}]\n")

    for b, stems in bursts.items():
        for stem in stems:
            rows.append(dict(original=stem, burst=b, cls=cls_of[stem], level=names[cls_of[stem]],
                             shipped_split=split_of[stem], fold=fold_of_burst[b],
                             n_frames=len(frames[stem])))
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["fold"], r["cls"], r["original"])))

    print(f"{len(originals)} originals, {len(bursts)} bursts, "
          f"{sum(len(v) for v in frames.values())} frames -> {args.folds} folds")
    print("fold  val originals per level (" + ", ".join(names) + ")   val frames  train frames")
    for f in range(args.folds):
        root = os.path.join(OUT, f"fold{f:02d}")
        nv = len(os.listdir(os.path.join(root, "valid", "images")))
        nt = len(os.listdir(os.path.join(root, "train", "images")))
        print(f"{f:>4}  {load[f]}  {nv:>10}  {nt:>12}")
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
