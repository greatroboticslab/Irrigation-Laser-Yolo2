"""Materialise leave-one-burst-out folds, each paired with a matched random control.

Reads the burst assignment from scripts/data_prep/build_cv_groups.py and writes, per fold, two
datasets that differ in exactly one respect: whether the validation frames have
burst siblings sitting in training.

  grouped  validation holds out one whole capture burst from EVERY class at
           once, so validation stays multi-class and no validation frame has a
           burst sibling in training
  random   validation is the same number of frames of the same classes, drawn at
           random within each class, so burst siblings stay in training exactly
           as the shipped split has them

Validation must span more than one class. An earlier leave-one-burst-out version
held out a single burst per fold, which made every validation set single-class:
with one class present the detector never has to discriminate hue, only locate
the LED, and all 28 runs returned exactly 0.9950 -- YOLOv5's ceiling for perfect
detection -- in both arms. Zero variance across 28 trainings was the tell. That
design cannot detect leakage because it is not measuring classification.

Same class, same frame count, same everything else. The gap between them is the
size of the leak in the shipped split, measured rather than argued.

Only classes holding two or more bursts are eligible: holding out the single
burst of a one-burst class would delete that class from training, so those
classes are dropped and the label space shrinks. The fold count is then capped by
the thinnest eligible class -- with classes holding as few as two bursts, two
folds is the honest maximum, and per-fold variance has to come from seeds rather
than from more folds. The reduced space is written
into every fold's data.yaml and must be stated wherever the numbers are quoted --
an mAP over six levels is not comparable to an mAP over eleven.

Class indices are remapped to a contiguous range. Source indices are alphabetical
(index 2 is moisture level 10), so names are carried across from the source
data.yaml rather than reconstructed.

Usage (from the project root):
  python scripts/data_prep/build_cv_groups.py --dataset soil-moisture-v4-uv
  python scripts/data_prep/build_cv_folds.py  --dataset soil-moisture-v4-uv
"""

import argparse
import collections
import csv
import os
import random
import shutil
import sys
from pathlib import Path

PROJECT_DIR = "/projects/kp9e/Irrigation-Laser-Yolo2"
if not os.path.isdir(PROJECT_DIR):
    PROJECT_DIR = str(Path(__file__).resolve().parents[2])

SRC_ROOT = os.path.join(PROJECT_DIR, "data", "training-data", "downloaded")
CV_ROOT = os.path.join(PROJECT_DIR, "data", "training-data", "cv")


def read_source_names(dataset):
    """names: from the source data.yaml, block list or inline."""
    path = os.path.join(SRC_ROOT, dataset, "data.yaml")
    names, in_names = [], False
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("names:"):
                rest = stripped[len("names:"):].strip()
                if rest.startswith("["):
                    return [x.strip().strip("'\"") for x in rest[1:-1].split(",") if x.strip()]
                in_names = True
                continue
            if in_names:
                if stripped.startswith("- "):
                    names.append(stripped[2:].strip().strip("'\""))
                elif stripped and not line.startswith((" ", "\t", "-")):
                    break
    if not names:
        sys.exit(f"ERROR: could not parse names from {path}")
    return names


def write_yaml(path, root, names):
    with open(path, "w") as f:
        f.write(f"train: {root}/train/images\n")
        f.write(f"val: {root}/valid/images\n")
        f.write(f"nc: {len(names)}\n")
        f.write("names: [" + ", ".join(f"'{n}'" for n in names) + "]\n")


def place(frames, dst_root, remap):
    """Copy frames into dst_root, rewriting label class indices through remap."""
    for split in ("train", "valid"):
        os.makedirs(os.path.join(dst_root, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(dst_root, split, "labels"), exist_ok=True)
    for split, path in frames:
        base = os.path.basename(path)
        shutil.copy2(path, os.path.join(dst_root, split, "images", base))
        src_lbl = path.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
        src_lbl = src_lbl.rsplit(".", 1)[0] + ".txt"
        dst_lbl = os.path.join(dst_root, split, "labels", base.rsplit(".", 1)[0] + ".txt")
        out = []
        try:
            with open(src_lbl) as f:
                for line in f:
                    parts = line.split()
                    if not parts:
                        continue
                    old = parts[0]
                    if old not in remap:      # class dropped from this study
                        continue
                    out.append(" ".join([str(remap[old])] + parts[1:]))
        except OSError:
            pass
        with open(dst_lbl, "w") as f:
            f.write("\n".join(out) + ("\n" if out else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="soil-moisture-v4-uv")
    ap.add_argument("--groups", default=None)
    ap.add_argument("--max-folds", type=int,
                    default=int(os.environ["CV_MAX_FOLDS"]) if os.environ.get("CV_MAX_FOLDS") else None,
                    help="cap the fold count (default: the thinnest class's burst count)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the random control draw")
    args = ap.parse_args()

    groups_csv = args.groups or os.path.join(PROJECT_DIR, "runs", "cv", f"{args.dataset}_groups.csv")
    if not os.path.isfile(groups_csv):
        sys.exit(f"ERROR: {groups_csv} not found -- run scripts/data_prep/build_cv_groups.py first")
    rows = list(csv.DictReader(open(groups_csv)))

    src_names = read_source_names(args.dataset)

    bursts_by_class = collections.defaultdict(set)
    frames_by_burst = collections.defaultdict(list)
    for r in rows:
        bursts_by_class[r["classes"]].add(r["group"])
        frames_by_burst[r["group"]].append(r["path"])

    eligible = sorted(c for c, gs in bursts_by_class.items() if len(gs) > 1)
    dropped = sorted(c for c, gs in bursts_by_class.items() if len(gs) <= 1)
    if not eligible:
        sys.exit("ERROR: no class holds two or more bursts; grouped CV is impossible "
                 "on this dataset. See the note in scripts/data_prep/build_cv_groups.py.")

    remap = {old: i for i, old in enumerate(eligible)}
    names = [src_names[int(old)] if int(old) < len(src_names) else old for old in eligible]
    print(f"{args.dataset}: {len(eligible)} eligible classes -> nc={len(eligible)}")
    print(f"  kept   (src idx -> new idx, moisture level): "
          + ", ".join(f"{o}->{remap[o]} ({names[remap[o]]})" for o in eligible))
    print(f"  dropped (single burst): "
          + ", ".join(f"{o} ({src_names[int(o)] if int(o) < len(src_names) else o})" for o in dropped))

    eligible_frames = [r for r in rows if r["classes"] in eligible]
    eligible_paths = {r["path"] for r in eligible_frames}

    burst_classes = collections.defaultdict(set)
    for r in eligible_frames:
        burst_classes[r["group"]].add(r["classes"])
    straddling = {g: cs for g, cs in burst_classes.items() if len(cs) > 1}
    if straddling:
        for g, cs in sorted(straddling.items()):
            print(f"ERROR: burst {g} spans eligible classes {sorted(cs)}; holding it out "
                  f"would remove frames of more than one level at once.")
        sys.exit("Refusing to build folds. Re-run scripts/data_prep/build_cv_groups.py with a tighter "
                 "--threshold so that bursts do not merge across levels.")

    # Each fold holds out a 1/k share of every class's bursts, so validation
    # always spans every class and never shares a burst with training. Where a
    # class holds only two bursts that reduces to one burst per fold; where it
    # holds twenty, the share scales up so validation is not a handful of frames.
    bursts_sorted = {c: sorted(bursts_by_class[c]) for c in eligible}
    n_folds = min(len(bursts_sorted[c]) for c in eligible)
    if args.max_folds is not None:
        n_folds = min(n_folds, args.max_folds)
    if n_folds < 2:
        sys.exit("ERROR: the thinnest eligible class holds one burst; no fold can hold "
                 "it out and still train on it.")
    held_by_fold = [{c: [b for i, b in enumerate(bursts_sorted[c]) if i % n_folds == k]
                     for c in eligible} for k in range(n_folds)]
    print(f"  {len(eligible_frames)} frames, {n_folds} folds "
          f"(each holds out 1/{n_folds} of every class's bursts)")
    print()

    by_class_frames = collections.defaultdict(list)
    for r in eligible_frames:
        by_class_frames[r["classes"]].append(r["path"])

    rng = random.Random(args.seed)
    ds_root = os.path.join(CV_ROOT, args.dataset)
    if os.path.isdir(ds_root):
        shutil.rmtree(ds_root)
    manifest = []

    for k in range(n_folds):
        held = held_by_fold[k]

        g_val, r_val = set(), set()
        per_class = {}
        for c in eligible:
            burst_frames = [p for b in held[c] for p in frames_by_burst[b] if p in eligible_paths]
            g_val.update(burst_frames)
            # random control: same class, same count, burst structure ignored
            pool = list(by_class_frames[c])
            rng.shuffle(pool)
            r_val.update(pool[:len(burst_frames)])
            per_class[c] = len(burst_frames)

        g_frames = [("valid" if r["path"] in g_val else "train", r["path"]) for r in eligible_frames]
        r_frames = [("valid" if r["path"] in r_val else "train", r["path"]) for r in eligible_frames]

        for regime, frames in (("grouped", g_frames), ("random", r_frames)):
            root = os.path.join(ds_root, f"fold{k:02d}", regime)
            place(frames, root, remap)
            write_yaml(os.path.join(root, "data.yaml"), root, names)

        n_val = len(g_val)
        manifest.append({"fold": k, "n_val": n_val, "n_train": len(eligible_frames) - n_val,
                         "n_classes_val": len(eligible),
                         "bursts": "|".join(f"{c}:{','.join(held[c])}" for c in eligible),
                         "per_class_val": "|".join(f"{names[remap[c]]}:{per_class[c]}" for c in eligible)})
        print(f"fold {k:02d}: val={n_val:3d} frames over {len(eligible)} levels "
              f"({', '.join(f'{names[remap[c]]}={per_class[c]}' for c in eligible)}) "
              f"| train={len(eligible_frames) - n_val:3d}")

    man_path = os.path.join(PROJECT_DIR, "runs", "cv", f"{args.dataset}_folds.csv")
    os.makedirs(os.path.dirname(man_path), exist_ok=True)
    with open(man_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)
    print(f"\nwrote {man_path}")
    print(f"wrote {ds_root}/fold*/{{grouped,random}}/")


if __name__ == "__main__":
    main()
