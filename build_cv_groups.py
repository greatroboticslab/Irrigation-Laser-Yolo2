"""Group near-duplicate capture bursts so cross-validation folds are honest.

The v4-family datasets are burst captures: a static overhead rig photographed
many consecutive frames at one moisture level. Roboflow then split those frames
randomly, so nearly every validation frame has a near-twin in train carrying the
same label. Measured on soil-moisture-v4-uv, a validation frame's nearest train
neighbour sits at pixel MSE ~39 while two random frames of the same class sit at
~45 and two frames of different classes at ~3896.

A plain k-fold inherits that structure and reports a tight mean close to the
single-run number, which looks like evidence of stability and is not. Folds have
to be split by capture burst, not by frame.

Bursts are not recoverable from filenames: the `N_png.rf.<hash>` prefix is a
per-batch index, not a source-photo id (verified separately -- `1_png` is class 7
in train and class 1 in valid). So bursts are recovered from the pixels: build a
graph over all frames, join any pair closer than a threshold, and take connected
components.

The threshold is not hardcoded. The script measures the within-class and
between-class distance distributions and places the cut in the gap between them,
reporting both so the choice is auditable.

One outcome makes grouped CV impossible and the script says so loudly: if every
class collapses to a single burst, then holding out a burst also holds out a
class, and no fold can train on the label it is scored against. That is not a
fixable split -- it means the dataset is single-session and cannot support a
cross-session generalisation claim at all.

Usage (from the project root):
  python build_cv_groups.py                          # default soil-moisture-v4-uv
  python build_cv_groups.py --dataset soil-moisture-v4
  python build_cv_groups.py --threshold 800          # override the measured cut
"""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = "/projects/kp9e/Irrigation-Laser-Yolo2"
if not os.path.isdir(PROJECT_DIR):  # allow local runs
    PROJECT_DIR = str(Path(__file__).resolve().parent)

DATA_ROOT = os.path.join(PROJECT_DIR, "data", "training-data", "downloaded")
SPLITS = ("train", "valid", "test")
WORK = 128  # frames are compared at 128x128; full resolution changes nothing here


def load_frames(dataset):
    """Every frame of every split, with its split of origin and its label set."""
    rows = []
    for split in SPLITS:
        for path in sorted(glob.glob(os.path.join(DATA_ROOT, dataset, split, "images", "*"))):
            if not path.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            label_path = path.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
            label_path = label_path.rsplit(".", 1)[0] + ".txt"
            try:
                with open(label_path) as f:
                    classes = tuple(sorted({line.split()[0] for line in f if line.strip()}))
            except OSError:
                classes = ()
            rows.append({"path": path, "split": split, "classes": classes})
    return rows


def descriptors(rows):
    out = []
    for r in rows:
        im = cv2.imread(r["path"])
        if im is None:
            print(f"WARNING: unreadable, dropped: {r['path']}", flush=True)
            out.append(None)
            continue
        out.append(cv2.resize(im, (WORK, WORK), interpolation=cv2.INTER_AREA).astype(np.float32))
    return out


def distance_matrix(descs):
    n = len(descs)
    flat = np.stack([d.ravel() for d in descs])
    # ||a-b||^2 / dim  == mean squared error, computed in one pass
    sq = (flat ** 2).sum(1)
    d = sq[:, None] + sq[None, :] - 2.0 * (flat @ flat.T)
    np.maximum(d, 0, out=d)
    return d / flat.shape[1]


def choose_threshold(dist, classes):
    """Pick the cut from the plateau of the burst-count sweep.

    An earlier version placed the cut between the within-class and between-class
    percentiles. That works only when the two distributions separate cleanly; on
    soil-moisture-v4 the within-class spread is wide (median ~3200) and the
    heuristic over-merged 351 frames into 6 bursts, when the stable answer across
    thresholds 50-800 is 35-43. A grouping that wrong produces folds that look
    valid and are not.

    Burst count falls as the threshold rises, but it falls slowly across the range
    where the grouping is genuinely right and steeply where bursts start fusing.
    So sweep, find the longest run of near-constant burst count, and take its
    geometric middle. The sweep is returned so the choice can be inspected.
    """
    grid = [50, 70, 100, 140, 200, 280, 400, 560, 800, 1130, 1600, 2260, 3200]
    sweep = []
    for t in grid:
        _, n = connected_components(dist, t)
        sweep.append((t, n))

    best, run_start = None, 0
    for i in range(1, len(sweep) + 1):
        # a run holds while the burst count stays within 15% of where it started
        if i < len(sweep) and abs(sweep[i][1] - sweep[run_start][1]) <= 0.15 * max(sweep[run_start][1], 1):
            continue
        span = (sweep[run_start][0], sweep[i - 1][0], i - run_start, sweep[run_start][1])
        if best is None or span[2] > best[2]:
            best = span
        run_start = i
    lo, hi, _, _ = best
    return float((lo * hi) ** 0.5), sweep


def connected_components(dist, threshold):
    n = dist.shape[0]
    adj = dist <= threshold
    np.fill_diagonal(adj, True)
    group = -np.ones(n, dtype=int)
    g = 0
    for start in range(n):
        if group[start] != -1:
            continue
        stack, members = [start], []
        group[start] = g
        while stack:
            i = stack.pop()
            members.append(i)
            for j in np.flatnonzero(adj[i] & (group == -1)):
                group[j] = g
                stack.append(j)
        g += 1
    return group, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="soil-moisture-v4-uv")
    ap.add_argument("--threshold", type=float, default=None,
                    help="distance cut; default is measured from the data")
    ap.add_argument("--out", default=None, help="groups csv (default runs/cv/<dataset>_groups.csv)")
    args = ap.parse_args()

    rows = load_frames(args.dataset)
    if not rows:
        sys.exit(f"ERROR: no frames under {os.path.join(DATA_ROOT, args.dataset)}")
    descs = descriptors(rows)
    keep = [i for i, d in enumerate(descs) if d is not None]
    rows = [rows[i] for i in keep]
    descs = [descs[i] for i in keep]
    classes = [r["classes"] for r in rows]
    counts = {s: sum(1 for r in rows if r["split"] == s) for s in SPLITS}
    shown = ", ".join(f"{s}={counts[s]}" for s in SPLITS)
    print(f"{args.dataset}: {len(rows)} frames ({shown})", flush=True)

    dist = distance_matrix(descs)
    measured, sweep = choose_threshold(dist, classes)
    print("\nburst-count sweep (the cut is taken from the flattest stretch):")
    print("   " + "  ".join(f"{t}:{n}" for t, n in sweep))
    threshold = args.threshold if args.threshold is not None else measured
    print(f"threshold                  : {threshold:9.1f}"
          f"{'  (measured)' if args.threshold is None else '  (overridden)'}")

    group, n_groups = connected_components(dist, threshold)

    # A burst is "near-identical frames that share a label". Pixel clustering
    # alone can merge frames carrying different labels, and holding such a group
    # out would remove two levels at once. Split every group by label.
    #
    # Where the frames joined across a label boundary are also near-identical,
    # that is a conflict in the data rather than a clustering artifact, and no
    # threshold repairs it -- so report those pairs instead of hiding the split.
    CONFLICT_DIST = 20.0
    conflicts = []
    for gid in range(n_groups):
        idx = np.flatnonzero(group == gid)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                if classes[i] != classes[j] and dist[i, j] <= CONFLICT_DIST:
                    conflicts.append((dist[i, j], rows[i], rows[j]))
    key, refined = {}, []
    for gid, cls in zip(group, classes):
        k = (gid, cls)
        if k not in key:
            key[k] = len(key)
        refined.append(key[k])
    split = len(key) - n_groups
    group, n_groups = np.array(refined), len(key)
    if split:
        print(f"\nsplit {split} burst(s) that spanned more than one label")
    if conflicts:
        conflicts.sort(key=lambda c: c[0])
        print(f"\nWARNING: {len(conflicts)} near-identical frame pair(s) carry different "
              f"labels (distance <= {CONFLICT_DIST:g}).")
        print("No clustering threshold separates these; they are a labelling conflict.")
        for d, ri, rj in conflicts[:5]:
            print(f"   d={d:6.1f}  {ri['classes']} {os.path.basename(ri['path'])}")
            print(f"            {rj['classes']} {os.path.basename(rj['path'])}")
        if len(conflicts) > 5:
            print(f"   ... and {len(conflicts) - 5} more")

    sizes = np.bincount(group)
    print(f"\ngroups (capture bursts)    : {n_groups}")
    print(f"group size                 : min={sizes.min()} median={int(np.median(sizes))} max={sizes.max()}")

    # Does each class collapse to one burst? If so, grouped CV is impossible.
    by_class = {}
    for cls, g in zip(classes, group):
        by_class.setdefault(cls, set()).add(g)
    multi = {c: gs for c, gs in by_class.items() if len(gs) > 1}
    print(f"\nclasses                    : {len(by_class)}")
    print(f"classes split over >1 burst: {len(multi)}")
    for cls in sorted(by_class, key=lambda c: (len(by_class[c]), c)):
        print(f"   class {str(cls):10s} -> {len(by_class[cls])} burst(s)")

    if not multi:
        print("\n" + "=" * 70)
        print("STOP: every class is a single capture burst.")
        print("Holding out a burst therefore holds out a whole class, and no fold")
        print("can train on the label it is scored against. Grouped CV cannot be")
        print("run on this dataset, and a plain k-fold would only re-measure the")
        print("leak. The honest reading is that this capture is single-session:")
        print("report it as in-session readout and make no cross-session claim.")
        print("=" * 70)
    elif len(multi) < len(by_class):
        print("\nNOTE: some classes sit in a single burst. Those classes cannot appear")
        print("in a held-out fold without vanishing from training. build_cv_folds.py")
        print("will refuse to place them and will report the reduced label space.")

    out = args.out or os.path.join(PROJECT_DIR, "runs", "cv", f"{args.dataset}_groups.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "split", "classes", "group"])
        for r, g in zip(rows, group):
            w.writerow([r["path"], r["split"], "|".join(r["classes"]), int(g)])
    print(f"\nwrote {out}")

    # How badly is the shipped split leaking? Report it so the paper can quote it.
    tr = {g for r, g in zip(rows, group) if r["split"] == "train"}
    va = {g for r, g in zip(rows, group) if r["split"] == "valid"}
    shared = tr & va
    n_val = sum(1 for r in rows if r["split"] == "valid")
    leaked = sum(1 for r, g in zip(rows, group) if r["split"] == "valid" and g in tr)
    print(f"\nshipped split: {len(shared)}/{len(va)} validation bursts also appear in train; "
          f"{leaked}/{n_val} validation frames ({100*leaked/max(n_val,1):.0f}%) sit in a burst "
          f"that training has already seen.")


if __name__ == "__main__":
    main()
