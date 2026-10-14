"""Score the Stage 3 stir regularisation sweep: does the overfitting turn-up go away?

For every run under runs/stir_reg/ (from scripts/training/tr_stir_regularize.py),
each saved checkpoint (every 5 epochs, plus last.pt) is rescored on the shipped
validation split with *unsmoothed* classification loss, so arms trained with
label smoothing sit on the same scale as the paper's run. The paper's own
Stage 3 last.pt is rescored the same way as the reference.

Per run it reports:
  min_ep      epoch of the lowest validation cls loss (all epochs; Stage 3 starts
              from trained weights, so there is no warm-up artefact to exclude)
  final_min   val cls loss at the last checkpoint / lowest val cls loss
  map_peak    best validation mAP@0.5 over the checkpoints
  map_final   validation mAP@0.5 of last.pt
  test_final  test mAP@0.5 of last.pt (95 frames, 5 originals; not used for selection)

An arm counts as not overfitting when, averaged over seeds, final_min <= 1.10
(the other Stage 3 runs in Table IV sit at 1.01-1.02) and map_final is within
0.02 of map_peak. The arm is picked on the validation split only; its test and
grouped cross-validation numbers are what the paper should report.

Stage 1 arms (names starting s1_) are scored with the Stage 1 loss gains
(data/hyps/hyp.stir-soil.yaml) against the paper's Stage 1 run ('paper_s1').
A COCO-initialised detector predicts almost nothing for its first epochs, which
gives a spuriously low loss, so for Stage 1 the minimum skips checkpoints
before the first with validation mAP@0.5 >= SILENT_MAP.

ratio_sm and map_drop_sm repeat final_min and map_peak - map_final on
3-checkpoint (15-epoch) rolling means: with 12 validation originals a single
checkpoint swings by about 0.05 mAP, which inflates max/min comparisons.

Outputs: results/stir_reg_curves.csv (one row per checkpoint) and
         results/stir_reg_summary.csv (one row per run, then per arm).

Usage (from the project root, after scripts/training/tr_stir_regularize.py):
  python scripts/evaluation/eval_stir_regularize.py
"""

import csv
import glob
import os
import re
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = "/projects/kp9e/Irrigation-Laser-Yolo2"
if not os.path.isdir(PROJECT_DIR):
    PROJECT_DIR = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

import torch  # noqa: E402
import yaml  # noqa: E402

import val as validate  # noqa: E402
from utils.dataloaders import create_dataloader  # noqa: E402
from utils.general import check_dataset, colorstr  # noqa: E402
from utils.loss import ComputeLoss  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402

RUNS = os.path.join(PROJECT_DIR, "runs", "stir_reg")
PAPER_RUN = os.path.join(PROJECT_DIR, "runs", "fine-tuned", "soil_moisture_stir_september")
PAPER_RUN_S1 = os.path.join(PROJECT_DIR, "runs", "train", "downloaded", "soil_moisture_stir_september")
S1_HYP = os.path.join(PROJECT_DIR, "data", "hyps", "hyp.stir-soil.yaml")
SILENT_MAP = 0.1
DATA = os.path.join(PROJECT_DIR, "data", "training-data", "downloaded",
                    "soil_moisture_stir_september", "data.yaml")
BASE_HYP = os.path.join(PROJECT_DIR, "data", "hyps", "hyp.stir-finetune.yaml")
OUT_DIR = os.path.join(PROJECT_DIR, "results")
NAME_RE = re.compile(r"stir__(\w+?)__seed(\d+)$")
IMGSZ, BATCH = 640, 64          # train.py validates at 2x the training batch
RATIO_MAX, MAP_DROP_MAX = 1.10, 0.02
WORKERS = int(os.environ.get("STIR_REG_WORKERS", "4"))   # 0 for local macOS runs


def checkpoints(run):
    """{epoch: path}. last.pt is the final epoch; a periodic save wins a tie."""
    w = os.path.join(run, "weights")
    out = {}
    last = os.path.join(w, "last.pt")
    res = os.path.join(run, "results.csv")
    if os.path.isfile(last) and os.path.isfile(res):
        with open(res) as f:
            out[sum(1 for _ in f) - 2] = last   # header + one row per epoch
    for p in glob.glob(os.path.join(w, "epoch*.pt")):
        out[int(re.search(r"epoch(\d+)\.pt$", p).group(1))] = p
    return sorted(out.items())


def loader(data, split, stride):
    return create_dataloader(data[split], IMGSZ, BATCH, stride, False, rect=True,
                             pad=0.5, workers=WORKERS, prefix=colorstr(f"{split}: "))[0]


def score(path, data, hyp, device, loaders):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = (ckpt.get("ema") or ckpt["model"]).float().to(device)
    # Same loss gains train.py uses, so values match the logged val/cls_loss scale,
    # but with label smoothing off for every arm.
    nl, nc = model.model[-1].nl, int(data["nc"])
    model.hyp = dict(hyp, label_smoothing=0.0, box=hyp["box"] * 3 / nl,
                     cls=hyp["cls"] * nc / 80 * 3 / nl, obj=hyp["obj"] * (IMGSZ / 640) ** 2 * 3 / nl)
    stride = int(max(model.stride.max(), 32))
    for split in ("val", "test"):
        if split not in loaders:
            loaders[split] = loader(data, split, stride)
    row = {}
    with tempfile.TemporaryDirectory() as tmp:
        for split in ("val", "test"):
            (mp, mr, map50, map95, box, obj, cls), _, _ = validate.run(
                data, batch_size=BATCH, imgsz=IMGSZ, model=model, dataloader=loaders[split],
                compute_loss=ComputeLoss(model), plots=False, half=device.type != "cpu",
                save_dir=Path(tmp), verbose=False)
            row[f"{split}_map50"], row[f"{split}_cls_loss"] = map50, cls
    return row


def stage1(arm):
    return arm.startswith("s1_") or arm == "paper_s1"


def rolling(xs, k=3):
    return [sum(xs[i - k + 1:i + 1]) / k for i in range(k - 1, len(xs))] or xs


def summarise(arm, seed, curve):
    curve = sorted(curve, key=lambda r: r["epoch"])
    if stage1(arm):  # drop the silent start of a COCO-initialised run
        first = next((i for i, r in enumerate(curve) if r["val_map50"] >= SILENT_MAP), 0)
        curve = curve[first:] if len(curve) - first >= 2 else curve
    losses = [r["val_cls_loss"] for r in curve]
    maps = [r["val_map50"] for r in curve]
    lo = min(range(len(losses)), key=losses.__getitem__)
    ls, ms = rolling(losses), rolling(maps)
    return {
        "arm": arm, "seed": seed, "n_ckpt": len(curve),
        "min_ep": curve[lo]["epoch"],
        "final_min": losses[-1] / losses[lo],
        "map_peak": max(maps),
        "map_final": maps[-1],
        "test_final": curve[-1]["test_map50"],
        "ratio_sm": ls[-1] / min(ls),
        "map_drop_sm": max(ms) - ms[-1],
    }


def main():
    device = select_device("")
    data = check_dataset(DATA)
    # data.yaml says "test: ../test/images", which YOLOv5 resolves against the
    # working directory rather than the yaml file; point it at the real folder.
    data["test"] = os.path.join(os.path.dirname(DATA), "test", "images")
    if not os.path.isdir(data["test"]):
        sys.exit(f"ERROR: {data['test']} not found")
    hyps = {}
    for stage, path in ((3, BASE_HYP), (1, S1_HYP)):
        with open(path) as f:
            hyps[stage] = yaml.safe_load(f)

    runs = [(a, "0", r) for a, r in (("paper", PAPER_RUN), ("paper_s1", PAPER_RUN_S1))
            if os.path.isdir(r)]
    for run in sorted(glob.glob(os.path.join(RUNS, "stir__*"))):
        m = NAME_RE.search(os.path.basename(run))
        if m:
            runs.append((m.group(1), m.group(2), run))
    if not runs:
        sys.exit(f"ERROR: no runs under {RUNS}")

    loaders, curves = {}, defaultdict(list)
    for arm, seed, run in runs:
        for epoch, path in checkpoints(run):
            row = {"arm": arm, "seed": seed, "epoch": epoch, **score(path, data, hyps[1 if stage1(arm) else 3], device, loaders)}
            curves[(arm, seed)].append(row)
            print(f"{arm:10s} seed{seed} ep{epoch:3d}  val cls {row['val_cls_loss']:.4f}"
                  f"  val mAP50 {row['val_map50']:.3f}  test mAP50 {row['test_map50']:.3f}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = [r for c in curves.values() for r in c]
    with open(os.path.join(OUT_DIR, "stir_reg_curves.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    per_run = [summarise(a, s, c) for (a, s), c in curves.items()]
    by_arm = defaultdict(list)
    for r in per_run:
        by_arm[r["arm"]].append(r)
    keys = ["min_ep", "final_min", "map_peak", "map_final", "test_final", "ratio_sm", "map_drop_sm"]
    per_arm = []
    for arm, rs in by_arm.items():
        mean = {k: statistics.mean(r[k] for r in rs) for k in keys}
        sd = {k: statistics.stdev(r[k] for r in rs) if len(rs) > 1 else 0.0 for k in keys}
        # A run with one checkpoint (the paper's, last.pt only) has final/min = 1
        # by construction, so it cannot pass or fail.
        ok = (mean["final_min"] <= RATIO_MAX and mean["map_peak"] - mean["map_final"] <= MAP_DROP_MAX
              if rs[0]["n_ckpt"] > 1 else "n/a")
        per_arm.append({"arm": arm, "seed": f"mean of {len(rs)}", "n_ckpt": rs[0]["n_ckpt"],
                        **mean, **{f"{k}_sd": sd[k] for k in keys},
                        "no_overfit": ok})
    fields = list(per_arm[0])
    with open(os.path.join(OUT_DIR, "stir_reg_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(per_run)
        w.writerows(per_arm)

    print("\narm         final/min  min ep  val mAP peak  val mAP final  test mAP final  no overfit")
    for r in sorted(per_arm, key=lambda r: r["final_min"]):
        print(f"{r['arm']:10s}  {r['final_min']:5.2f}±{r['final_min_sd']:.2f}  {r['min_ep']:6.0f}"
              f"  {r['map_peak']:.3f}        {r['map_final']:.3f}±{r['map_final_sd']:.3f}"
              f"    {r['test_final']:.3f}±{r['test_final_sd']:.3f}      {r['no_overfit']}")


if __name__ == "__main__":
    main()
