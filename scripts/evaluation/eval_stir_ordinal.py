"""Ordinal metrics for the stir-probe grouped cross-validation.

The CV tables report top-1 accuracy. A reading task also needs to know how far off a wrong
reading is, so this adds, per arm at its final epoch, over the 56 held-out originals:
  acc       top-1 accuracy (class match and IoU >= 0.5, as in eval_stir_cv.py)
  cls_acc   class match of the top detection, ignoring IoU
  mae       mean absolute error in levels {1, 3, 7, 8, 10} of the top detection's class
  rank_mae  the same in rank steps (1 -> 3 is one step, 1 -> 10 four)
  far       share of originals read three or more ranks away (1 <-> 8 or 10, 3 <-> 10)
Each is computed per seed and reported as mean and s.d. over seeds. An original with no
detection counts as wrong for acc and cls_acc and is left out of the error averages.

Reads results/stir_cv_oof.csv (full, freeze arms) and results/stir_cv_oof_reg.csv (reg arm).
Writes results/stir_cv_ordinal.csv.

Usage (from the project root): python scripts/evaluation/eval_stir_ordinal.py
"""

import csv
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMES = ["1", "10", "3", "7", "8"]                 # data.yaml order
LEVEL = [int(n) for n in NAMES]
RANK = {lv: i for i, lv in enumerate(sorted(LEVEL))}
FINAL = {"full": 99, "freeze": 99, "reg": 59}
SRC = {"full": "stir_cv_oof.csv", "freeze": "stir_cv_oof.csv", "reg": "stir_cv_oof_reg.csv"}


def per_seed(rows):
    out = {}
    for seed in sorted({r["seed"] for r in rows}):
        rs = [r for r in rows if r["seed"] == seed]
        hit = [r for r in rs if int(r["pred_cls"]) >= 0]
        err = [abs(LEVEL[int(r["pred_cls"])] - LEVEL[int(r["gt_cls"])]) for r in hit]
        rerr = [abs(RANK[LEVEL[int(r["pred_cls"])]] - RANK[LEVEL[int(r["gt_cls"])]]) for r in hit]
        out[seed] = {
            "n": len(rs),
            "acc": sum(int(r["correct"]) for r in rs) / len(rs),
            "cls_acc": sum(1 for r in hit if r["pred_cls"] == r["gt_cls"]) / len(rs),
            "mae": sum(err) / len(err),
            "rank_mae": sum(rerr) / len(rerr),
            "far": sum(1 for e in rerr if e >= 3) / len(rs),
        }
    return out


def main():
    table = []
    for arm, ep in FINAL.items():
        rows = [r for r in csv.DictReader(open(ROOT / "results" / SRC[arm]))
                if r["arm"] == arm and int(r["epoch"]) == ep and r["is_original"] == "1"]
        seeds = per_seed(rows)
        row = {"arm": arm, "epoch": ep, "n_seeds": len(seeds),
               "n_originals": seeds[next(iter(seeds))]["n"]}
        for k in ("acc", "cls_acc", "mae", "rank_mae", "far"):
            vals = [s[k] for s in seeds.values()]
            row[k], row[k + "_sd"] = round(st.mean(vals), 4), round(st.stdev(vals), 4)
        table.append(row)
        print(row)
    out = ROOT / "results" / "stir_cv_ordinal.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print("saved", out)


if __name__ == "__main__":
    main()
