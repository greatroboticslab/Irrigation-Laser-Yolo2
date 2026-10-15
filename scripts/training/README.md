# scripts/training/

Training drivers. Each one loops over datasets or folds and calls the root `train.py`; the matching job is in [`../../slurm/`](../../slurm/).

| Script | Job | Purpose |
|---|---|---|
| `tr.py` | `run_tr.slurm` | Stage 1: train every dataset from COCO `yolov5s.pt` |
| `tr_merged.py` | `run_tr_merged.slurm` | Stage 2: train the two pooled datasets |
| `fine_tuning.py` | `run_fine_tuning.slurm` | Stage 3: fine-tune five targets from the strongest Stage-1 checkpoint (selected automatically: IR-Pot-Background) |
| `tr_threearm.py` | `run_threearm.slurm` | Laser / no-laser / sham retrain, 3 seeds each |
| `tr_ablation.py` | `run_tr_ablation.slurm` | Earlier four-condition laser ablation |
| `tr_cv.py` | `run_cv_v4uv.slurm` | Cross-burst vs in-burst cross-validation (v4 family) |
| `tr_stir_cv.py` | `run_stir_cv.slurm` | Stir-probe grouped CV: paper recipe vs frozen backbone vs the regularisation-sweep winner (`reg`), checkpoint every 10 epochs |
| `tr_stir_regularize.py` | `run_stir_regularize.slurm` | Stage 1 (`s1_` arms) and Stage 3 stir regularisation sweep (label smoothing, weight decay, shorter schedule, lower lr), checkpoint every 5 epochs |

Outputs go to `runs/` (git-ignored). Trained paper weights are on the [`paper-data-v1`](https://github.com/greatroboticslab/Irrigation-Laser-Yolo2/releases/tag/paper-data-v1) release.
