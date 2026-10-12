# Irrigation Laser YOLO2

A YOLOv5s pipeline that reads a low-cost commercial soil-moisture stir probe by camera. The detector localises the probe tip and classifies the moisture level from the **colour of its tip LED** (red/orange = dry, blue-violet = wet), turning an unmodified consumer device into a non-contact moisture sensor.

---

## Downloads: data and trained weights

Data and weights are attached to the GitHub Release **[`paper-data-v1`](https://github.com/greatroboticslab/Irrigation-Laser-Yolo2/releases/tag/paper-data-v1)** rather than committed, so the repository stays small. Unzip both at the repository root; the archives already carry the paths the scripts expect.

| Asset | Size | Contents |
|---|---|---|
| `paper_datasets.zip` | 277 MB | The 9 datasets used in the paper, exactly as trained on (including the augmented stir-probe build), plus the capture-burst groupings and cross-validation fold manifests under `runs/cv/` |
| `paper_weights.zip` | 394 MB | `best.pt` and `last.pt` for all 15 paper runs, with each run's `opt.yaml`, `hyp.yaml` and `results.csv` |

```bash
unzip paper_datasets.zip   # -> data/training-data/downloaded/<dataset>/, runs/cv/
unzip paper_weights.zip    # -> runs/train/downloaded/, runs/train/merged/, runs/fine-tuned/, runs/train/stir-finetune/
```

`best.pt` is the checkpoint of best YOLOv5 fitness (0.1·mAP@0.5 + 0.9·mAP@0.5:0.95), not of best mAP@0.5; `last.pt` is the final epoch.

---

## Datasets

All collections come from [Roboflow Universe](https://universe.roboflow.com) (workspace `robotics-lab-1`, CC BY 4.0). The paper uses role-based display names:

| On disk | Name in the paper | Classes | Train / valid / test |
|---|---|---|---|
| `soil-moisture-v4` | UV-Soil-Only-Baseline | 11 | 246 / 70 / 35 |
| `soil-moisture-v4-uv` | UV-Soil-Only-Target | 11 | 142 / 40 / 20 |
| `soil-moisture-v4-ir` | IR-Soil-Only | 7 | 104 / 30 / 15 |
| `soil-moisture-ir` | IR-Pot-Background | 5 | 70 / 20 / 10 |
| `soil-moisture-5sagf` | UV-Pot-Background | 5 | 70 / 20 / 10 |
| `soil_moisture_september` | UV-Strawberry-Pot | 11 | 46 / 11 / 11 |
| `soil_moisture_stir_september` | Mixed Color and Background | 5 (levels 1, 3, 7, 8, 10) | 858 / 204 / 95 frames from 39 / 12 / 5 originals |
| `soil-moisture-ir-combined` | IR-Combined | 5 | pool of `5sagf` + `ir` |
| `soil-moisture-v4-september-combined` | UV-Combined | 11 | pool of `v4` + `september` |

`soil-moisture-v3` (87 raw voltage classes) is excluded.

**Class order.** Roboflow exports names alphabetically, so in the 11-class sets index 2 is level 10. `merge_datasets.py` remaps indices before pooling.

**Stir-probe build.** The 56 original photographs were expanded offline in two passes. Each split was expanded only from its own originals:

| Split | Originals | Pass 1 (`augment_stir_september.py`) | Pass 2 (`augment_stir_dataset.py`) | Total |
|---|---|---|---|---|
| train | 39 | 13 copies each | 8 copies each | 858 |
| valid | 12 | 8 | 8 | 204 |
| test | 5 | 10 | 8 | 95 |

- **Pass 1** uses Albumentations with hue shift capped at ±8°, because LED colour is the class signal.
- **Pass 2** applies eight fixed transforms (flips, 90/180/270° rotations, brightness/contrast, blur with noise, combined), none of which alters hue.
- **Independence:** the copies are not independent samples. The 56 originals also fall into 42 capture bursts, a few of which cross the shipped split boundaries.

---

## Training protocol (three stages)

All runs use YOLOv5s from COCO weights (`yolov5s.pt`), image size 640, AdamW, cosine LR, 200 epochs, patience 100. `hyp.stir-soil.yaml` is applied to the overhead datasets (`stir`, `v4`, `v4-uv`); others use the default hyperparameters.

| Stage | Script / job | What it does | Output |
|---|---|---|---|
| 1. Baselines | `tr.py` / `run_tr.slurm` | Trains each dataset independently from COCO | `runs/train/downloaded/<ds>/` |
| 2. Pooling | `merge_datasets.py`, `tr_merged.py` / `run_tr_merged.slurm` | Trains the two label-compatible pools | `runs/train/merged/<ds>/` |
| 3. Fine-tuning | `fine_tuning.py` / `run_fine_tuning.slurm` | Fine-tunes five targets from the strongest Stage-1 checkpoint, selected automatically. That checkpoint is **IR-Pot-Background (`soil-moisture-ir`, mAP@0.5 = 0.804)** | `runs/fine-tuned/<ds>/` |

The Stage-3 targets are IR-Soil-Only, UV-Soil-Only-Target, Mixed Color and Background, IR-Combined and UV-Combined. A further stir-probe run (`stir5`, `runs/train/stir-finetune/`) was trained from the Stage-1 `soil-moisture-v4` checkpoint with `hyp.stir-soil.yaml`.

### Results (best-epoch validation mAP@0.5, from `results.csv`)

| Dataset | Stage 1 | Stage 2 (pool) | Stage 3 |
|---|---|---|---|
| UV-Soil-Only-Baseline | 0.760 | — | — |
| UV-Soil-Only-Target | 0.612 | — | 0.912 |
| IR-Soil-Only | 0.550 | — | 0.987 |
| IR-Pot-Background | 0.804 | — | — |
| UV-Pot-Background | 0.587 | — | — |
| UV-Strawberry-Pot | 0.716 | — | — |
| Mixed Color and Background | 0.683 | — | 0.711 |
| IR-Combined | — | 0.591 | 0.995 |
| UV-Combined | — | 0.942 | 0.955 |
| stir5 | — | — | 0.625 |

These are peak-epoch values read from the validation log, and for the stir-probe runs they precede a collapse (see below). Validation splits of the soil-only family share capture bursts with training, so their figures measure in-session readout, not cross-session generalisation.

---

## Overfitting and validation analyses

| Analysis | Script(s) | Result files |
|---|---|---|
| Loss-curve overfitting diagnostic | `make_overfitting_figure.py`, `make_training_curves_figure.py` | `updated results/overfitting_curves.png`, `training_curves.*` |
| Post-training calibration (confident-error test) | `eval_calibration.py` | `updated results/posthoc_calibration*.{csv,png}` |
| Stir-probe test split | `val.py` | `updated results/stir_test_scores.csv` |
| Capture-burst grouping and leakage | `build_cv_groups.py`, `build_cv_folds.py`, `tr_cv.py` / `run_cv_v4uv.slurm` | `runs/cv/*_groups.csv`, `*_folds.csv` |
| **Stir-probe grouped 5-fold CV** | `build_stir_cv.py`, `tr_stir_cv.py`, `eval_stir_cv.py` / `run_stir_cv.slurm` | `updated results/stir_cv_{summary,oof,analysis}.csv` |
| Laser / overlay / card leakage ablations | `laser_ablation_dataset.py`, `build_nolaser_datasets.py`, `tr_threearm.py` / `run_threearm.slurm`, `eval_laser_dependence.py`, `eval_overlay_leak.py`, `eval_card_leak.py` | `updated results/threearm_summary.csv`, `laser_dependence.csv`, `overlay_leak.csv`, `card_leak.csv`, `LASER_ABLATION_RESULTS.md` |

**Findings in brief**

- **Shipped split:** only the stir-probe collection overfits. Validation classification loss bottoms at epoch 54 (Stage 1) and 57 (Stage 3) and ends 1.6–1.7× above that minimum. mAP@0.5 decays from 0.683 to 0.440 and from 0.711 to 0.480, and the final Stage-1 checkpoint makes confident (>0.9) errors. The other runs were still improving at epoch 200.
- **Grouped CV** over all 56 stir originals (42 bursts, five folds, three seeds, the Stage-3 recipe on a 100-epoch schedule):
  - Out-of-fold top-1 accuracy is **0.83** (95% burst-bootstrap interval 0.72–0.93) and mean held-out mAP@0.5 is **0.778**.
  - There is **no late-epoch decay** and calibration is good (ECE 0.063).
  - Freezing the backbone fails (0.31).
  - Remaining weaknesses: spread across folds (sd 0.165) and level 1 (0.50).

Reproduce the CV on the cluster:

```bash
sbatch run_stir_cv.slurm   # STIR_CV_SEEDS=0 sbatch run_stir_cv.slurm for a 10-run pilot
```

---

## Synthetic data

`generate_synthetic_stir.py` (Imagen 4 via the Gemini API) and `generate_stir_synthetic.py` (Stable Diffusion 1.5, `run_generate_synthetic.slurm`) generate labelled synthetic stir-probe frames. **No synthetic frames are in any training set behind the reported results.** Their bounding boxes come from per-class spatial priors rather than detection, so they are labels by assumption.

---

## Credentials

No keys are stored in the code. Set them in the environment before running:

```bash
export ROBOFLOW_API_KEY=...   # download.py, main.py
export GEMINI_API_KEY=...     # generate_synthetic_stir.py
export HUGGING_FACE_HUB_TOKEN=...   # run_generate_synthetic.slurm
```

---

## Requirements and cluster

```bash
pip install -r requirements.txt
pip install albumentations google-genai pillow diffusers transformers accelerate
```

Jobs target the MTSU Hamilton cluster (CUDA 12.4, venv at `/projects/kp9e/Irrigation-Laser-Yolo2/venv`). The `a100` partition requires `a100_users` membership; the CV jobs use `research`. Recorded runs used torch 2.6.

Locally with torch ≥ 2.6, our checkpoints load with `weights_only=False`. The evaluation scripts patch this themselves; for `train.py`, set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.
