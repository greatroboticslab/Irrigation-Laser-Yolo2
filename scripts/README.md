# scripts/

Project code, grouped by stage of the workflow. YOLOv5 itself (`train.py`, `val.py`, `models/`, `utils/`) stays at the repository root.

**Run every script from the repository root**, e.g. `python scripts/training/tr.py`. Paths inside the scripts (`data/…`, `runs/…`, `results/…`) are relative to the root.

| Folder | What it holds |
|---|---|
| [`data_prep/`](data_prep/) | Download, merge, augment and ablate datasets; build cross-validation folds |
| [`training/`](training/) | Training drivers for the three-stage protocol and the follow-up experiments |
| [`evaluation/`](evaluation/) | Post-training tests: calibration, cross-validation scoring, leakage checks |
| [`figures/`](figures/) | Paper figures, slide decks and result plots |
| [`synthetic/`](synthetic/) | Synthetic stir-probe image generators (not used in any reported result) |

Cluster jobs that call these scripts live in [`../slurm/`](../slurm/).
