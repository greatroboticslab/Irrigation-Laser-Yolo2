# results/

Result tables and figures behind the paper. They are produced by [`../scripts/evaluation/`](../scripts/evaluation/) and [`../scripts/figures/`](../scripts/figures/).

| File | Contents |
|---|---|
| `overfitting_curves.png`, `training_curves.{png,pdf}` | Loss and mAP curves for the reported runs |
| `posthoc_calibration.csv`, `posthoc_calibration_preds.csv`, `posthoc_calibration.png` | Calibration test, summary and per-image |
| `stir_test_scores.csv` | Stir-probe best/last checkpoints on the validation and test splits |
| `stir_cv_summary.csv`, `stir_cv_oof.csv`, `stir_cv_analysis.csv` | Stir-probe grouped 5-fold CV: learning curve, per-image out-of-fold predictions, error bars / per-level / calibration |
| `threearm_summary.csv`, `threearm_retrain_comparison.png` | Laser / no-laser / sham retrain |
| `laser_dependence.csv`, `overlay_leak.csv`, `card_leak.csv` | Inference-time leakage checks |
| `LASER_ABLATION_RESULTS.md` | Write-up of the laser ablation |
| `feature_map_evolution.pdf` | Feature-map figure |
| `slides/` | Generated slide decks |

Trained runs (`runs/`) are not committed; the paper weights are on the [`paper-data-v1`](https://github.com/greatroboticslab/Irrigation-Laser-Yolo2/releases/tag/paper-data-v1) release.
