# scripts/data_prep/

Everything that creates or reshapes a dataset under `data/training-data/`.

| Script | Purpose |
|---|---|
| `download.py` | Download the Roboflow datasets (needs `ROBOFLOW_API_KEY`) |
| `merge_datasets.py` | Build the two pools, `soil-moisture-ir-combined` and `soil-moisture-v4-september-combined`, fixing alphabetical class order |
| `augment_stir_september.py` | Stir-probe augmentation pass 1 (Albumentations, hue capped at ±8°): 13 / 8 / 10 copies per original |
| `augment_stir_dataset.py` | Stir-probe augmentation pass 2: eight fixed, hue-free transforms per original |
| `laser_ablation_dataset.py` | Laser mask and inpainting helpers; builds the with/without-laser ablation sets |
| `build_nolaser_datasets.py` | Laser-removed and sham twins of trained datasets (three-arm retrain) |
| `build_overlay_masked.py` | Blanks the burned-in "Moisture: N" overlay |
| `build_card_masked.py` | Blanks the handwritten moisture cards in the pot-background captures |
| `build_cv_groups.py` | Groups near-duplicate frames into capture bursts by pixel similarity |
| `build_cv_folds.py` | Leave-one-burst-out folds with a matched random control (v4 family) |
| `build_stir_cv.py` | Grouped, class-stratified 5-fold split over the 56 stir-probe originals |

Datasets ready to train on are on the [`paper-data-v1`](https://github.com/greatroboticslab/Irrigation-Laser-Yolo2/releases/tag/paper-data-v1) release.
