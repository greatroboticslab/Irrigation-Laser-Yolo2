# scripts/evaluation/

Tests run on trained models. Results are written to [`../../results/`](../../results/).

| Script | Purpose | Output |
|---|---|---|
| `eval_calibration.py` | Confident-error / calibration test on every Stage-1 model | `results/posthoc_calibration*` |
| `eval_stir_cv.py` | Scores every stir-CV checkpoint out-of-fold over the 56 originals | `runs/stir_cv/stir_cv_{oof,summary}.csv` |
| `eval_stir_regularize.py` | Rescores every sweep checkpoint with unsmoothed loss; flags arms whose validation loss no longer turns up | `results/stir_reg_{curves,summary}.csv` |
| `eval_laser_dependence.py` | mAP with the laser present vs inpainted out | `results/laser_dependence.csv` |
| `eval_overlay_leak.py` | mAP with the burned-in overlay present vs masked | `results/overlay_leak.csv` |
| `eval_card_leak.py` | mAP with the in-scene moisture cards present vs masked | `results/card_leak.csv` |
| `analyze.py` | Mean mAP of the last epochs for every run | text summary |
| `compare_selected.py` | Bar chart comparing selected datasets | PNG |

`eval_stir_cv.py` imports `eval_calibration.py`, so keep the two in the same folder.
