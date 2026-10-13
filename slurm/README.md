# slurm/

Batch jobs for the MTSU Hamilton cluster. Each job activates `/projects/kp9e/Irrigation-Laser-Yolo2/venv`, `cd`s to the project root and runs scripts from [`../scripts/`](../scripts/). Logs go to `/projects/kp9e/Irrigation-Laser-Yolo2/logs/`.

```bash
sbatch slurm/run_tr.slurm            # submit from the project root
```

| Job | Runs |
|---|---|
| `run_tr.slurm` | Stage 1 baselines, then the analysis and plots |
| `run_tr_merged.slurm` | Stage 2 pooled datasets |
| `run_fine_tuning.slurm` | Stage 3 fine-tuning, then the analysis and plots |
| `run_threearm.slurm` | Three-arm laser retrain |
| `run_tr_ablation.slurm` | Four-condition laser ablation |
| `run_cv_v4uv.slurm` | Burst groups, folds and cross-burst CV (v4 family; set `CV_DATASET`) |
| `run_stir_cv.slurm` | Stir-probe grouped CV and out-of-fold scoring |
| `run_stir_regularize.slurm` | Stage 3 stir regularisation sweep and rescoring |
| `run_generate_synthetic*.slurm`, `run_stir_synthetic.slurm` | Synthetic generation and its follow-up training |
| `check_python.slurm`, `test_cuda.slurm` | Environment checks |

The `a100` partition needs `a100_users` membership; the CV jobs use `research`.
