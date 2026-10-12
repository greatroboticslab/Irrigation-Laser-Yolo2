# Irrigation Laser YOLO2

A YOLOv5-based computer vision pipeline for soil moisture sensor detection and classification. The model detects a cylindrical stir-probe moisture sensor inserted into a plant pot and classifies its moisture level by reading the **LED colour** emitted at the probe tip (red/orange = dry, blue-violet = wet).

---

## Project Overview

| Item | Details |
|------|---------|
| **Model** | YOLOv5m (medium) |
| **Task** | Object detection + moisture-level classification |
| **Classes** | Up to 11 moisture levels depending on dataset |
| **Cluster** | MTSU research cluster — RTX 2080Ti, CUDA 12.4, 32 GB RAM |
| **Framework** | PyTorch + Ultralytics YOLOv5 |

---

## Datasets

All datasets downloaded from [Roboflow Universe](https://universe.roboflow.com) (workspace: `robotics-lab-1` / `mtsu`).

| Dataset | Classes | Notes |
|---------|---------|-------|
| `soil-moisture-v4` | 11 | Primary dataset |
| `soil-moisture-v4-ir` | 11 | Infrared variant |
| `soil-moisture-v4-uv` | 11 | UV variant |
| `soil-moisture-ir` | 5 | IR sensor images |
| `soil-moisture-5sagf` | 5 | General sensor images |
| `soil_moisture_september` | 11 | September collection |
| `soil_moisture_stir_september` | 5 | Stir probe, 56 real images |
| `soil-moisture-ir-combined` | 5 | Merged: 5sagf + ir |
| `soil-moisture-v4-september-combined` | 11 | Merged: v4 + september |

> Datasets skipped during training: `soil-moisture-v3` (87 classes, unusable), individual component datasets that were merged.

---

## Training Pipeline

Three sequential training runs, all using YOLOv5m with AdamW optimizer, cosine LR schedule, 200 epochs, patience 100, image size 640.

### Run 1 — Baseline (`run_tr.slurm`)
Trains on all individual datasets from scratch using ImageNet pretrained weights.
```bash
sbatch run_tr.slurm
```
Output: `runs/train/downloaded/<dataset>/weights/best.pt`

### Run 2 — Merged Datasets (`run_tr_merged.slurm`)
Merges compatible dataset pairs then trains on the combined sets.
```bash
# Step 1: merge locally
python merge_datasets.py

# Step 2: submit training job
sbatch run_tr_merged.slurm
```
- **Merge 1**: `soil-moisture-5sagf` + `soil-moisture-ir` → `soil-moisture-ir-combined`
- **Merge 2**: `soil-moisture-v4` + `soil_moisture_september` → `soil-moisture-v4-september-combined`

Output: `runs/train/merged/<dataset>/weights/best.pt`

### Run 3 — Fine-Tuning (`run_fine_tuning.slurm`)
Fine-tunes all valid datasets starting from `soil-moisture-v4/weights/best.pt`.
```bash
sbatch run_fine_tuning.slurm
```
Output: `runs/fine-tuned/<dataset>/weights/best.pt`

---

## Results — Best Validation mAP@0.5

| Dataset | Run 1 | Run 2 | Fine-tune | Combined |
|---------|-------|-------|-----------|----------|
| soil-moisture-v4 | 0.9907 | 0.9896 | **0.9907** | — |
| soil-moisture-v4-ir | 0.9834 | 0.9869 | **0.9903** | — |
| soil-moisture-v4-uv | 0.9099 | **0.9950** | 0.9308 | — |
| soil-moisture-ir | 0.7459 | failed | 0.7135 | **0.8699** |
| soil-moisture-5sagf | 0.5518 | 0.5789 | 0.5537 | **0.8699** |
| soil_moisture_september | 0.5693 | 0.4094 | 0.3238 | **0.9430** |
| soil_moisture_stir_september | **0.4988** | 0.4184 | 0.4184 | — |

> Merging datasets produced the largest gains: IR Combined +13.7%, v4+September +37.4%.

---

## Data Augmentation — `soil_moisture_stir_september`

The stir probe dataset had only 56 images. An albumentations pipeline expanded it to **685 images**.

| Split | Originals | Copies each | Total |
|-------|-----------|-------------|-------|
| train | 39 | 13 | 546 |
| valid | 12 | 6 | 84 |
| test | 5 | 10 | 55 |

**Key design constraint**: hue shift capped at ±8° to preserve LED colour as the class signal.

Augmentations: horizontal/vertical flip, rotation ±20°, perspective warp, random crop+resize, shift-scale-rotate, brightness/contrast, Gaussian blur, motion blur, Gaussian noise, JPEG compression, HSV shift, RGB shift, random shadow, fog, downscale.

```bash
python augment_stir_september.py
```

---

## Synthetic Data Generation

Synthetic images for `soil_moisture_stir_september` generated using **Google Imagen 4** (`imagen-4.0-generate-001`) via the Gemini API, targeting 375 images (5 classes x 75).

```bash
python generate_synthetic_stir.py
```

Alternatively, generate locally on the MTSU cluster using Stable Diffusion 1.5:
```bash
sbatch run_generate_synthetic.slurm
```

Each synthetic image includes:
- Per-class LED colour matching real sensor behaviour
- Randomised scene variations (leaf coverage, lighting, camera angle, soil texture)
- Auto-annotated YOLO bounding boxes via HSV colour masking on the LED glow

---

## Analysis & Visualization

```bash
# Analyze training results (avg mAP last 5 epochs)
python analyze.py --out-dir first-run
python analyze.py --runs-root runs/fine-tuned --out-dir fine-tuned-results

# Plot training curves and comparison bar charts
python plot_results.py --out-dir first-run
python plot_results.py --runs-root runs/fine-tuned --out-dir fine-tuned-results

# Augmentation slide deck (PDF)
python make_augmentation_slides.py
# Output: plots/stir_september_augmentation_slides.pdf

# Original vs synthetic comparison slides (PDF)
python make_synthetic_slides.py
# Output: plots/synthetic_comparison_slides.pdf
```

---

## Project Structure

```
Irrigation-Laser-Yolo2/
├── data/training-data/downloaded/   # All datasets
├── runs/train/downloaded/           # Run 1 results
├── runs/train/merged/               # Run 2 results
├── runs/fine-tuned/                 # Run 3 results
├── plots/                           # Charts and slide deck PDFs
├── logs/                            # SLURM job logs
├── tr.py                            # Run 1 training script
├── tr_merged.py                     # Run 2 training script
├── fine_tuning.py                   # Run 3 fine-tuning script
├── merge_datasets.py                # Dataset merger
├── augment_stir_september.py        # Albumentations augmentation pipeline
├── generate_synthetic_stir.py       # Imagen 4 synthetic generation
├── generate_stir_synthetic.py       # Stable Diffusion synthetic generation
├── analyze.py                       # Results analyzer
├── plot_results.py                  # Training curve plotter
├── make_augmentation_slides.py      # Augmentation PDF slide deck
├── make_synthetic_slides.py         # Synthetic comparison PDF slide deck
├── run_tr.slurm                     # SLURM: Run 1
├── run_tr_merged.slurm              # SLURM: Run 2
├── run_fine_tuning.slurm            # SLURM: Run 3
└── run_generate_synthetic.slurm     # SLURM: synthetic generation
```

---

## Requirements

```bash
pip install -r requirements.txt
pip install albumentations google-genai pillow diffusers transformers accelerate
```

---

## Hardware

Training runs on the **MTSU research cluster**:
- GPU: NVIDIA RTX 2080Ti (11 GB VRAM)
- RAM: 32 GB
- CUDA: 12.4
- Wall time: 48h (training), 12h (synthetic generation)
- Python venv: `/projects/kp9e/Irrigation-Laser-Yolo2/venv`
