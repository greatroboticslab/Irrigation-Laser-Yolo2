# scripts/figures/

Figures for the paper and slide decks. Paper figures go to `results/ieee/IEEE_LaTeX/figs/` (local manuscript folder); review copies go to `results/`.

| Script | Produces |
|---|---|
| `make_overfitting_figure.py` | Train vs validation classification loss (paper Fig. 4) |
| `make_best_map_figure.py` | Best-mAP@0.5 bars, Stage 1 (paper Fig. 5) and Stage 3 (paper Fig. 6), drawn at column width |
| `make_training_curves_figure.py` | Validation mAP curves for the reported runs |
| `make_threearm_figure.py` | Three-arm laser retrain bars (paper Fig. 8) |
| `make_confusion_figure.py` | Count-based confusion matrix for the IR-Combined model |
| `make_feature_map_figure.py` | Backbone feature maps P1–P5 |
| `make_dataset_overview_figure.py` | One image per dataset |
| `make_led_appearance_figure.py` | LED appearance at each moisture level |
| `generate_pipeline_figure.py` | Training-pipeline diagram |
| `make_training_infographic.py` | Excalidraw protocol diagram |
| `plot_results.py`, `plot_finetuned.py` | Per-dataset training curves and comparison bars |
| `make_augmentation_slides.py`, `make_synthetic_slides.py`, `make_flux_slides.py`, `make_gemini_vs_raw_slides.py` | PDF slide decks (augmentation, synthetic samples) |
| `build_docx.js` | Converts the Markdown manuscript draft to `.docx` (Node, `docx` package) |
