# scripts/synthetic/

Generators for synthetic stir-probe frames. **No synthetic frame is in any training set behind the reported results.** Their bounding boxes come from per-class spatial priors rather than detection, so treat them as labels by assumption.

| Script | Backend | Notes |
|---|---|---|
| `generate_synthetic_stir.py` | Google Imagen 4 (`imagen-4.0-generate-001`) via the Gemini API | needs `GEMINI_API_KEY` |
| `generate_stir_synthetic.py` | FLUX.1-schnell via diffusers | GPU; run with `slurm/run_generate_synthetic_gpu.slurm` |
