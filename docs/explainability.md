# Explainability (Grad-CAM) — one method per model

Predictions run on each model's configured backend. Heat-maps need gradients, which ONNX Runtime cannot give, so on the first
explanation the **deployed `.onnx` itself is converted to a differentiable PyTorch module** (onnx2torch). Same weights by
construction: **no `.pth` and no retraining are needed.** The conversion is checked against ONNX Runtime on 3 probe images
before it is used (max logit difference <= 2e-3); if it differs, the heat-map is refused. Loaded PyTorch models use an isolated
copy of their own weights; an optional separately pinned state-dict is parity-checked the same way.

| Model | How it scores | Explanation used |
|---|---|---|
| EfficientNet-B4, Xception | conv → global-avg-pool → linear | **Grad-CAM** on the last conv block (`layer4`, `bn2`, `act4`) |
| ViT-B/16 | patch tokens + attention | **Grad-CAM on the last block's patch tokens** (14×14); alternatives: Grad-CAM++, attention rollout |
| Organika SDXL Detector | Swin Transformer | **Grad-CAM** on the final normalization output (7×7); supports sequence-token and spatial-map layouts |
| FFT/DCT + SVM | hand-made frequency features | **Frequency-band ablation** (Grad-CAM is not defined: no conv layers, no gradient). Each band/DCT position is replaced by the training mean; the fall in the "manipulated" score is drawn on the 2-D spectrum / 8×8 DCT grid |

Grad-CAM++ is available for the CNNs and Swin model, and attention rollout for ViT. Direct PyTorch state-dict models use a
separate copy of their loaded weights for explanations; TorchScript models need a matching configured explanation checkpoint.
`evaluate_explanations` picks the most faithful method on your data. The FFT/DCT + SVM model uses frequency-band ablation instead:
Grad-CAM is not mathematically applicable to a non-differentiable feature/SVM pipeline.

## Setup (once, on the machine that runs the site)
```bash
pip install -r requirements.txt                           # includes CPU-capable Grad-CAM dependencies
python manage.py check_models
```
The first heat-map after a restart takes a few seconds longer (conversion; ViT ~5-10 s); set `AI_MODEL_TIMEOUT=60`.
Optional: if you do have the training notebooks' `*_state_dict.pth`, `python manage.py attach_explain key=path.pth ...` makes the
explanation use those weights instead (SHA-256 pinned + parity checked).

## Choose the method with evidence
```bash
python manage.py evaluate_explanations --images D:\faces_test --n 60 --out reports/explain
```
Deletion test: blur the model's top-20 % cells and measure how far the *deployed* model's manipulated score falls, against a
control that blurs the same number of random cells (5 repeats). Gain = drop − random drop (± 95 % CI). Only images the model itself
calls manipulated are used. Put the winner in `models.json` → `explain.method` (or `attach_explain --method vit_b16=rollout ...`).

## Checks that are automated (`detector/tests_explain.py`)
- all six image architectures × their supported methods give a finite map in [0, 1] with the expected grid (7×7, 10×10, 12×12, 14×14, 32×32);
- model-randomisation sanity check (Adebayo et al. 2018): a re-initialised network gives a different map;
- ONNX conversion parity: the five original training architectures are checked against their source PyTorch maps
  (corr > 0.999); all converted models are also refused if they disagree with ONNX Runtime. SDXL sequence/spatial feature layouts
  and class mapping have dedicated regression tests;
- optional `.pth` gate: wrong weights, wrong checksum, path outside `MODEL_DIR` → "unavailable" with a reason;
- FFT/DCT explainers render and return per-band contributions.

## What to say in the report
A heat-map shows where the model's score is sensitive; it is not proof of manipulation and not a ground-truth localisation
(for the ViT it is coarser, 14×14). Report the deletion-test table from `evaluate_explanations` as the quantitative evidence.
Not run here: real trained weights (only random-init networks were available), so the faithfulness numbers must come from your run.
