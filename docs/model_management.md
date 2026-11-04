# Model management

## How models are wired

`trained_models/models.json` (or env `*_MODEL_PATH`) describes each model: file, framework, input size, normalisation,
output mode, `fake_index`, threshold, version, SHA-256, and training/evaluation metadata. Nothing is hard-coded and no weights are
shipped. `python manage.py check_models` loads each model and runs a smoke inference; statuses are `AVAILABLE`, `NOT_CONFIGURED`,
`DISABLED`, `FAILED`, `TIMEOUT`.

## Versions and reproducibility

Every `ModelResult` links to a `ModelVersion (key, version, weights_sha256)` row and stores preprocessing version, threshold version,
score type and inference time. Any stored result can therefore be tied to the exact weights that produced it.
Production requires a checksum (`MODEL_REQUIRE_CHECKSUM`), so a swapped or corrupted file is refused at load.

## Deploying a new version

1. Put the new file next to the old one (e.g. `xception/model-v2.onnx`); compute `sha256sum`.
2. Edit `models.json`: new `file`, `version`, `sha256`, evaluation metadata. Keep the previous block in your deployment notes.
3. `python manage.py check_models` on the target machine.
4. `python manage.py evaluate_models ...` on your labelled validation/test split; commit the resulting `ensemble_config.json`
   (a new model changes the score distribution, so an old validated ensemble no longer applies — remove it until re-validated; the
   system then falls back to the clearly-labelled unvalidated mean).
5. Restart web and worker containers (the registry is loaded per process).

## Rollback

Rollback is a configuration change, not a migration: restore the previous `models.json` block (file, version, sha256) and the previous
`ensemble_config.json`, run `check_models`, restart web+worker. Old results keep their `ModelVersion` links, so history remains
interpretable. Keep at least the last known-good weights file on the volume.

## Monitoring and drift

`python manage.py model_report [--days 7] [--json] [--fail-on-alert]` summarises, per model version, over the current and previous window:
volume, failure rate, score mean/std/percentiles and the share of runs flagged manipulated. Alerts: flagged-rate shift ≥ 0.15, mean-score shift
≥ 0.10 (both windows ≥ 30 runs), failure rate ≥ 5%. Prometheus exposes `deepshield_model_failures_total{model}`.

What this **can** tell you: something changed (model, input population, camera/codec mix, attack mix) or a model is failing.
What it **cannot** tell you: accuracy — there are no labels in production. Treat an alert as a trigger to re-evaluate on a fresh labelled sample.

Recommended loop (not automated): run `model_report` daily (cron/scheduler); on alert, sample recent analyses *with consent*, label them,
run `evaluate_models`, and decide on recalibration or rollback.

## Evaluation protocol (what `evaluate_models` enforces)

* Splits: train / validation / test / external; **identity-disjoint** and duplicate-hash checks (leakage audit) before any metric.
* Thresholds and ensemble weights are tuned on **validation only**, then frozen for test and external.
* Reports AUC, accuracy, precision/recall/F1, FPR/FNR, per-split and per-model; stacking is used only if it beats the mean on validation.
* Calibration is stated only if fitted on validation; otherwise scores are labelled *uncalibrated*.

The numbers depend on your datasets and weights; the toolkit is tested on synthetic data only.

## Face detector

`FACE_DETECTOR=yunet` (recommended; needs `YUNET_MODEL_PATH`, provides confidence and eye landmarks) or `haar` (no extra
weights; stricter cascade voting and eye-structure validation reduce false detections). Haar remains the default for deployments
that do not have YuNet weights. Configure `FACE_HAAR_VALIDATE_EYES`, `FACE_HAAR_MIN_NEIGHBORS`, and `FACE_MIN_AREA_RATIO` to tune
the precision/recall trade-off. A detector cannot guarantee correct face identity on every image; review uncertain or low-resolution
face crops before relying on their scores.

## Organika SDXL detector

The `sdxl_detector` registry entry is prepared but disabled until an ONNX export is supplied. Its output is two raw logits:
class `0` is `artificial`, class `1` is `human`; the configured fake index is `0`. See
[the SDXL model card](../trained_models/sdxl_detector/README.md) for preprocessing, license, export, checksum, and validation steps.
Do not treat the upstream validation metrics or the default threshold as independently verified production performance.
