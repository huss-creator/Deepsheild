"""
forensics/__init__.py — DeepShield Forensic Analysis Package

This package contains image forensics algorithms that provide
supporting evidence for the AI deepfake detection results.

Modules:
    ela           — Error Level Analysis (JPEG recompression artifacts)
    frequency     — FFT Frequency-domain analysis
    noise_analysis — Noise residual extraction and analysis
    metadata      — EXIF metadata extraction
    evidence      — Evidence aggregation and risk scoring

IMPORTANT:
    These modules provide SUPPORTING EVIDENCE only.
    They do NOT independently prove that an image is fake or authentic.
    Results must be interpreted alongside AI model predictions.
"""
