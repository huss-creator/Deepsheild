"""
Deterministic feature extraction for the FFT/DCT + SVM model.

Pipeline (each step fixed and versioned so results are reproducible):
    RGB face -> grayscale float32 -> resize to SxS (area/cubic) -> transform
    -> low-dimensional feature vector

The feature extractor MUST match the one used when the SVM was trained. Add a new
extractor to EXTRACTORS (and bump FEATURE_VERSION) if your SVM used different
features. Feature normalisation (StandardScaler) lives INSIDE the saved sklearn
Pipeline so it cannot drift from the classifier.
"""

import cv2
import numpy as np

FEATURE_VERSION = 'freq-features-1'


def _to_gray(face_rgb: np.ndarray, size: int) -> np.ndarray:
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    interp = cv2.INTER_AREA if max(gray.shape) > size else cv2.INTER_CUBIC
    return cv2.resize(gray, (size, size), interpolation=interp)


def azimuthal_fft(face_rgb: np.ndarray, size: int = 256, n_bins: int = 64) -> np.ndarray:
    """1-D radially averaged log power spectrum (azimuthal integral of the 2-D FFT).
    GAN/diffusion up-sampling tends to leave characteristic high-frequency structure here."""
    gray = _to_gray(face_rgb, size)
    window = np.outer(np.hanning(size), np.hanning(size)).astype(np.float32)   # reduce edge leakage
    spectrum = np.fft.fftshift(np.fft.fft2((gray - gray.mean()) * window))
    power = np.log1p(np.abs(spectrum) ** 2)
    yy, xx = np.indices(power.shape)
    radius = np.hypot(yy - size / 2, xx - size / 2)
    edges = np.linspace(0, size / 2, n_bins + 1)
    idx = np.clip(np.digitize(radius.ravel(), edges) - 1, 0, n_bins - 1)
    sums = np.bincount(idx, weights=power.ravel(), minlength=n_bins)
    counts = np.maximum(np.bincount(idx, minlength=n_bins), 1)
    return (sums / counts).astype(np.float64)


def dct_block_stats(face_rgb: np.ndarray, size: int = 256, n_bins: int = 64) -> np.ndarray:
    """Mean log |coefficient| at each of the 64 positions of the 8x8 block DCT."""
    gray = _to_gray(face_rgb, size)
    acc = np.zeros((8, 8), dtype=np.float64)
    blocks = 0
    for y in range(0, size - 7, 8):
        for x in range(0, size - 7, 8):
            acc += np.abs(cv2.dct(gray[y:y + 8, x:x + 8] - 128.0))
            blocks += 1
    return np.log1p(acc / max(blocks, 1)).ravel()


def fft_dct_combined(face_rgb: np.ndarray, size: int = 256) -> np.ndarray:
    """128-dim vector: 64 azimuthal-FFT bins followed by the 64 block-DCT statistics.
    Use feature_method='fft_dct_combined' for SVMs trained on both feature families."""
    return np.concatenate([azimuthal_fft(face_rgb, size, 64), dct_block_stats(face_rgb, size)])


EXTRACTORS = {'azimuthal_fft': azimuthal_fft, 'dct_block_stats': dct_block_stats,
              'fft_dct_combined': fft_dct_combined}


def extract(face_rgb: np.ndarray, method: str, size: int, n_bins: int = 64) -> np.ndarray:
    if method not in EXTRACTORS:
        raise ValueError(f'unknown feature_method {method!r}; available: {sorted(EXTRACTORS)}')
    fn = EXTRACTORS[method]
    feats = fn(face_rgb, size, n_bins) if method == 'azimuthal_fft' else fn(face_rgb, size)
    return feats.reshape(1, -1)
