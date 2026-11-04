"""
forensics/resampling.py — Resampling / Interpolation Artifact Detection

What is resampling detection?
    Whenever part of an image (or the whole image) is resized, rotated,
    or stretched, the software has to invent new pixel values in between
    the original ones — it interpolates (bilinear, bicubic, etc). That
    interpolation makes each new pixel a weighted average of its
    neighbours, which introduces small, PERIODIC correlations between
    pixels that a native, never-resampled camera image does not have.

    This is exactly the kind of thing you'd expect from:
        - A region pasted in at a different scale than it was
          originally captured (a classic splicing tell).
        - A face/object that was warped to fit before compositing.
        - An image that's been upscaled to look higher-resolution
          than the source.

Algorithm (second-derivative periodicity, after Gallagher 2005):
    1. Convert to grayscale.
    2. Take the second derivative of the image (Laplacian) — resampling
       correlations are far more visible in the derivative signal than
       in raw pixel values.
    3. Take the 2D FFT magnitude spectrum of that derivative signal.
    4. A native (never-resampled) image has a fairly smooth,
       featureless derivative spectrum. A resampled image shows sharp,
       regularly-spaced PEAKS in that spectrum, because the
       interpolation kernel repeats at a fixed spatial period tied to
       the resampling ratio.
    5. Score = how much the spectrum's peak energy exceeds its local
       background, outside the low-frequency DC region (which every
       image has regardless of resampling).

VALIDATION NOTE: on real photographs with synthetic resampling this detector separates
resampled from untouched images only weakly (AUC ~0.65; see docs/forensic_validation.md),
so the level is capped at MEDIUM. The image is analysed at NATIVE resolution (the previous
version resized to 512x512 first, which destroys the periodic traces it looks for); for
JPEG input the 8x8-grid frequencies are excluded.

IMPORTANT DISCLAIMER:
    JPEG compression, sharpening filters, and some camera demosaicing
    algorithms can also introduce periodic patterns. This module flags
    a statistical signature, not a certainty. Supporting evidence only.

Requirements:
    pip install numpy Pillow
"""

import logging
import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d

from forensics.common import central_crop, cfg, open_rgb, save_png

logger = logging.getLogger('deepshield.forensics')
LIMITATIONS = ('Ordinary editing (resizing, rotating, screenshots), sharpening and JPEG '
               'compression also create periodic correlations; regular textures can too. '
               'Supporting indicator only; not proof of manipulation.')


def _axis_peak_ratio(gray, axis, skip_jpeg_grid):
    d2 = np.abs(np.diff(gray, 2, axis=axis))
    s = d2.mean(axis=1 - axis)
    s = s - uniform_filter1d(s, 15)
    n = len(s)
    sp = np.abs(np.fft.rfft(s * np.hanning(n)))
    f = np.arange(len(sp))
    keep = f >= int(0.03 * n)
    if skip_jpeg_grid:
        for k in range(1, 5):
            b = int(round(k * n / 8))
            keep &= ~((f >= b - 1) & (f <= b + 1))
    v = sp[keep]
    return float(v.max() / (np.median(v) + 1e-9)) if v.size else 0.0


def _resampling_score(gray_array, skip_jpeg_grid=False):
    """Core scoring on a 2-D float array. Returns (score_0_100, peak_ratio, spectrum_vis).
    peak_ratio = strongest periodic peak / median of the second-derivative variance spectrum."""
    g = np.asarray(gray_array, dtype=np.float64)
    ratio = max(_axis_peak_ratio(g, 0, skip_jpeg_grid), _axis_peak_ratio(g, 1, skip_jpeg_grid))
    score = float(min(100.0, max(0.0, (ratio - 6.0) * 10.0)))
    lap = cv2.Laplacian(g, cv2.CV_64F)
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(lap))))
    vis = (255 * (mag - mag.min()) / max(mag.max() - mag.min(), 1e-9)).astype(np.uint8)
    return score, ratio, vis


def run_resampling_analysis(image_path, save_folder):
    result = {'resampling_image_path': None, 'resampling_score': 0.0, 'resampling_level': 'UNAVAILABLE',
              'resampling_peak_ratio': 0.0, 'resampling_interpretation': 'Resampling analysis was not available.',
              'success': False, 'error': '', 'details': {}}
    try:
        rgb, fmt = open_rgb(image_path)
        win = central_crop(rgb, int(cfg('RESAMPLE_WINDOW', 512)))
        if min(win.shape[:2]) < 96:
            result.update(success=True, resampling_interpretation='Image too small (<96 px) for resampling analysis.')
            return result
        gray = cv2.cvtColor(win, cv2.COLOR_RGB2GRAY).astype(np.float64)
        score, ratio, vis = _resampling_score(gray, skip_jpeg_grid=(fmt == 'JPEG'))
        level = 'MEDIUM' if ratio >= float(cfg('RESAMPLE_RATIO_THRESHOLD', 12.0)) else 'LOW'   # capped
        result.update(
            resampling_image_path=save_png(vis, save_folder, 'resample'), resampling_score=round(score, 2),
            resampling_level=level, resampling_peak_ratio=round(ratio, 3), success=True,
            resampling_interpretation=(
                f'Strongest periodic peak is {ratio:.1f}x the median of the derivative-variance spectrum '
                + ('(periodic structure present). ' if level == 'MEDIUM' else '(no marked periodic structure). ')
                + LIMITATIONS),
            details={'peak_ratio': round(ratio, 3), 'window': list(win.shape[:2]),
                     'jpeg_grid_excluded': fmt == 'JPEG', 'reliability': 'low', 'limitations': LIMITATIONS})
    except Exception:
        logger.exception('resampling analysis failed')
        result['error'] = 'Resampling analysis failed.'
    return result
