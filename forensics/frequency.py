"""
forensics/frequency.py — FFT frequency-domain analysis

What it measures: narrow spectral peaks (periodic patterns) standing out from the image's
radially-averaged spectrum. Up-sampling layers in some generators leave such peaks.
Also reports the share of energy in high frequencies (descriptive only).

What it CANNOT show: lots of natural content, sharpening and — above all — JPEG
compression produce periodic structure. JPEG's 8x8 block grid creates peaks at multiples
of N/8, which coincide with the period-2/4 peaks a generator would leave. For JPEG input
those frequencies are excluded, so generator artefacts at exactly those frequencies
cannot be seen in JPEG files. Absence of peaks proves nothing.

Fixes vs the previous version: no resampling to 512x512 (that destroyed the very
up-sampling traces being sought) — a native-resolution central window is analysed;
Hann windowing; local-median spectral background with robust (MAD) peak detection; the
unreachable `high_freq_ratio > 0.35` rule is gone.

Benchmark: see `manage.py benchmark_forensics` / docs/forensic_validation.md. Levels are
capped at MEDIUM unless the benchmark supports more.

Parameters (settings): FREQ_WINDOW (512), FREQ_PEAK_Z (6.0).
"""

import logging

import cv2
import numpy as np
from scipy.ndimage import median_filter

from forensics.common import central_crop, cfg, open_rgb, save_png

logger = logging.getLogger('deepshield.forensics')
LIMITATIONS = ('JPEG compression, sharpening and natural periodic textures also create spectral '
               'peaks, and generator artefacts at JPEG-grid frequencies are invisible in JPEG files. '
               'Supporting indicator only; absence of peaks proves nothing.')


def find_spectral_peaks(gray: np.ndarray, is_jpeg: bool, z_thr: float = 6.0):
    """Return (peaks [(z, dy, dx)], high_freq_ratio, log_power_spectrum)."""
    g = gray.astype(np.float64)
    h, w = g.shape
    g = (g - g.mean()) * np.outer(np.hanning(h), np.hanning(w))
    power = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    L = np.log1p(power)
    cy, cx = h // 2, w // 2
    yy, xx = np.indices(L.shape)
    r = np.hypot(yy - cy, xx - cx).astype(int)
    # Background = local 2-D median of the log spectrum. (A radial-mean model absorbs peaks that
    # fill a whole radius bin, e.g. a Nyquist checkerboard in the spectrum corners.)
    res = L - median_filter(L, size=11, mode='reflect')
    mad = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-9
    z = res / mad
    z[r < 6] = 0                                              # DC / lowest frequencies
    if is_jpeg:
        # exclude JPEG 8x8 grid frequencies (multiples of N/8 along either axis)
        dy, dx = np.abs(yy - cy), np.abs(xx - cx)
        by, bx = max(h // 8, 1), max(w // 8, 1)
        near = lambda d, b: (d % b <= 1) | (d % b >= b - 1)
        z[(near(dx, bx) & near(dy, by)) | (near(dx, bx) & (dy <= 1)) | (near(dy, by) & (dx <= 1))] = 0
    z32 = z.astype(np.float32)                                # same dtype on both sides of ==
    local_max = z32 == cv2.dilate(z32, np.ones((5, 5), np.uint8))
    ys, xs = np.nonzero((z32 > z_thr) & local_max)
    peaks = sorted(((float(z[y, x]), int(y - cy), int(x - cx)) for y, x in zip(ys, xs)), reverse=True)
    n = min(h, w)
    high = power[r >= 0.4 * (n / 2)].sum() / max(power.sum(), 1e-12)
    return peaks, float(high), L


def run_frequency_analysis(image_path, save_folder):
    result = {'freq_image_path': None, 'freq_score': 0.0, 'freq_level': 'UNAVAILABLE',
              'freq_low_pattern': 'N/A', 'freq_high_pattern': 'N/A',
              'freq_interpretation': 'Frequency analysis was not available.',
              'success': False, 'error': '', 'details': {}}
    try:
        rgb, fmt = open_rgb(image_path)
        win = central_crop(rgb, int(cfg('FREQ_WINDOW', 512)))
        if min(win.shape[:2]) < 128:
            result.update(success=True, freq_interpretation='The image is too small (<128 px) for '
                                                            'reliable frequency analysis.')
            return result
        gray = cv2.cvtColor(win, cv2.COLOR_RGB2GRAY)
        peaks, high_ratio, L = find_spectral_peaks(gray, fmt == 'JPEG', float(cfg('FREQ_PEAK_Z', 6.0)))
        n = len(peaks)
        level = 'LOW' if n == 0 else 'MEDIUM'                 # capped: see module docstring
        vis = (255 * (L - L.min()) / max(L.max() - L.min(), 1e-9)).astype(np.uint8)
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
        cy, cx = vis.shape[0] // 2, vis.shape[1] // 2
        for _, dy, dx in peaks[:10]:
            cv2.circle(vis, (cx + dx, cy + dy), 6, (255, 0, 0), 1)
        result.update(
            freq_image_path=save_png(vis, save_folder, 'freq'), freq_score=float(min(100, 20 * n)),
            freq_level=level, success=True,
            freq_low_pattern='Dominant' if high_ratio < 0.05 else 'Moderate',
            freq_high_pattern=f'{high_ratio * 100:.2f}% of energy above 0.4 x Nyquist (descriptive)',
            freq_interpretation=(f'{n} narrow spectral peak(s) stood out from the radial spectrum '
                                 f'({"JPEG grid frequencies excluded" if fmt == "JPEG" else "no grid exclusion"}). '
                                 + LIMITATIONS),
            details={'peak_count': n, 'top_peaks': [{'z': round(z, 1), 'dy': dy, 'dx': dx}
                                                     for z, dy, dx in peaks[:5]],
                     'high_freq_energy_ratio': round(high_ratio, 5), 'window': list(win.shape[:2]),
                     'jpeg_grid_excluded': fmt == 'JPEG', 'reliability': 'low', 'limitations': LIMITATIONS})
    except Exception:
        logger.exception('frequency analysis failed')
        result['error'] = 'Frequency analysis failed.'
    return result
