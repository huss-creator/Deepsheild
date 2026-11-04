"""
forensics/ela.py — Error Level Analysis (ELA)

What it measures: how much each region of a JPEG changes when re-compressed once at a
fixed quality. Regions with a different compression history (e.g. pasted from another
source) can respond differently from the rest of the image.

What it CANNOT show: ELA is NOT a manipulation classifier and NOT proof of anything.
Error level also rises with texture/edges and varies with the original save quality.
Only applicable to JPEG input: PNG/other lossless files have no JPEG history to analyse,
so ELA is reported as not applicable instead of producing a meaningless score.

Benchmark (see `manage.py benchmark_forensics`, docs/forensic_validation.md): on real
photographs with synthetic splices the compact-region statistic used here separates
spliced from untouched images only weakly (AUC ~0.65), so the level is deliberately
capped at MEDIUM and reported as a low-reliability visual aid.

Parameters (Django settings, all optional): ELA_QUALITY (90), ELA_SCALE ('auto' or a
number), ELA_MODE ('color'|'gray'), ELA_BLOCK (8), ELA_REGION_FRACTION (0.08),
ELA_REGION_CONTRAST (2.0).

Output keys are unchanged: ela_image_path, ela_score, ela_level, ela_interpretation,
success, error — plus `details` with the raw statistics.
"""

import io
import logging

import cv2
import numpy as np
from PIL import Image

from forensics.common import cfg, open_rgb, save_png

logger = logging.getLogger('deepshield.forensics')

LIMITATIONS = ('ELA responds to texture, edges and the original JPEG quality as well as to editing. '
               'It is a visual aid and supporting indicator only; it is not proof of manipulation '
               'and the absence of an anomaly does not show an image is untouched.')


def _region_stats(err: np.ndarray, block: int):
    """Largest compact high-error region: (area_fraction_of_image, contrast_vs_median)."""
    h, w = err.shape
    hb, wb = h // block, w // block
    if hb < 4 or wb < 4:
        return 0.0, 0.0
    blk = err[:hb * block, :wb * block].reshape(hb, block, wb, block).mean(axis=(1, 3)).astype(np.float32)
    sm = cv2.blur(blk, (3, 3))
    med = float(np.median(sm))
    mask = (sm > 1.6 * max(med, 0.3)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))   # drop thin edge lines
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best_area, best_contrast = 0, 0.0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_contrast = float(blk[lab == i].mean() / max(med, 1e-6))
    return best_area / float(hb * wb), best_contrast


def run_ela_analysis(image_path, save_folder, quality=None, scale=None, mode=None):
    result = {'ela_image_path': None, 'ela_score': 0.0, 'ela_level': 'UNAVAILABLE',
              'ela_interpretation': 'ELA analysis could not be completed.',
              'success': False, 'error': '', 'details': {}}
    try:
        quality = int(quality or cfg('ELA_QUALITY', 90))
        scale = scale if scale is not None else cfg('ELA_SCALE', 'auto')
        mode = mode or cfg('ELA_MODE', 'color')
        rgb, fmt = open_rgb(image_path)
        if fmt != 'JPEG':
            result.update(success=True, ela_level='UNAVAILABLE', details={'applicable': False, 'format': fmt},
                          ela_interpretation=(f'ELA is not applicable to {fmt or "this"} images: it analyses '
                                              'JPEG compression history, which lossless files do not have. '
                                              'No ELA indicator is reported.'))
            return result
        buf = io.BytesIO()                                   # in-memory: no temp files to leak
        Image.fromarray(rgb).save(buf, 'JPEG', quality=quality)
        buf.seek(0)
        rec = np.asarray(Image.open(buf).convert('RGB'), dtype=np.float32)
        diff = np.abs(rgb.astype(np.float32) - rec)         # raw error, 0..255
        err = diff.max(axis=2)
        mean_err, p99 = float(err.mean()), float(np.percentile(err, 99))

        gain = (255.0 / max(float(np.percentile(err, 99.5)), 1.0)) if scale == 'auto' else float(scale)
        vis_src = diff if mode == 'color' else np.repeat(err[..., None], 3, axis=2)
        vis = np.clip(vis_src * gain, 0, 255).astype(np.uint8)   # amplified ONCE, for display only
        frac, contrast = _region_stats(err, int(cfg('ELA_BLOCK', 8)))

        flagged = frac >= cfg('ELA_REGION_FRACTION', 0.08) and contrast >= cfg('ELA_REGION_CONTRAST', 2.0)
        level = 'MEDIUM' if flagged else 'LOW'              # never HIGH: see module docstring
        score = round(min(100.0, frac * 100.0), 1)
        result.update(
            ela_image_path=save_png(vis, save_folder, 'ela'), ela_score=score, ela_level=level, success=True,
            ela_interpretation=(
                f'JPEG re-compressed at quality {quality}. The largest compact high-error region covers '
                f'{score:.1f}% of the image (contrast {contrast:.1f}x the median error). '
                + ('A region responds differently from the rest of the image; this can come from editing, '
                   'but also from texture or mixed compression history. ' if flagged else
                   'No compact region stands out from the rest of the image. ')
                + LIMITATIONS),
            details={'applicable': True, 'quality': quality, 'mean_error': round(mean_err, 3),
                     'p99_error': round(p99, 3), 'region_fraction': round(frac, 4),
                     'region_contrast': round(contrast, 3), 'display_gain': round(gain, 2),
                     'reliability': 'low', 'limitations': LIMITATIONS})
        logger.info('ELA complete: level=%s region=%.3f contrast=%.2f', level, frac, contrast)
    except Exception:
        logger.exception('ELA failed')
        result.update(error='ELA analysis failed.', ela_level='UNAVAILABLE')
    return result
