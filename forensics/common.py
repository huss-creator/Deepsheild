"""Shared helpers for forensic modules (single decode path, native-resolution crops,
lossless visualisation output, optional Django-settings parameters)."""

import logging
import os
import uuid

import numpy as np
from PIL import Image

logger = logging.getLogger('deepshield.forensics')


def cfg(name: str, default):
    """Module parameter from Django settings when available, else the default."""
    try:
        from django.conf import settings
        return getattr(settings, name, default)
    except Exception:
        return default


def open_rgb(path):
    """Decode WITHOUT orientation changes (forensic traces live on the stored pixel grid).
    Returns (rgb uint8 array, file format string e.g. 'JPEG'/'PNG')."""
    with Image.open(path) as im:
        fmt = im.format or ''
        if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
            rgba = im.convert('RGBA')
            bg = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, rgba)
        return np.asarray(im.convert('RGB'), dtype=np.uint8).copy(), fmt


def central_crop(arr: np.ndarray, size: int) -> np.ndarray:
    """Native-resolution central window (NO resampling) of at most size x size pixels."""
    h, w = arr.shape[:2]
    ch, cw = min(h, size), min(w, size)
    y, x = (h - ch) // 2, (w - cw) // 2
    return np.ascontiguousarray(arr[y:y + ch, x:x + cw])


def save_png(arr: np.ndarray, folder: str, prefix: str):
    """Save a visualisation losslessly; returns path or None."""
    try:
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f'{prefix}_{uuid.uuid4().hex[:10]}.png')
        # Display maps only: statistics are always computed at full resolution; the saved picture is
        # capped (FORENSIC_VIS_MAX_SIDE, default 2048 px) so a 12 MP upload does not spend seconds in zlib.
        max_side = int(cfg('FORENSIC_VIS_MAX_SIDE', 2048))
        h, w = arr.shape[:2]
        if max_side > 0 and max(h, w) > max_side:
            import cv2
            k = max_side / float(max(h, w))
            arr = cv2.resize(arr, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
        Image.fromarray(arr).save(path, 'PNG', compress_level=int(cfg('FORENSIC_PNG_LEVEL', 3)))
        return path
    except Exception:
        logger.exception('could not save %s visualisation', prefix)
        return None
