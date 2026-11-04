"""
Server-side upload validation based on file CONTENT.

Nothing here trusts the filename, extension, or client-supplied Content-Type.
Checks, in order:
  1. size limits
  2. magic bytes (JPEG / PNG only)
  3. suspicious trailing data after the image end marker (polyglot heuristic)
  4. header parse + dimension / pixel limits BEFORE full decode
  5. full decode under a warnings-as-errors guard (catches truncation, bombs)
  6. single-frame only

The original bytes are never modified; the caller keeps the original evidence file.
"""

import io
import warnings
from dataclasses import dataclass

from django.conf import settings
from django.core.exceptions import ValidationError
from PIL import Image

JPEG_MAGIC = b'\xff\xd8\xff'
PNG_MAGIC = b'\x89PNG\r\n\x1a\n'

# Byte signatures that should never follow the end of a genuine image.
_DANGEROUS_TRAILER_MARKERS = (
    b'<?php', b'<script', b'<html', b'#!/', b'PK\x03\x04', b'MZ\x90\x00',
    b'\x7fELF', b'%PDF-',
)
_FORMAT_EXT = {'JPEG': 'jpg', 'PNG': 'png'}


@dataclass(frozen=True)
class ValidatedImage:
    format: str      # 'JPEG' | 'PNG' (from content)
    extension: str   # 'jpg' | 'png' — safe to use in a server-side filename
    width: int
    height: int
    size: int


def _fail(message: str, code: str) -> ValidationError:
    from detector import metrics
    metrics.record_upload_rejected(code)
    return ValidationError(message, code=code)


def _sniff_format(head: bytes):
    if head.startswith(JPEG_MAGIC):
        return 'JPEG'
    if head.startswith(PNG_MAGIC):
        return 'PNG'
    return None


def _trailing_data(data: bytes, fmt: str) -> bytes:
    if fmt == 'JPEG':
        idx = data.rfind(b'\xff\xd9')
        return data[idx + 2:] if idx != -1 else b''
    idx = data.rfind(b'IEND')
    return data[idx + 8:] if idx != -1 else b''   # 4 (name) + 4 (CRC)


def validate_image_bytes(data: bytes) -> ValidatedImage:
    """Validate raw upload bytes; raise ValidationError (safe message) on failure."""
    max_size = settings.MAX_UPLOAD_SIZE
    if len(data) == 0:
        raise _fail('The uploaded file is empty.', 'empty')
    if len(data) > max_size:
        raise _fail(f'The file is too large. Maximum size is {max_size // (1024 * 1024)} MB.',
                    'too_large')

    fmt = _sniff_format(data[:16])
    if fmt is None or fmt not in settings.ALLOWED_IMAGE_FORMATS:
        raise _fail('Unsupported file type. Please upload a JPEG or PNG image.', 'unsupported')

    trailer = _trailing_data(data, fmt)
    if trailer and any(marker in trailer for marker in _DANGEROUS_TRAILER_MARKERS):
        raise _fail('The file contains unexpected embedded content and was rejected.', 'suspicious')

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error')          # DecompressionBombWarning -> error
            with Image.open(io.BytesIO(data)) as probe:
                if probe.format != fmt:
                    raise _fail('The file content does not match a supported image type.',
                                'mismatch')
                width, height = probe.size
                if width < settings.MIN_IMAGE_SIDE or height < settings.MIN_IMAGE_SIDE:
                    raise _fail('The image is too small to analyse.', 'too_small')
                if (width > settings.MAX_IMAGE_WIDTH or height > settings.MAX_IMAGE_HEIGHT
                        or width * height > settings.MAX_PIXELS):
                    raise _fail('The image resolution is too large.', 'too_many_pixels')
                if getattr(probe, 'n_frames', 1) > 1:
                    raise _fail('Animated or multi-frame images are not supported.', 'multi_frame')
                probe.verify()
            # verify() invalidates the object; decode fully from a fresh handle
            with Image.open(io.BytesIO(data)) as full:
                full.load()
    except ValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise _fail('The image resolution is too large.', 'too_many_pixels')
    except Exception:
        # Corrupt/truncated/malformed. Never echo parser internals to the user.
        raise _fail('The image is corrupted or could not be read.', 'corrupt')

    return ValidatedImage(format=fmt, extension=_FORMAT_EXT[fmt], width=width, height=height,
                          size=len(data))


def validate_uploaded_image(uploaded_file) -> ValidatedImage:
    """Validate a Django UploadedFile and rewind it for later saving."""
    if uploaded_file.size > settings.MAX_UPLOAD_SIZE:   # cheap check before reading
        raise _fail(f'The file is too large. Maximum size is '
                    f'{settings.MAX_UPLOAD_SIZE // (1024 * 1024)} MB.', 'too_large')
    uploaded_file.seek(0)
    data = uploaded_file.read(settings.MAX_UPLOAD_SIZE + 1)
    uploaded_file.seek(0)
    return validate_image_bytes(data)
