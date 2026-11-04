"""
Private file storage abstraction (local disk or S3-compatible object storage).

Design
------
* Uploads are saved through Django's default storage (``STORAGES['default']``), so with
  ``STORAGE_BACKEND=s3`` the original image goes straight to the private bucket.
* The analysis pipeline works on *local files* (OpenCV / PIL / forensic modules need real paths).
  Workers therefore pull the original into a scratch directory, write derived images (face crops,
  heat-maps, forensic maps) into ``MEDIA_ROOT`` (acting as scratch), and when the job finishes
  ``publish_artifacts`` uploads them to the bucket under the same relative name and removes the
  local copies.
* Delivery never exposes the bucket: Django checks ownership, then answers with a short-lived
  pre-signed URL (``AWS_QUERYSTRING_EXPIRE``) instead of streaming or making objects public.
* With local storage every function is a harmless no-op / passthrough.
"""

import contextlib
import logging
import os
import shutil
import tempfile

from django.conf import settings
from django.core.files.storage import FileSystemStorage, default_storage

logger = logging.getLogger('deepshield.storage')

_ARTIFACT_FIELDS = {
    'uploaded': ('cropped_face',),
}


def uses_remote_storage() -> bool:
    storage = default_storage._wrapped if hasattr(default_storage, '_wrapped') else default_storage
    return not isinstance(storage, FileSystemStorage)


@contextlib.contextmanager
def local_copy(field_file):
    """Yield a readable local path for a FileField value (downloads from object storage if needed)."""
    if not uses_remote_storage():
        yield field_file.path
        return
    scratch = tempfile.mkdtemp(prefix='ds_scratch_')
    try:
        target = os.path.join(scratch, os.path.basename(field_file.name))
        with field_file.storage.open(field_file.name, 'rb') as src, open(target, 'wb') as dst:
            shutil.copyfileobj(src, dst)
        yield target
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def artifact_names(upload) -> set:
    """Relative storage names of every derived file belonging to an upload's analysis."""
    from detector.models import Face, ForensicAnalysis, ModelResult
    names = set()
    if upload.cropped_face:
        names.add(upload.cropped_face.name)
    for n in Face.objects.filter(analysis__uploaded_image=upload).values_list('crop_image', flat=True):
        if n:
            names.add(n)
    for n in ModelResult.objects.filter(analysis__uploaded_image=upload).values_list('gradcam_image', flat=True):
        if n:
            names.add(n)
    fields = ('ela_image', 'freq_image', 'noise_image', 'copy_move_image',
              'face_boundary_image', 'resampling_image')
    for row in ForensicAnalysis.objects.filter(analysis__uploaded_image=upload).values(*fields):
        names.update(v for v in row.values() if v)
    from detector.models import Analysis
    for info in Analysis.objects.filter(uploaded_image=upload).values_list('pipeline_info', flat=True):
        rel = ((info or {}).get('image') or {}).get('processing_copy')
        if rel:
            names.add(rel)
    return names


def publish_artifacts(upload) -> int:
    """Move locally produced artifacts to object storage (no-op for local storage). Returns count."""
    if not uses_remote_storage():
        return 0
    moved = 0
    root = str(settings.MEDIA_ROOT)
    for rel in artifact_names(upload):
        local = os.path.join(root, rel)
        if not os.path.isfile(local):
            continue
        if default_storage.exists(rel):
            default_storage.delete(rel)
        with open(local, 'rb') as fh:
            saved = default_storage.save(rel, fh)
        if saved != rel:                       # storage renamed it: keep DB pointing at the real object
            logger.error('storage renamed artifact %s -> %s', rel, saved)
        os.remove(local)
        moved += 1
    return moved


def signed_url(rel_path: str) -> str:
    """Short-lived pre-signed URL (lifetime from AWS_QUERYSTRING_EXPIRE)."""
    return default_storage.url(rel_path)


def storage_exists(rel_path: str) -> bool:
    return default_storage.exists(rel_path)
