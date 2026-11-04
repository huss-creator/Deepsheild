"""Remove files from storage when their database rows are deleted (user deletion, API DELETE,
retention jobs, cascades), so deleting an analysis really deletes the images."""

import logging
import os

from django.conf import settings
from django.core.exceptions import SuspiciousFileOperation
from django.core.files.storage import default_storage
from django.db.models.signals import post_delete
from django.dispatch import receiver
from django.utils._os import safe_join

from detector.models import Analysis, Face, ForensicAnalysis, ModelResult, UploadedImage

logger = logging.getLogger('deepshield.storage')

_FILE_FIELDS = {
    UploadedImage: ('image', 'cropped_face'),
    Face: ('crop_image',),
    ModelResult: ('gradcam_image',),
    ForensicAnalysis: ('ela_image', 'freq_image', 'noise_image', 'copy_move_image',
                       'face_boundary_image', 'resampling_image'),
}


def _delete_files(sender, instance, **kwargs):
    for name in _FILE_FIELDS[sender]:
        f = getattr(instance, name, None)
        if f:
            try:
                f.storage.delete(f.name)      # missing files are ignored by FileSystemStorage
            except Exception:
                logger.exception('could not delete stored file for %s', sender.__name__)


for _model in _FILE_FIELDS:
    post_delete.connect(_delete_files, sender=_model, dispatch_uid=f'delete_files_{_model.__name__}')


@receiver(post_delete, sender=Analysis)
def _delete_processing_copy(sender, instance, **kwargs):
    rel = ((instance.pipeline_info or {}).get('image') or {}).get('processing_copy')
    if rel:
        try:
            os.remove(safe_join(settings.MEDIA_ROOT, rel))
        except (OSError, SuspiciousFileOperation):
            pass
        from detector import storage
        if storage.uses_remote_storage():
            try:
                default_storage.delete(rel)
            except Exception:
                logger.exception('could not delete processing copy from object storage')
