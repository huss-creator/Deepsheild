"""
Authenticated delivery of private media (uploads, face crops, forensic images).

Media is never mapped as a public static directory. Every /media/<path> request:
  1. requires a logged-in user,
  2. resolves the path safely inside MEDIA_ROOT (no traversal),
  3. is served only if a database record owned by the requesting user references it.
Non-owned and non-existent files both return 404 so file names cannot be enumerated.
"""

import logging
import mimetypes
import posixpath

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.exceptions import SuspiciousFileOperation
from django.db.models import Q
from django.http import FileResponse, Http404, HttpResponse, HttpResponseRedirect
from django.utils._os import safe_join

from detector.models import Face, ForensicAnalysis, ModelResult, UploadedImage

logger = logging.getLogger('deepshield.security')

_SERVABLE_TYPES = {'image/jpeg', 'image/png'}
_FORENSIC_IMAGE_FIELDS = ('ela_image', 'freq_image', 'noise_image', 'copy_move_image',
                          'face_boundary_image', 'resampling_image')


def _normalise(path: str) -> str:
    clean = posixpath.normpath(path.replace('\\', '/'))
    if clean.startswith(('/', '..')) or '\x00' in clean or clean == '.':
        raise Http404
    return clean


def user_owns_media(user, rel_path: str) -> bool:
    # Defence in depth: files under users/user_<id>/ are only ever served to that user.
    # Legacy flat paths (uploads/, faces/ ...) fall through to the database ownership check.
    parts = rel_path.replace('\\', '/').split('/')
    if parts[0] == 'users':
        if len(parts) < 3 or parts[1] != f'user_{user.pk}':
            return False
    # Normalise separators: DB may store backslashes on Windows, URL always uses forward slashes.
    rel_path_fwd = rel_path.replace('\\', '/')
    rel_path_bwd = rel_path.replace('/', '\\')
    path_q = Q(image=rel_path_fwd) | Q(image=rel_path_bwd) | \
             Q(cropped_face=rel_path_fwd) | Q(cropped_face=rel_path_bwd)
    if UploadedImage.objects.filter(user=user).filter(path_q).exists():
        return True
    if ModelResult.objects.filter(analysis__uploaded_image__user=user).filter(
            Q(gradcam_image=rel_path_fwd) | Q(gradcam_image=rel_path_bwd)).exists():
        return True
    if Face.objects.filter(analysis__uploaded_image__user=user).filter(
            Q(crop_image=rel_path_fwd) | Q(crop_image=rel_path_bwd)).exists():
        return True
    forensic_q = Q()
    for field in _FORENSIC_IMAGE_FIELDS:
        forensic_q |= Q(**{field: rel_path_fwd}) | Q(**{field: rel_path_bwd})
    return ForensicAnalysis.objects.filter(
        analysis__uploaded_image__user=user).filter(forensic_q).exists()


def serve_media_file(rel_path: str):
    """Stream an already-authorised private image (X-Accel-Redirect in production)."""
    rel_path = _normalise(rel_path)
    try:
        full_path = safe_join(settings.MEDIA_ROOT, rel_path)
    except SuspiciousFileOperation:
        raise Http404
    content_type = mimetypes.guess_type(full_path)[0] or ''
    if content_type not in _SERVABLE_TYPES:
        raise Http404
    from detector import storage
    if storage.uses_remote_storage():
        # Object storage: bucket stays private; hand out a short-lived pre-signed URL.
        if not storage.storage_exists(rel_path):
            raise Http404
        response = HttpResponseRedirect(storage.signed_url(rel_path))
        response['Cache-Control'] = 'private, no-store'
        response['Referrer-Policy'] = 'no-referrer'
        return response
    if settings.MEDIA_X_ACCEL_PREFIX:
        response = HttpResponse(content_type=content_type)
        response['X-Accel-Redirect'] = settings.MEDIA_X_ACCEL_PREFIX.rstrip('/') + '/' + rel_path
    else:
        try:
            response = FileResponse(open(full_path, 'rb'), content_type=content_type)
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            raise Http404
    response['Cache-Control'] = 'private, no-store'
    response['Content-Disposition'] = 'inline'
    response['X-Content-Type-Options'] = 'nosniff'
    return response


@login_required
def protected_media(request, path):
    rel_path = _normalise(path)
    try:
        safe_join(settings.MEDIA_ROOT, rel_path)
    except SuspiciousFileOperation:
        logger.warning('media path traversal attempt', extra={'user_id': request.user.pk})
        raise Http404
    if not user_owns_media(request.user, rel_path):
        logger.warning('media access denied', extra={'user_id': request.user.pk})
        from detector import audit
        audit.audit(audit.ACCESS_DENIED, request, success=False, resource='media')
        raise Http404
    return serve_media_file(rel_path)
