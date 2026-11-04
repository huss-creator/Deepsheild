"""
Data retention and deletion.

* Analyses (original image, face crops, heat-maps, forensic maps, report, job) older than
  ``DELETE_ANALYSIS_AFTER_DAYS`` are deleted by ``manage.py purge_expired``. 0 disables the policy.
* Deleting an ``UploadedImage`` cascades to every dependent row and the ``post_delete`` signals remove
  the stored files (local disk or object storage), so deletion is real, not a flag.
* Analyses with a job still in flight are never purged mid-processing.
* Audit rows are pruned after ``AUDIT_LOG_RETENTION_DAYS``. They hold no image data.
* Whole-account deletion removes the user, all their analyses/files and revokes refresh tokens.
"""

import os
import logging
from datetime import timedelta

from django.conf import settings
from django.utils import timezone

from detector import audit, metrics
from detector.models import AnalysisJob, AuditLog, UploadedImage

logger = logging.getLogger('deepshield.retention')


def purge_expired(now=None, dry_run=False, batch=200):
    """Delete expired analyses. Returns {'analyses': n, 'audit_rows': m, 'cutoff': iso|None}."""
    now = now or timezone.now()
    days = settings.DELETE_ANALYSIS_AFTER_DAYS
    result = {'analyses': 0, 'audit_rows': 0, 'cutoff': None}
    if days and days > 0:
        cutoff = now - timedelta(days=days)
        result['cutoff'] = cutoff.isoformat()
        qs = (UploadedImage.objects.filter(uploaded_at__lt=cutoff)
              .exclude(job__status__in=AnalysisJob.ACTIVE))
        ids = list(qs.values_list('pk', flat=True))
        result['analyses'] = len(ids)
        if not dry_run:
            for i in range(0, len(ids), batch):
                for rec in UploadedImage.objects.filter(pk__in=ids[i:i + batch]):
                    rec.delete()
            metrics.record_retention_deleted(len(ids))
            logger.info('retention purge deleted %d analyses older than %d days', len(ids), days)
    audit_days = settings.AUDIT_LOG_RETENTION_DAYS
    if audit_days and audit_days > 0:
        old = AuditLog.objects.filter(created_at__lt=now - timedelta(days=audit_days))
        result['audit_rows'] = old.count()
        if not dry_run:
            old.delete()
    return result


def delete_account(user, request=None):
    """Permanently delete a user and everything they uploaded. Returns number of analyses removed."""
    n = UploadedImage.objects.filter(user=user).count()
    audit.audit(audit.ACCOUNT_DELETED, request, user=user, analyses=n)   # username survives in the log
    uid = user.pk
    user.delete()                                  # post_delete signals remove individual files
    _remove_user_media_dir(uid)                    # then sweep the now-empty per-user folder
    return n


def _remove_user_media_dir(user_id):
    import shutil
    from django.conf import settings
    from detector.models import user_media_prefix
    try:
        root = os.path.realpath(str(settings.MEDIA_ROOT))
        target = os.path.realpath(os.path.join(root, *user_media_prefix(user_id).split('/')))
        if target.startswith(root + os.sep) and os.path.isdir(target):
            shutil.rmtree(target, ignore_errors=True)
    except Exception:
        logging.getLogger(__name__).warning('could not remove media dir for user %s', user_id)
