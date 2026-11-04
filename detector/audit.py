"""Audit trail for security-relevant events. Never store passwords, tokens, image content or
sensitive metadata. Failures to write an audit row never break the request."""

import hashlib
import logging

from detector import ratelimit
from detector.models import AuditLog

logger = logging.getLogger('deepshield.audit')

# actions
LOGIN, LOGIN_FAILED, LOGOUT, REGISTER = 'login', 'login_failed', 'logout', 'register'
ANALYSIS_CREATED, ANALYSIS_DELETED = 'analysis_created', 'analysis_deleted'
ANALYSIS_CANCELLED = 'analysis_cancelled'
REPORT_VIEWED, ACCESS_DENIED, THROTTLED = 'report_viewed', 'access_denied', 'throttled'
PASSWORD_RESET_REQUESTED = 'password_reset_requested'  # nosec B105 (event name, not a secret)
ACCOUNT_DELETED = 'account_deleted'

_FORBIDDEN_KEYS = {'password', 'token', 'refresh', 'access', 'authorization', 'secret'}


def hash_identifier(value: str) -> str:
    """Short irreversible fingerprint (used for attempted usernames on failed logins, which may
    contain a mistyped password)."""
    return hashlib.sha256((value or '').encode()).hexdigest()[:12]


def audit(action, request=None, user=None, obj=None, success=True, **metadata):
    try:
        user = user or (getattr(request, 'user', None) if request is not None else None)
        if user is not None and not getattr(user, 'is_authenticated', False):
            user = None
        ip = ratelimit.client_ip(request) if request is not None else None
        clean = {k: v for k, v in metadata.items()
                 if k.lower() not in _FORBIDDEN_KEYS and isinstance(v, (str, int, float, bool, type(None)))}
        AuditLog.objects.create(
            user=user, username=(user.get_username() if user else '')[:150], action=action,
            object_type=obj.__class__.__name__ if obj is not None else '',
            object_id=str(getattr(obj, 'public_id', None) or getattr(obj, 'pk', '') or '')[:64] if obj is not None else '',
            success=success, ip_address=ip or None,
            request_id=(getattr(request, 'request_id', '') or '')[:64] if request is not None else '',
            metadata=clean)
    except Exception:
        logger.exception('audit write failed for %s', action)
