"""
Structured (JSON) logging with request/job correlation IDs and automatic redaction.

Redaction is a safety net, not licence to log sensitive data: code must still never log passwords,
tokens, image bytes, EXIF/GPS values or full file paths. The filter removes Bearer tokens, JWTs,
e-mail addresses and anything keyed like a secret, and shortens absolute filesystem paths.
"""

import json
import logging
import re
from datetime import datetime, timezone

_SECRET_KEYS = re.compile(r'(pass(word)?|secret|token|authorization|api[_-]?key|cookie|signature|dsn)', re.I)
_PATTERNS = [
    (re.compile(r'(?i)bearer\s+[A-Za-z0-9._~+/=-]+'), 'Bearer [REDACTED]'),
    (re.compile(r'eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}'), '[REDACTED_JWT]'),
    (re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'), '[REDACTED_EMAIL]'),
    (re.compile(r'(?i)(X-Amz-(?:Signature|Credential|Security-Token)=)[^&\s]+'), r'\1[REDACTED]'),
    (re.compile(r'(?i)((?:password|secret|token|api[_-]?key)\s*[=:]\s*)[^\s,;]+'), r'\1[REDACTED]'),
    (re.compile(r'(?<![\w.])/(?:srv|home|var|tmp|opt|app|mnt|root|etc|usr)/[^\s\'",;:)]+'), '[PATH]'),
]
_STD_ATTRS = set(logging.LogRecord('', 0, '', 0, '', (), None).__dict__) | {'message', 'asctime', 'taskName'}


def redact_text(text: str) -> str:
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    return text


def redact_value(key, value):
    if _SECRET_KEYS.search(str(key)):
        return '[REDACTED]'
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {k: redact_value(k, v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_value(key, v) for v in value]
    return value


class ContextFilter(logging.Filter):
    def filter(self, record):
        from detector.logctx import job_id_var
        from detector.middleware import request_id_var
        record.request_id = request_id_var.get() or None
        record.job_id = job_id_var.get() or None
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            'ts': datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec='milliseconds'),
            'level': record.levelname,
            'logger': record.name,
            'msg': redact_text(record.getMessage()),
            'request_id': getattr(record, 'request_id', None),
            'job_id': getattr(record, 'job_id', None),
        }
        for key, value in record.__dict__.items():
            if key not in _STD_ATTRS and key not in payload and not key.startswith('_'):
                payload[key] = redact_value(key, value)
        if record.exc_info:
            # Exception type + redacted message + stack frames; no local variables.
            payload['exc_type'] = record.exc_info[0].__name__
            payload['exc'] = redact_text(self.formatException(record.exc_info))
        return json.dumps(payload, default=str, ensure_ascii=False)


class RedactingTextFormatter(logging.Formatter):
    def format(self, record):
        return redact_text(super().format(record))


def build_logging(level='INFO', fmt='json'):
    formatter = {'()': 'config.logging_conf.JsonFormatter'} if fmt == 'json' else {
        '()': 'config.logging_conf.RedactingTextFormatter',
        'format': '%(asctime)s %(levelname)s %(name)s [req=%(request_id)s job=%(job_id)s] %(message)s'}
    return {
        'version': 1, 'disable_existing_loggers': False,
        'filters': {'context': {'()': 'config.logging_conf.ContextFilter'}},
        'formatters': {'default': formatter},
        'handlers': {'console': {'class': 'logging.StreamHandler', 'filters': ['context'], 'formatter': 'default'}},
        'root': {'handlers': ['console'], 'level': level},
        'loggers': {
            'django.request': {'level': 'WARNING'},
            'django.security': {'level': 'WARNING'},
            'celery': {'level': 'INFO'},
        },
    }
