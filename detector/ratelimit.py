"""
Fixed-window rate limiting on the Django cache (use Redis in production so all
workers share counters).

Limits come from settings.RATE_LIMITS = {scope: (max_requests, window_seconds)}.
"""

import hashlib

from django.conf import settings
from django.core.cache import cache


def client_ip(request) -> str:
    """Client IP. X-Forwarded-For is honoured only when NUM_PROXIES trusted proxies
    are configured, and then only the entry those proxies appended (not client-supplied)."""
    remote = request.META.get('REMOTE_ADDR', '')
    proxies = getattr(settings, 'NUM_PROXIES', 0)
    forwarded = request.META.get('HTTP_X_FORWARDED_FOR', '')
    if proxies > 0 and forwarded:
        parts = [p.strip() for p in forwarded.split(',') if p.strip()]
        if len(parts) >= proxies:
            return parts[-proxies]
    return remote


def _key(scope: str, ident: str) -> str:
    return f'rl:{scope}:{hashlib.sha256(ident.encode()).hexdigest()[:32]}'


def _limits(scope: str):
    return settings.RATE_LIMITS[scope]


def hit(scope: str, ident: str) -> int:
    """Count one event; return the count within the current window."""
    _, window = _limits(scope)
    key = _key(scope, ident)
    cache.add(key, 0, window)
    try:
        return cache.incr(key)
    except ValueError:              # key expired between add() and incr()
        cache.set(key, 1, window)
        return 1


def is_limited(scope: str, ident: str) -> bool:
    """True if the scope has already reached its maximum (does not count)."""
    maximum, _ = _limits(scope)
    return (cache.get(_key(scope, ident)) or 0) >= maximum


def allow(scope: str, ident: str) -> bool:
    """Count one event and return False once it exceeds the scope's maximum."""
    maximum, _ = _limits(scope)
    return hit(scope, ident) <= maximum


def reset(scope: str, ident: str) -> None:
    cache.delete(_key(scope, ident))


def retry_after(scope: str) -> int:
    return _limits(scope)[1]
