"""Development settings. Never use on a server reachable from the internet."""

import secrets

from .base import *  # noqa: F401,F403
from .base import BASE_DIR, env_bool, env_list, env_str

DEBUG = env_bool('DEBUG', True)
ALLOWED_HOSTS = env_list('ALLOWED_HOSTS', ['localhost', '127.0.0.1', '[::1]', 'testserver'])


def _dev_secret_key() -> str:
    """Use SECRET_KEY if provided, else generate one per checkout and keep it in a
    git-ignored file. No secret is ever hardcoded in source control."""
    key = env_str('SECRET_KEY')
    if key:
        return key
    path = BASE_DIR / '.secret_key.dev'
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        key = secrets.token_urlsafe(64)
        path.write_text(key)
        try:
            path.chmod(0o600)
        except OSError:
            pass
        return key


SECRET_KEY = _dev_secret_key()

REPORT_SIGNING_KEY = env_str('REPORT_SIGNING_KEY') or SECRET_KEY
EMAIL_BACKEND = env_str('EMAIL_BACKEND', 'django.core.mail.backends.console.EmailBackend')

# Opt-in PostgreSQL for dev/CI (same engine as production): set DB_NAME (+ DB_USER, DB_PASSWORD, DB_HOST, DB_PORT).
if env_str('DB_NAME'):  # noqa: F405
    DATABASES = {'default': {
        'ENGINE': 'django.db.backends.postgresql', 'NAME': env_str('DB_NAME'),  # noqa: F405
        'USER': env_str('DB_USER', 'postgres'), 'PASSWORD': env_str('DB_PASSWORD'),  # noqa: F405
        'HOST': env_str('DB_HOST', 'localhost'), 'PORT': env_str('DB_PORT', '5432')}}  # noqa: F405


import sys as _sys  # noqa: E402

if 'test' not in _sys.argv:
    JOB_RUN_IN_THREAD = env_bool('JOB_RUN_IN_THREAD', True)
