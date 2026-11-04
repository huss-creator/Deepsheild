"""
Base settings shared by every environment.

Everything deployment-specific comes from environment variables (optionally
loaded from a git-ignored `.env`). See `.env.example` for the full list.
Environment-specific modules: `development.py`, `production.py`.
"""

import os
import sys
from pathlib import Path

if os.name == 'nt':
    scripts_dir = os.path.join(sys.prefix, 'Scripts')
    if os.path.isdir(scripts_dir):
        try:
            os.add_dll_directory(scripts_dir)
        except Exception:
            pass

from django.contrib.messages import constants as messages

BASE_DIR = Path(__file__).resolve().parent.parent.parent

try:  # python-dotenv is optional; real environment variables always win
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / '.env')
except ImportError:
    pass


# ---------------------------------------------------------------- env helpers
def env_str(name: str, default: str = '') -> str:
    return os.environ.get(name, default)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def env_list(name: str, default=None) -> list:
    raw = os.environ.get(name)
    if raw is None:
        return list(default or [])
    return [item.strip() for item in raw.split(',') if item.strip()]


# ------------------------------------------------------------------ core
# SECRET_KEY, DEBUG and ALLOWED_HOSTS are defined per environment.
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'corsheaders',
    'rest_framework',
    'rest_framework_simplejwt.token_blacklist',
    'drf_spectacular',
    'detector',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'detector.middleware.RequestIDMiddleware',
    'detector.metrics.MetricsMiddleware',
    'detector.csp.CSPMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'detector.middleware.RequestSizeLimitMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'
WSGI_APPLICATION = 'config.wsgi.application'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'detector.csp.csp_nonce',
                'detector.context_processors.site_context',
            ],
        },
    },
]

# ---------------------------------------------------------------- database
# Development default is SQLite; production.py requires PostgreSQL.
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': env_str('SQLITE_PATH', str(BASE_DIR / 'db.sqlite3')),
    }
}
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ---------------------------------------------------------------- passwords
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
     'OPTIONS': {'min_length': 10}},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# Password hashing algorithm preference.
# When argon2-cffi is installed (production standard), Argon2id is preferred.
# Otherwise, Django's hardened PBKDF2-SHA256 (720,000 rounds) is used automatically.
try:
    import argon2  # noqa: F401
    _has_argon2 = True
except ImportError:
    _has_argon2 = False

PASSWORD_HASHERS = (
    ['django.contrib.auth.hashers.Argon2PasswordHasher'] if _has_argon2 else []
) + [
    'django.contrib.auth.hashers.PBKDF2PasswordHasher',
    'django.contrib.auth.hashers.PBKDF2SHA1PasswordHasher',
    'django.contrib.auth.hashers.BCryptSHA256PasswordHasher',
    'django.contrib.auth.hashers.ScryptPasswordHasher',
]


# ----------------------------------------------------------- i18n / static
LANGUAGE_CODE = 'en-us'
TIME_ZONE = env_str('TIME_ZONE', 'Asia/Karachi')
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = Path(env_str('STATIC_ROOT', str(BASE_DIR / 'staticfiles')))

# ------------------------------------------------------------------- media
# Media is PRIVATE. It is never exposed by a static file mapping; every
# request goes through detector.views.protected_media, which checks ownership.
MEDIA_URL = '/media/'
MEDIA_ROOT = Path(env_str('MEDIA_ROOT', str(BASE_DIR / 'media')))
# When set (e.g. '/_protected_media/'), the view answers with X-Accel-Redirect
# so Nginx streams the file. Empty = Django streams it (development).
MEDIA_X_ACCEL_PREFIX = env_str('MEDIA_X_ACCEL_PREFIX', '')

STORAGES = {
    'default': {'BACKEND': 'django.core.files.storage.FileSystemStorage'},
    'staticfiles': {'BACKEND': 'django.contrib.staticfiles.storage.StaticFilesStorage'},
}

# -------------------------------------------------------------------- auth
LOGIN_REDIRECT_URL = 'upload'
LOGOUT_REDIRECT_URL = 'home'
LOGIN_URL = 'login'
PASSWORD_RESET_TIMEOUT = 60 * 60   # reset links valid for 1 hour

# ------------------------------------------------------------ upload limits
MAX_UPLOAD_SIZE = env_int('MAX_UPLOAD_SIZE', 10 * 1024 * 1024)      # bytes
MAX_IMAGE_WIDTH = env_int('MAX_IMAGE_WIDTH', 8000)                  # px
MAX_IMAGE_HEIGHT = env_int('MAX_IMAGE_HEIGHT', 8000)                # px
MAX_PIXELS = env_int("MAX_PIXELS", 25_000_000)                      # w*h
MIN_IMAGE_SIDE = env_int('MIN_IMAGE_SIDE', 32)                      # px
# Whole request body cap, checked before multipart parsing (headroom for form fields).
MAX_REQUEST_BODY_SIZE = MAX_UPLOAD_SIZE + 1024 * 1024
DATA_UPLOAD_MAX_MEMORY_SIZE = 1024 * 1024       # non-file form data
FILE_UPLOAD_MAX_MEMORY_SIZE = 2 * 1024 * 1024   # larger uploads spool to disk
DATA_UPLOAD_MAX_NUMBER_FIELDS = 200
# Formats are decided from file CONTENT, never from extension or Content-Type.
ALLOWED_IMAGE_FORMATS = ('JPEG', 'PNG')

# PIL / OpenCV decompression-bomb guards (applied again in detector.apps).
PIL_MAX_IMAGE_PIXELS = MAX_PIXELS
os.environ.setdefault('OPENCV_IO_MAX_IMAGE_PIXELS', str(MAX_PIXELS))

# ------------------------------------------------------------ rate limiting
# (max requests, window seconds). Backed by the Django cache (Redis in production).
RATE_LIMITS = {
    'login_fail': (env_int('RL_LOGIN_FAIL_MAX', 5), 15 * 60),   # per IP+username
    'login_fail_ip': (env_int('RL_LOGIN_FAIL_IP_MAX', 30), 15 * 60), # per IP, any username
    'register':   (env_int('RL_REGISTER_MAX', 5), 60 * 60),     # per IP
    'password_reset': (env_int('RL_RESET_MAX', 5), 60 * 60),    # per IP
    'upload':     (env_int('RL_UPLOAD_MAX', 20), 60 * 60),      # per user
    'api_user':   (env_int('RL_API_USER_MAX', 1000), 60 * 60),  # all authenticated API calls, per user
}
# Number of trusted reverse proxies in front of Django (0 = use REMOTE_ADDR only).
NUM_PROXIES = env_int('NUM_PROXIES', 0)

CACHES = {'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}}

# ---------------------------------------------------------------- AI models
# Model files live in MODEL_DIR (never hardcoded elsewhere). Per-model settings are
# read from MODEL_CONFIG_FILE (JSON, default MODEL_DIR/models.json — see
# trained_models/models.example.json) with optional env overrides, e.g.
# XCEPTION_MODEL_PATH, EFFICIENTNET_MODEL_PATH, VIT_MODEL_PATH,
# FREQUENCY_MODEL_PATH, SDXL_MODEL_PATH (and *_ENABLED).
MODEL_DIR = Path(env_str('MODEL_DIR', str(BASE_DIR / 'trained_models')))
MODEL_CONFIG_FILE = env_str('MODEL_CONFIG_FILE', str(MODEL_DIR / 'models.json'))
# Verify weights against the configured SHA-256 and refuse models without one.
# (Always enforced for pickle-based formats regardless of this flag.)
MODEL_REQUIRE_CHECKSUM = env_bool('MODEL_REQUIRE_CHECKSUM', False)
DEVICE = env_str('DEVICE', 'auto')                    # auto | cuda | cpu
AI_MODEL_TIMEOUT = env_int('AI_MODEL_TIMEOUT', 30)    # seconds per model per face
AI_EXPLAIN_ENABLED = env_bool('AI_EXPLAIN_ENABLED', True)
AI_EXPLAIN_MAX_FACES = env_int('AI_EXPLAIN_MAX_FACES', 3)

# Face detection / alignment. 'haar' = OpenCV Haar cascade (previous behaviour);
# 'yunet' = OpenCV YuNet (needs YUNET_MODEL_PATH; gives confidence + 5 landmarks
# used for eye-level alignment).
FACE_DETECTOR = env_str('FACE_DETECTOR', 'auto')
FACE_MIN_SIZE = env_int('FACE_MIN_SIZE', 30)                 # px, detector minimum
FACE_MIN_AREA_RATIO = float(env_str('FACE_MIN_AREA_RATIO', '0.005'))
FACE_QUALITY_MIN_SIZE = env_int('FACE_QUALITY_MIN_SIZE', 64)  # px; smaller => UNCERTAIN
FACE_CONFIDENCE_THRESHOLD = float(env_str('FACE_CONFIDENCE_THRESHOLD', '0.8'))
FACE_HAAR_MIN_NEIGHBORS = env_int('FACE_HAAR_MIN_NEIGHBORS', 7)
FACE_HAAR_VALIDATE_EYES = env_bool('FACE_HAAR_VALIDATE_EYES', True)
FACE_PADDING = float(env_str('FACE_PADDING', '0.2'))
AI_MAX_FACES = env_int('AI_MAX_FACES', 10)
YUNET_MODEL_PATH = env_str('YUNET_MODEL_PATH', str(MODEL_DIR / 'face_detector' / 'yunet.onnx'))

# Ensemble: defaults are UNVALIDATED (equal-weight mean). Produce a validated
# config with `manage.py evaluate_models` and point ENSEMBLE_CONFIG_PATH at it.
ENSEMBLE_CONFIG_PATH = env_str('ENSEMBLE_CONFIG_PATH', str(MODEL_DIR / 'ensemble_config.json'))

# ------------------------------------------------------------ background jobs
# Heavy analysis runs in Celery workers (GPU/CPU), never in the web request. With no broker
# configured (development, tests) tasks run eagerly in-process so everything still works.
CELERY_BROKER_URL = env_str('CELERY_BROKER_URL') or env_str('REDIS_URL')
CELERY_TASK_ALWAYS_EAGER = env_bool('CELERY_TASK_ALWAYS_EAGER', not bool(CELERY_BROKER_URL))
CELERY_TASK_EAGER_PROPAGATES = False
# Without a Celery broker (local development) run each analysis in a background thread so the
# processing page can show live progress. Tests keep the synchronous behaviour (default False).
JOB_RUN_IN_THREAD = env_bool('JOB_RUN_IN_THREAD', False)
CELERY_TASK_IGNORE_RESULT = True
CELERY_TASK_ACKS_LATE = True                 # a job is only acknowledged once it finished ...
CELERY_TASK_REJECT_ON_WORKER_LOST = True     # ... so a killed worker's job is redelivered
CELERY_WORKER_PREFETCH_MULTIPLIER = 1        # one job per worker slot (fair, GPU-friendly)
CELERY_TASK_DEFAULT_QUEUE = 'analysis'
CELERY_BROKER_CONNECTION_RETRY_ON_STARTUP = True
CELERY_BROKER_TRANSPORT_OPTIONS = {'visibility_timeout': 3600}
PROCESSING_TIMEOUT = env_int('PROCESSING_TIMEOUT', 180)          # seconds per job (soft limit)
CELERY_TASK_SOFT_TIME_LIMIT = PROCESSING_TIMEOUT
CELERY_TASK_TIME_LIMIT = PROCESSING_TIMEOUT + 30                 # hard kill
JOB_MAX_RETRIES = env_int('JOB_MAX_RETRIES', 2)                  # transient failures only
JOB_RETRY_BACKOFF = env_int('JOB_RETRY_BACKOFF', 5)              # seconds, doubled per retry
MAX_ACTIVE_JOBS_PER_USER = env_int('MAX_ACTIVE_JOBS_PER_USER', 3)
MAX_QUEUED_JOBS = env_int('MAX_QUEUED_JOBS', 200)                # queue-full back-pressure
STUCK_JOB_SECONDS = env_int('STUCK_JOB_SECONDS', PROCESSING_TIMEOUT * 3)

# Forensic display maps: longest side of saved PNGs (0 = no cap) and zlib level (1 fast .. 9 small).
FORENSIC_VIS_MAX_SIDE = env_int('FORENSIC_VIS_MAX_SIDE', 2048)
FORENSIC_PNG_LEVEL = env_int('FORENSIC_PNG_LEVEL', 3)

# --------------------------------------------------------------------- API
DEEPSHIELD_VERSION = '2.0.0'
# HMAC key for report signatures (default: SECRET_KEY, see detector/reports.py)
REPORT_SIGNING_KEY = env_str('REPORT_SIGNING_KEY', '')
# CORS: explicit origins only (never allow-all). Native Android clients do not need CORS.
CORS_ALLOWED_ORIGINS = env_list('CORS_ALLOWED_ORIGINS', [])
CORS_ALLOW_ALL_ORIGINS = False
CORS_ALLOW_CREDENTIALS = False

REST_FRAMEWORK = {
    'DEFAULT_VERSIONING_CLASS': 'rest_framework.versioning.URLPathVersioning',
    'DEFAULT_VERSION': 'v1', 'ALLOWED_VERSIONS': ['v1'],
    'DEFAULT_AUTHENTICATION_CLASSES': ['rest_framework_simplejwt.authentication.JWTAuthentication'],
    'DEFAULT_PERMISSION_CLASSES': ['rest_framework.permissions.IsAuthenticated'],
    'DEFAULT_RENDERER_CLASSES': ['detector.api.envelope.EnvelopeJSONRenderer'],
    'DEFAULT_PARSER_CLASSES': ['rest_framework.parsers.JSONParser', 'rest_framework.parsers.MultiPartParser'],
    'EXCEPTION_HANDLER': 'detector.api.envelope.exception_handler',
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
    'DEFAULT_SCHEMA_CLASS': 'drf_spectacular.openapi.AutoSchema',
    'NUM_PROXIES': NUM_PROXIES or None,
}
from datetime import timedelta  # noqa: E402
SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=env_int('JWT_ACCESS_MINUTES', 15)),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=env_int('JWT_REFRESH_DAYS', 7)),
    'ROTATE_REFRESH_TOKENS': True,          # every refresh issues a new refresh token ...
    'BLACKLIST_AFTER_ROTATION': True,       # ... and revokes the old one
    'UPDATE_LAST_LOGIN': True,
    'AUTH_HEADER_TYPES': ('Bearer',),
}
SPECTACULAR_SETTINGS = {
    'TITLE': 'DeepShield API', 'VERSION': '2.0.0',
    'DESCRIPTION': (
        'Face-deepfake detection and digital image forensics.\n\n'
        '**Authentication:** `POST /api/v1/auth/login/` returns a short-lived access token (Bearer) and a '
        'rotating refresh token.\n\n**Envelope:** every response is `{success, data, error, request_id}`.\n\n'
        '**Upload limits:** JPEG/PNG only, 10 MB, 8000 px per side, 25 MP.\n\n'
        '**Analysis states:** pending, processing, completed, no_face, failed.\n\n'
        '**Interpretation:** results are probabilistic; forensic indicators are supporting evidence, never proof.\n\n'
        '**Privacy:** by using the API you confirm you may analyse the images you upload and accept the Privacy Notice '
        '(`/privacy/`). Analyses can be deleted with `DELETE /analyses/{id}/`; accounts with `DELETE /auth/account/`.'),
    'SERVE_INCLUDE_SCHEMA': False, 'COMPONENT_SPLIT_REQUEST': True,
    'SCHEMA_PATH_PREFIX': r'/api/v[0-9]',
}
READINESS_MIN_MODELS = env_int('READINESS_MIN_MODELS', 0)   # production default: 1

# ------------------------------------------------------------------ CSP
CSP_REPORT_ONLY = env_bool('CSP_REPORT_ONLY', False)
CSP_IMG_SOURCES = env_list('CSP_IMG_SOURCES', [])

# ------------------------------------------------------------------ misc
MESSAGE_TAGS = {
    messages.DEBUG: 'secondary',
    messages.INFO: 'info',
    messages.SUCCESS: 'success',
    messages.WARNING: 'warning',
    messages.ERROR: 'danger',
}

SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'
CSRF_COOKIE_SAMESITE = 'Lax'
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = 'same-origin'
X_FRAME_OPTIONS = 'DENY'
SESSION_COOKIE_AGE = env_int('SESSION_COOKIE_AGE', 60 * 60 * 12)   # 12 h

# ----------------------------------------------------------------- logging
# JSON lines to stdout (collected by Docker / any log shipper). 'text' for local reading.
import sys  # noqa: E402
LOG_LEVEL = env_str('LOG_LEVEL', 'WARNING' if 'test' in sys.argv else 'INFO').upper()
LOG_FORMAT = env_str('LOG_FORMAT', 'text').lower()
from config.logging_conf import build_logging  # noqa: E402
LOGGING = build_logging(LOG_LEVEL, LOG_FORMAT)

# ------------------------------------------------------- privacy / retention
# Days to keep an analysis (original, crops, heat-maps, forensic maps, report) after upload.
# 0 = keep until the user deletes it. Run `manage.py purge_expired` daily (scheduler / cron).
DELETE_ANALYSIS_AFTER_DAYS = env_int('DELETE_ANALYSIS_AFTER_DAYS', 0)
AUDIT_LOG_RETENTION_DAYS = env_int('AUDIT_LOG_RETENTION_DAYS', 365)

# ------------------------------------------------------------ observability
# /metrics requires "Authorization: Bearer <METRICS_TOKEN>" (or a staff session). Empty = disabled.
METRICS_TOKEN = env_str('METRICS_TOKEN')

