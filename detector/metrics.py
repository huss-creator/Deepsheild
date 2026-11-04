"""
Prometheus metrics (optional dependency: prometheus-client).

Exposed at GET /metrics, protected by METRICS_TOKEN (Bearer) or a staff session; with neither
configured the endpoint returns 404. Labels are low-cardinality (status, model name) — never user
names, file names or IDs. If prometheus-client is not installed every function is a no-op.

Multi-process note: Gunicorn workers and Celery workers each hold their own registry. Set
PROMETHEUS_MULTIPROC_DIR (shared, empty dir) to aggregate across Gunicorn workers; scrape the Celery
worker separately or rely on the DB-derived gauges below (queue depth, jobs by stage), which are
computed from the database at scrape time and are therefore always correct.
"""

import hmac
import logging

from django.conf import settings
from django.http import Http404, HttpResponse

logger = logging.getLogger('deepshield.metrics')

try:
    from prometheus_client import (CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram,
                                   generate_latest, multiprocess)
    _AVAILABLE = True
except ImportError:                                           # pragma: no cover
    _AVAILABLE = False

if _AVAILABLE:
    JOBS = Counter('deepshield_jobs_finished_total', 'Analysis jobs finished', ['status'])
    JOB_SECONDS = Histogram('deepshield_job_duration_seconds', 'Wall time of analysis jobs',
                            buckets=(1, 2, 5, 10, 20, 40, 80, 160, 320))
    HTTP = Counter('deepshield_http_requests_total', 'HTTP requests', ['method', 'status_class'])
    MODEL_FAIL = Counter('deepshield_model_failures_total', 'Model inference failures', ['model'])
    UPLOAD_REJECTED = Counter('deepshield_uploads_rejected_total', 'Uploads rejected', ['reason'])
    RETENTION_DELETED = Counter('deepshield_retention_deleted_total', 'Analyses deleted by retention policy')


def record_job_finished(status, seconds):
    if _AVAILABLE and status:
        JOBS.labels(status=str(status)).inc()
        JOB_SECONDS.observe(max(seconds, 0))


def record_http(method, status):
    if _AVAILABLE:
        HTTP.labels(method=method if method in ('GET', 'POST', 'DELETE', 'PUT', 'PATCH') else 'OTHER',
                    status_class=f'{int(status) // 100}xx').inc()


def record_model_failure(model):
    if _AVAILABLE:
        MODEL_FAIL.labels(model=str(model)[:40]).inc()


def record_upload_rejected(reason):
    if _AVAILABLE:
        UPLOAD_REJECTED.labels(reason=str(reason)[:40]).inc()


def record_retention_deleted(n):
    if _AVAILABLE and n:
        RETENTION_DELETED.inc(n)


def _db_gauges(registry):
    """Always-correct, DB-derived gauges (valid whichever process answers the scrape)."""
    from django.db.models import Count
    from detector.models import AnalysisJob as J
    g = Gauge('deepshield_jobs_by_status', 'Jobs currently in each state', ['status'], registry=registry)
    for row in J.objects.values('status').annotate(n=Count('id')):
        g.labels(status=row['status']).set(row['n'])
    q = Gauge('deepshield_queue_depth', 'Jobs waiting for a worker', registry=registry)
    q.set(J.objects.filter(status=J.QUEUED).count())


class MetricsMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        record_http(request.method, response.status_code)
        return response


def _authorised(request) -> bool:
    token = getattr(settings, 'METRICS_TOKEN', '')
    header = request.META.get('HTTP_AUTHORIZATION', '')
    if token and header.startswith('Bearer ') and hmac.compare_digest(header[7:], token):
        return True
    user = getattr(request, 'user', None)
    return bool(user and user.is_authenticated and user.is_staff)


def metrics_view(request):
    if not _AVAILABLE or not (getattr(settings, 'METRICS_TOKEN', '') or request.user.is_staff):
        raise Http404
    if not _authorised(request):
        raise Http404                      # do not reveal that the endpoint exists
    import os
    if os.environ.get('PROMETHEUS_MULTIPROC_DIR'):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
    else:
        from prometheus_client import REGISTRY as registry
    out = generate_latest(registry)
    db_registry = CollectorRegistry()
    try:
        _db_gauges(db_registry)
        out += generate_latest(db_registry)
    except Exception:
        logger.exception('could not collect DB gauges')
    resp = HttpResponse(out, content_type=CONTENT_TYPE_LATEST)
    resp['Cache-Control'] = 'no-store'
    return resp
