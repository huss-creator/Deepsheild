"""Celery tasks. All state is persisted by detector.jobs; the task only drives it."""

import logging

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from celery.signals import worker_process_init
from django.conf import settings

from detector import jobs

logger = logging.getLogger('deepshield.jobs')


@worker_process_init.connect
def preload_models(**kwargs):
    """Load the model registry once per worker process (after fork), so the first job is not slow
    and model problems show up in the worker log at startup."""
    try:
        from ai_models.registry import get_registry
        for h in get_registry().health():
            logger.info('worker model %s: %s', h['key'], h['status'])
    except Exception:
        logger.exception('model preload failed (jobs will report model status individually)')


@shared_task(bind=True, name='detector.analyze_upload', max_retries=settings.JOB_MAX_RETRIES,
             soft_time_limit=settings.CELERY_TASK_SOFT_TIME_LIMIT, time_limit=settings.CELERY_TASK_TIME_LIMIT)
def analyze_upload(self, job_id):
    attempt = self.request.retries + 1
    try:
        jobs.run_job(job_id, attempt=attempt)
    except jobs.TransientJobError as exc:
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=jobs.backoff(self.request.retries))
        jobs.fail_job(job_id, 'RETRIES_EXHAUSTED')
    except SoftTimeLimitExceeded:
        jobs.fail_job(job_id, 'TIMEOUT')
