"""Per-request / per-job logging context (contextvars, safe across threads and Celery tasks)."""

import contextlib
import contextvars

job_id_var = contextvars.ContextVar('job_id', default='')


@contextlib.contextmanager
def job_context(job_id):
    token = job_id_var.set(str(job_id))
    try:
        yield
    finally:
        job_id_var.reset(token)
