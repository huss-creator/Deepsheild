"""
Background-job lifecycle for analyses.

  submit()  -> limits check, save upload, create AnalysisJob(QUEUED), enqueue Celery task
  run_job() -> executed by the worker; moves through the real stages
               VALIDATING -> FACE_DETECTION -> AI_ANALYSIS -> FORENSIC_ANALYSIS -> AGGREGATING
               -> REPORT_GENERATION -> COMPLETED (or FAILED / CANCELLED)
  request_cancel(), reap_stuck_jobs(), requeue_failed()

Failure policy
  * one AI model / forensic module failing never fails the job (job.partial + warnings);
  * transient infrastructure errors (DB connection, I/O timeouts) are retried with backoff;
  * everything else is a permanent failure with a SAFE error code (no exception text stored);
  * restarts are idempotent: partial results of a previous attempt are wiped first.
"""

import logging
import uuid
from datetime import timedelta

from django.conf import settings
from django.db import OperationalError
from django.utils import timezone

from detector.models import AnalysisJob as J
from detector.models import Analysis, UploadedImage

logger = logging.getLogger('deepshield.jobs')

ERROR_TEXT = {
    'INVALID_IMAGE': 'The stored image failed validation.',
    'PROCESSING_ERROR': 'Processing failed. Please try again or contact support.',
    'TIMEOUT': 'Processing exceeded the time limit.',
    'QUEUE_UNAVAILABLE': 'The analysis queue is unavailable. Please try again later.',
    'WORKER_LOST': 'The processing worker stopped unexpectedly.',
    'RETRIES_EXHAUSTED': 'Processing failed after several attempts.',
    'CANCELLED': 'Cancelled by user.',
}


class JobLimitError(Exception):
    def __init__(self, code, message, status):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


class TransientJobError(Exception):
    """Infrastructure hiccup worth retrying."""


class PermanentJobError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


class JobCancelled(Exception):
    pass


def _now():
    return timezone.now()


# ------------------------------------------------------------------ submit
def submit(user, uploaded_file, original_filename) -> UploadedImage:
    if J.objects.filter(upload__user=user, status__in=J.ACTIVE).count() >= settings.MAX_ACTIVE_JOBS_PER_USER:
        raise JobLimitError('TOO_MANY_ACTIVE_JOBS',
                            f'You already have {settings.MAX_ACTIVE_JOBS_PER_USER} analyses in progress. '
                            'Wait for one to finish or cancel it.', 429)
    if J.objects.filter(status=J.QUEUED).count() >= settings.MAX_QUEUED_JOBS:
        raise JobLimitError('QUEUE_FULL', 'The analysis queue is full. Please retry later.', 503)
    import hashlib
    h = hashlib.sha256()
    uploaded_file.seek(0)
    for chunk in uploaded_file.chunks():
        h.update(chunk)
    uploaded_file.seek(0)
    rec = UploadedImage(user=user, image=uploaded_file, original_filename=original_filename,
                        file_size=uploaded_file.size, sha256_hash=h.hexdigest(), processing_status=UploadedImage.STATUS_PENDING)
    rec.save()                                    # file stored under a server-generated name
    job = J.objects.create(upload=rec, celery_task_id=uuid.uuid4().hex)
    enqueue(job)
    return UploadedImage.objects.get(pk=rec.pk)


def enqueue(job):
    from detector.tasks import analyze_upload
    if settings.CELERY_TASK_ALWAYS_EAGER and getattr(settings, 'JOB_RUN_IN_THREAD', False):
        _start_thread(job.pk)                    # no broker (local development): run in the background
        return
    try:
        analyze_upload.apply_async(args=[job.pk], task_id=job.celery_task_id or None)
    except Exception:
        logger.exception('could not enqueue job %s', job.pk)
        fail_job(job.pk, 'QUEUE_UNAVAILABLE')


def _start_thread(job_id):
    """Local development without Redis/Celery: run the job in a background thread so the request
    returns at once and the processing page can show real progress. Production uses Celery."""
    import threading

    def _target():
        from django.db import connection
        try:
            run_job(job_id)
        except Exception:
            logger.exception('background job %s crashed', job_id)
            fail_job(job_id, 'PROCESSING_ERROR')
        finally:
            connection.close()

    threading.Thread(target=_target, name=f'ds-job-{job_id}', daemon=True).start()


# ------------------------------------------------------------------ state helpers
def _update(job_id, **fields):
    return J.objects.filter(pk=job_id).exclude(status__in=J.TERMINAL).update(updated_at=_now(), **fields)


def is_cancel_requested(job_id) -> bool:
    row = J.objects.filter(pk=job_id).values('cancel_requested', 'status').first()
    return bool(row) and (row['cancel_requested'] or row['status'] == J.CANCELLED)


LIVE_MAX_EVENTS = 40


def push_live(job_id, kind, data):
    """Append a progress event for the processing page. Best effort: never raises, never stores scores."""
    try:
        row = J.objects.filter(pk=job_id).values('live', 'started_at').first()
        if row is None:
            return
        live = dict(row['live'] or {})
        events = list(live.get('events') or [])
        t = round((_now() - (row['started_at'] or _now())).total_seconds(), 1)
        if kind == 'faces':
            live['image'] = data.get('image')
            live['detector'] = data.get('detector')
            live['faces'] = data.get('faces') or []
            n = len(live['faces'])
            events.append({'t': t, 'level': 'ok' if n else 'warn',
                           'msg': (f'Face detector ({data.get("detector")}): {n} face{"s" if n != 1 else ""} found'
                                   if n else 'Face detector: no face found')})
        elif kind == 'model':
            if data.get('status') == 'AVAILABLE':
                events.append({'t': t, 'level': 'ok', 'msg': f'{data.get("name")} finished'})
            else:
                events.append({'t': t, 'level': 'warn', 'msg': f'{data.get("name")} unavailable'})
        elif kind == 'forensic':
            events.append({'t': t, 'level': 'ok', 'msg': f'Forensic test done: {data.get("name")}'})
        elif kind == 'msg':
            events.append({'t': t, 'level': data.get('level', 'ok'), 'msg': data.get('msg', '')})
        live['events'] = events[-LIVE_MAX_EVENTS:]
        J.objects.filter(pk=job_id).exclude(status__in=J.TERMINAL).update(live=live)
    except Exception:
        logger.debug('could not store live progress for job %s', job_id, exc_info=True)


def set_stage(job_id, stage):
    """Record the stage the worker is entering; honours a pending cancellation."""
    if is_cancel_requested(job_id):
        raise JobCancelled()
    _update(job_id, status=stage)


def fail_job(job_id, code, message=None):
    job = J.objects.filter(pk=job_id).select_related('upload').first()
    if job is None or job.is_terminal:
        return
    J.objects.filter(pk=job_id).exclude(status__in=J.TERMINAL).update(
        status=J.FAILED, last_stage=job.status if job.status in J.STAGES else job.last_stage, error_code=code,
        error_message=message or ERROR_TEXT.get(code, ERROR_TEXT['PROCESSING_ERROR']),
        finished_at=_now(), updated_at=_now())
    UploadedImage.objects.filter(pk=job.upload_id).update(
        processing_status=UploadedImage.STATUS_FAILED,
        error_message=message or ERROR_TEXT.get(code, ERROR_TEXT['PROCESSING_ERROR']))
    logger.warning('job %s failed: %s', job_id, code)


def _finish_cancel(job_id):
    job = J.objects.filter(pk=job_id).select_related('upload').first()
    if job is None:
        return
    Analysis.objects.filter(uploaded_image_id=job.upload_id).delete()        # wipe partial results + files
    last = job.status if job.status in J.STAGES else job.last_stage
    J.objects.filter(pk=job_id).update(status=J.CANCELLED, last_stage=last, error_code='CANCELLED',
                                       error_message=ERROR_TEXT['CANCELLED'], finished_at=_now(), updated_at=_now())
    UploadedImage.objects.filter(pk=job.upload_id).update(processing_status=UploadedImage.STATUS_CANCELLED,
                                                          error_message='')


def request_cancel(job) -> str:
    """Returns 'cancelled' (was still queued) or 'cancelling' (running; stops at the next checkpoint).
    Raises ValueError if the job already finished."""
    if job.is_terminal:
        raise ValueError('job already finished')
    updated = J.objects.filter(pk=job.pk, status=J.QUEUED).update(
        status=J.CANCELLED, cancel_requested=True, last_stage=J.QUEUED, error_code='CANCELLED',
        error_message=ERROR_TEXT['CANCELLED'], finished_at=_now(), updated_at=_now())
    if updated:
        UploadedImage.objects.filter(pk=job.upload_id).update(processing_status=UploadedImage.STATUS_CANCELLED)
        if not settings.CELERY_TASK_ALWAYS_EAGER and job.celery_task_id:
            try:
                from config.celery import app
                app.control.revoke(job.celery_task_id)                # best effort; worker also checks state
            except Exception:
                logger.exception('revoke failed for job %s', job.pk)
        return 'cancelled'
    if not _update(job.pk, cancel_requested=True):
        raise ValueError('job already finished')
    return 'cancelling'


# ------------------------------------------------------------------ worker side
def backoff(retries: int) -> int:
    return min(settings.JOB_RETRY_BACKOFF * (2 ** retries), 300)


def _revalidate(path):
    """Defence in depth: the worker re-validates the STORED bytes before decoding them."""
    from django.core.exceptions import ValidationError
    from detector.image_validation import validate_image_bytes
    try:
        with open(path, 'rb') as fh:
            validate_image_bytes(fh.read(settings.MAX_UPLOAD_SIZE + 1))
    except ValidationError:
        raise PermanentJobError('INVALID_IMAGE')


def run_job(job_id, attempt=1):
    """Run one analysis job with log correlation (job_id) and metrics around it."""
    from detector import metrics
    from detector.logctx import job_context
    started = _now()
    with job_context(job_id):
        try:
            _run_job(job_id, attempt)
        finally:
            metrics.record_job_finished(J.objects.filter(pk=job_id).values_list('status', flat=True).first(),
                                        (_now() - started).total_seconds())


def _run_job(job_id, attempt=1):
    from ai_models.pipeline import PipelineCancelled

    job = J.objects.select_related('upload').filter(pk=job_id).first()
    if job is None or job.is_terminal:
        return                                           # duplicate delivery / already cancelled: no-op
    if job.cancel_requested:
        _finish_cancel(job_id)
        return
    upload = job.upload
    Analysis.objects.filter(uploaded_image=upload).delete()      # idempotent restart
    J.objects.filter(pk=job_id).update(attempts=attempt, started_at=job.started_at or _now(),
                                       status=J.VALIDATING, live={}, updated_at=_now())
    t_start = _now()
    cancel = lambda: is_cancel_requested(job_id)
    try:
        set_stage(job_id, J.VALIDATING)
        from detector import storage as ds_storage
        with ds_storage.local_copy(upload.image) as path:
            _process(job_id, upload, path, t_start, cancel)
        ds_storage.publish_artifacts(upload)
    except (JobCancelled, PipelineCancelled):
        _finish_cancel(job_id)
    except PermanentJobError as exc:
        fail_job(job_id, exc.code)
    except (OperationalError, ConnectionError, TimeoutError) as exc:
        logger.warning('transient error in job %s: %s', job_id, type(exc).__name__)
        raise TransientJobError() from exc
    except Exception:
        logger.exception('job %s failed', job_id)
        fail_job(job_id, 'PROCESSING_ERROR')


def _process(job_id, upload, path, t_start, cancel):
    """Pipeline body; runs on a local file (downloaded scratch copy when storage is remote)."""
    import os
    from ai_models.pipeline import run_ai_analysis
    from ai_models.preprocessing import calculate_sha256
    from detector import services
    from detector.reports import create_report
    _revalidate(path)
    digest = calculate_sha256(path)
    if upload.sha256_hash and upload.sha256_hash != digest:      # file changed between upload and analysis
        logger.error('integrity mismatch for upload %s', upload.pk)
        raise PermanentJobError('INVALID_IMAGE')
    upload.sha256_hash = digest
    upload.processing_status = UploadedImage.STATUS_PROCESSING
    upload.save(update_fields=['sha256_hash', 'processing_status'])

    from detector.models import user_media_prefix
    udir = lambda sub: os.path.join(settings.MEDIA_ROOT, *user_media_prefix(upload.user_id, sub).split('/'))  # noqa: E731
    ai = run_ai_analysis(path, faces_dir=udir('faces'),
                         heatmaps_dir=udir('heatmaps'),
                         processed_dir=udir('processed'),
                         on_stage=lambda s: set_stage(job_id, s), should_cancel=cancel,
                         on_event=lambda k, d: push_live(job_id, k, d))
    if ai['status'] == 'failed':
        raise PermanentJobError('INVALID_IMAGE')
    upload.face_detected = ai['face_count'] > 0
    upload.face_count = ai['face_count']
    if ai['faces']:
        upload.cropped_face = ai['faces'][0]['crop_path']                 # largest face (legacy field)
    upload.processing_status = (UploadedImage.STATUS_NO_FACE if ai['status'] == 'no_face'
                                else UploadedImage.STATUS_COMPLETED)
    upload.save()
    analysis = services.persist_ai_results(upload, ai)

    set_stage(job_id, J.FORENSIC_ANALYSIS)
    failed_modules = services._run_forensic_analysis(
        analysis, path, on_stage=lambda s: set_stage(job_id, J.AGGREGATING),
        on_event=lambda k, d: push_live(job_id, k, d)) or []
    if is_cancel_requested(job_id):
        raise JobCancelled()
    set_stage(job_id, J.REPORT_GENERATION)
    create_report(analysis)

    warnings = []
    bad = {}
    for f in ai['faces']:
        for o in f['model_outputs']:
            if o['status'] != 'AVAILABLE':
                bad[o['name']] = o['status']
    if not ai['faces']:                                   # nothing ran: report configuration state
        bad.update({m['name']: m['status'] for m in (ai.get('models') or []) if m['status'] != 'AVAILABLE'})
    if bad:
        warnings.append(f"{len(bad)} AI model(s) unavailable: {', '.join(sorted(bad))}")
    if failed_modules:
        warnings.append(f"Forensic module(s) without a result: {', '.join(failed_modules)}")
    J.objects.filter(pk=job_id).update(
        status=J.COMPLETED, partial=bool(warnings), warnings=warnings, finished_at=_now(), updated_at=_now(),
        timings_ms={**ai['timings_ms'], 'job_total_ms': round((_now() - t_start).total_seconds() * 1000, 1)})


# ------------------------------------------------------------------ operations
def reap_stuck_jobs(now=None):
    """Jobs that stopped updating (worker died, message lost). Re-queue while attempts remain,
    otherwise fail them permanently as WORKER_LOST (the dead-letter state, visible in admin/API)."""
    cutoff = (now or _now()) - timedelta(seconds=settings.STUCK_JOB_SECONDS)
    requeued = failed = 0
    for job in J.objects.filter(status__in=J.ACTIVE, updated_at__lt=cutoff):
        if job.attempts <= settings.JOB_MAX_RETRIES:
            J.objects.filter(pk=job.pk).update(status=J.QUEUED, updated_at=_now(), celery_task_id=uuid.uuid4().hex)
            enqueue(J.objects.get(pk=job.pk))
            requeued += 1
        else:
            fail_job(job.pk, 'WORKER_LOST')
            failed += 1
    return {'requeued': requeued, 'failed': failed}


def requeue_failed(job_ids=None, code=None):
    """Operator action: give failed jobs another full run."""
    qs = J.objects.filter(status=J.FAILED).select_related('upload')
    if job_ids:
        qs = qs.filter(pk__in=job_ids)
    if code:
        qs = qs.filter(error_code=code)
    n = 0
    for job in qs:
        J.objects.filter(pk=job.pk).update(status=J.QUEUED, last_stage='', error_code='', error_message='',
                                           finished_at=None, cancel_requested=False, attempts=0,
                                           celery_task_id=uuid.uuid4().hex, updated_at=_now())
        UploadedImage.objects.filter(pk=job.upload_id).update(processing_status=UploadedImage.STATUS_PENDING,
                                                              error_message='')
        enqueue(J.objects.get(pk=job.pk))
        n += 1
    return n
