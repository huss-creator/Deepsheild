"""
Phase 7 tests: background jobs — real stage reporting, 202/polling contract, cancellation
(queued and running), retries, timeouts, permanent failures, partial failures, idempotency,
concurrency/queue limits, stuck-job recovery, dead-letter requeue, worker/broker configuration.
Celery runs eagerly (in-process); "worker not yet run" is simulated by patching apply_async.
"""

import json
import os
import subprocess
import sys
from datetime import timedelta
from io import StringIO
from pathlib import Path
from unittest import mock

from celery.exceptions import SoftTimeLimitExceeded
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.db import OperationalError
from django.test import override_settings
from django.utils import timezone

from detector import jobs
from detector.models import Analysis, AnalysisJob as J, AuditLog, ModelResult, UploadedImage
from detector.tests_api import BASE, PW, ApiTestCase, two_face_png
from detector.tests_ai import StubDetector
from ai_models.face_detection import Face as DetectedFace
from detector.tests_security import make_image_bytes

QUEUE_NOOP = 'detector.tasks.analyze_upload.apply_async'


class JobTestCase(ApiTestCase):
    def png_file(self, name='in.png'):
        path = two_face_png(os.path.join(self.dir, name))
        return SimpleUploadedFile('upload.png', open(path, 'rb').read())

    def faces_patch(self, n=2):
        faces = [DetectedFace(0, 50, 50, 140, 140), DetectedFace(1, 250, 60, 120, 120)][:n]
        return mock.patch('ai_models.pipeline.get_face_detector', return_value=StubDetector(faces))

    def submit_queued(self, user=None):
        """Create a job but do not run it (as if no worker has picked it up yet)."""
        with mock.patch(QUEUE_NOOP):
            return jobs.submit(user or self.alice, self.png_file(), 'q.png')

    def run_and_refresh(self, rec):
        with self.faces_patch():
            jobs.run_job(rec.job.pk)
        rec.refresh_from_db()
        rec.job.refresh_from_db()
        return rec


class StageTests(JobTestCase):
    def test_job_moves_through_the_real_stages_in_order(self):
        seen = []
        real = jobs.set_stage
        with mock.patch('detector.jobs.set_stage', side_effect=lambda jid, st: (seen.append(st), real(jid, st))[1]), \
                self.faces_patch():
            rec = jobs.submit(self.alice, self.png_file(), 'two.png')
        job = J.objects.get(upload=rec)
        order = [s for i, s in enumerate(seen) if s not in seen[:i]]
        self.assertEqual(order, ['VALIDATING', 'FACE_DETECTION', 'AI_ANALYSIS', 'FORENSIC_ANALYSIS',
                                 'AGGREGATING', 'REPORT_GENERATION'])
        self.assertEqual((job.status, job.attempts), ('COMPLETED', 1))
        self.assertTrue(job.started_at and job.finished_at and job.finished_at >= job.started_at)
        self.assertIn('job_total_ms', job.timings_ms)
        self.assertEqual(rec.processing_status, UploadedImage.STATUS_COMPLETED)
        self.assertTrue(job.partial)                                  # four of seven model slots are unavailable here
        self.assertTrue(any('AI model(s) unavailable' in w for w in job.warnings))

    def test_api_returns_202_and_polling_shows_real_stage_without_percentages(self):
        with mock.patch(QUEUE_NOOP):
            r = self.call('post', '/analyses/', self.alice,
                          data={'image': SimpleUploadedFile('p.png', make_image_bytes('PNG'), content_type='image/png')})
        self.assertEqual(r.status_code, 202)
        d = r.json()['data']
        self.assertEqual((d['status'], d['assessment'], d['job']['stage']), ('pending', None, 'QUEUED'))
        self.assertEqual((d['job']['stage_index'], d['job']['stage_total'], d['job']['finished']), (1, 8, False))
        st = self.call('get', f"/analyses/{d['id']}/status/", self.alice).json()['data']
        self.assertEqual(st['job']['stage'], 'QUEUED')
        self.assertNotIn('percent', json.dumps(st).lower())               # never an invented progress number
        for sub in ('faces', 'models', 'forensics', 'report'):
            e = self.call('get', f"/analyses/{d['id']}/{sub}/", self.alice)
            self.assertEqual((e.status_code, e.json()['error']['code']), (409, 'NOT_AVAILABLE'), sub)
        # worker finally runs
        rec = UploadedImage.objects.get(public_id=d['id'])
        with self.faces_patch(0):
            jobs.run_job(rec.job.pk)
        st = self.call('get', f"/analyses/{d['id']}/status/", self.alice).json()['data']
        self.assertEqual((st['status'], st['job']['stage'], st['job']['finished']), ('no_face', 'COMPLETED', True))

    def test_eager_mode_completes_before_responding(self):
        r = self.call('post', '/analyses/', self.alice,
                      data={'image': SimpleUploadedFile('p.png', make_image_bytes('PNG'), content_type='image/png')})
        self.assertEqual(r.json()['data']['job']['stage'], 'COMPLETED')

    def test_status_endpoint_is_owner_only(self):
        rec = self.submit_queued()
        r = self.call('get', f'/analyses/{rec.public_id}/status/', self.bob)
        self.assertEqual((r.status_code, r.json()['error']['code']), (404, 'NOT_FOUND'))
        self.assertTrue(AuditLog.objects.filter(action='access_denied', user=self.bob).exists())


class CancellationTests(JobTestCase):
    def test_cancel_queued_job_is_immediate_and_late_worker_is_a_noop(self):
        rec = self.submit_queued()
        r = self.call('post', f'/analyses/{rec.public_id}/cancel/', self.alice)
        d = r.json()['data']
        self.assertEqual((r.status_code, d['outcome'], d['status']), (200, 'cancelled', 'cancelled'))
        self.assertEqual(d['job']['error']['code'], 'CANCELLED')
        with self.faces_patch():
            jobs.run_job(rec.job.pk)                                       # late delivery of the old message
        self.assertEqual(Analysis.objects.count(), 0)
        self.assertEqual(J.objects.get(pk=rec.job.pk).status, 'CANCELLED')
        again = self.call('post', f'/analyses/{rec.public_id}/cancel/', self.alice)
        self.assertEqual((again.status_code, again.json()['error']['code']), (409, 'JOB_FINISHED'))
        self.assertTrue(AuditLog.objects.filter(action='analysis_cancelled', user=self.alice).exists())

    def test_other_user_cannot_cancel(self):
        rec = self.submit_queued()
        r = self.call('post', f'/analyses/{rec.public_id}/cancel/', self.bob)
        self.assertEqual(r.status_code, 404)
        self.assertEqual(J.objects.get(pk=rec.job.pk).status, 'QUEUED')

    def _cancel_when_entering(self, stage):
        rec = self.submit_queued()
        real = jobs.set_stage
        fired = []

        def hook(job_id, st):
            if st == stage and not fired:
                fired.append(1)
                outcome = jobs.request_cancel(J.objects.get(pk=job_id))     # as if the API was called now
                assert outcome == 'cancelling'
            return real(job_id, st)
        with mock.patch('detector.jobs.set_stage', side_effect=hook):
            return self.run_and_refresh(rec)

    def test_cancel_while_running_stops_and_wipes_partial_results(self):
        # cancel is detected when the worker tries to ENTER `stage`; last_stage is the last stage it did enter
        for stage, last in (('AI_ANALYSIS', 'FACE_DETECTION'), ('FORENSIC_ANALYSIS', 'AI_ANALYSIS'),
                            ('REPORT_GENERATION', 'AGGREGATING')):
            with self.subTest(stage=stage):
                rec = self._cancel_when_entering(stage)
                self.assertEqual((rec.job.status, rec.job.last_stage, rec.job.error_code), ('CANCELLED', last, 'CANCELLED'))
                self.assertEqual(rec.processing_status, UploadedImage.STATUS_CANCELLED)
                self.assertFalse(Analysis.objects.filter(uploaded_image=rec).exists())     # partial results gone
                self.assertEqual(ModelResult.objects.count(), 0)
                faces_dir = Path(self.media) / 'faces'
                self.assertEqual(list(faces_dir.glob('*')) if faces_dir.exists() else [], [])   # and their files
                rec.delete()

    def test_cancel_endpoint_on_running_job_reports_cancelling(self):
        rec = self.submit_queued()
        J.objects.filter(pk=rec.job.pk).update(status='AI_ANALYSIS')
        r = self.call('post', f'/analyses/{rec.public_id}/cancel/', self.alice).json()['data']
        self.assertEqual(r['outcome'], 'cancelling')
        self.assertTrue(r['job']['cancel_requested'])
        self.assertEqual(r['job']['stage'], 'AI_ANALYSIS')                  # not claimed cancelled until it stops


class LimitTests(JobTestCase):
    def test_active_jobs_per_user_limit(self):
        for _ in range(settings.MAX_ACTIVE_JOBS_PER_USER):
            self.submit_queued()
        with mock.patch(QUEUE_NOOP):
            r = self.call('post', '/analyses/', self.alice,
                          data={'image': SimpleUploadedFile('p.png', make_image_bytes('PNG'), content_type='image/png')})
        self.assertEqual((r.status_code, r.json()['error']['code']), (429, 'TOO_MANY_ACTIVE_JOBS'))
        self.assertEqual(UploadedImage.objects.filter(user=self.alice).count(), settings.MAX_ACTIVE_JOBS_PER_USER)
        self.submit_queued(self.bob)                                         # other users are unaffected
        first = J.objects.filter(upload__user=self.alice).first()
        jobs.request_cancel(first)
        self.submit_queued()                                                 # a slot is free again

    @override_settings(MAX_QUEUED_JOBS=1)
    def test_queue_full_backpressure(self):
        self.submit_queued(self.alice)
        with mock.patch(QUEUE_NOOP):
            r = self.call('post', '/analyses/', self.bob,
                          data={'image': SimpleUploadedFile('p.png', make_image_bytes('PNG'), content_type='image/png')})
        self.assertEqual((r.status_code, r.json()['error']['code']), (503, 'QUEUE_FULL'))

    def test_web_upload_respects_the_limit(self):
        for _ in range(settings.MAX_ACTIVE_JOBS_PER_USER):
            self.submit_queued()
        self.client.login(username='alice', password=PW)
        with mock.patch(QUEUE_NOOP):
            r = self.client.post('/auth/upload/', {'image': SimpleUploadedFile('p.png', make_image_bytes('PNG'), content_type='image/png'), 'consent': 'on'}, follow=True)
        self.assertIn('in progress', r.content.decode())


class FailureTests(JobTestCase):
    def test_permanent_failure_is_safe_and_not_retried(self):
        calls = []
        def boom(*a, **k):
            calls.append(1)
            raise RuntimeError('/srv/secret/path exploded')
        with mock.patch('detector.services.persist_ai_results', side_effect=boom), self.faces_patch():
            rec = jobs.submit(self.alice, self.png_file(), 'x.png')
        job = J.objects.get(upload=rec)
        self.assertEqual((job.status, job.error_code, job.attempts, len(calls)), ('FAILED', 'PROCESSING_ERROR', 1, 1))
        self.assertNotIn('secret', job.error_message + rec.error_message)
        self.assertEqual(job.last_stage, 'AI_ANALYSIS')
        self.assertEqual(UploadedImage.objects.get(pk=rec.pk).processing_status, 'failed')

    def test_stored_file_is_revalidated_by_the_worker(self):
        rec = self.submit_queued()
        # decodes perfectly, so ONLY the validator (not the decoder) can reject it: valid PNG + PHP trailer
        Path(rec.image.path).write_bytes(make_image_bytes('PNG') + b'<?php system($_GET[0]); ?>')
        with self.faces_patch():
            jobs.run_job(rec.job.pk)
        job = J.objects.get(pk=rec.job.pk)
        self.assertEqual((job.status, job.error_code), ('FAILED', 'INVALID_IMAGE'))

    def test_transient_error_is_retried_then_succeeds(self):
        calls = []
        real = jobs.__dict__['Analysis'].objects  # noqa (keeps lint quiet)
        from detector import services
        orig = services.persist_ai_results
        def flaky(*a, **k):
            calls.append(1)
            if len(calls) == 1:
                raise OperationalError('database is locked')
            return orig(*a, **k)
        with mock.patch('detector.services.persist_ai_results', side_effect=flaky), self.faces_patch():
            rec = jobs.submit(self.alice, self.png_file(), 'x.png')
        job = J.objects.get(upload=rec)
        self.assertEqual((job.status, job.attempts), ('COMPLETED', 2))
        self.assertEqual(Analysis.objects.filter(uploaded_image=rec).count(), 1)     # no duplicate results

    def test_retries_are_bounded(self):
        with mock.patch('detector.services.persist_ai_results', side_effect=OperationalError('db down')), self.faces_patch():
            rec = jobs.submit(self.alice, self.png_file(), 'x.png')
        job = J.objects.get(upload=rec)
        self.assertEqual((job.status, job.error_code), ('FAILED', 'RETRIES_EXHAUSTED'))
        self.assertEqual(job.attempts, settings.JOB_MAX_RETRIES + 1)

    def test_backoff_grows_and_is_capped(self):
        vals = [jobs.backoff(i) for i in range(10)]
        self.assertEqual(vals[:3], [5, 10, 20])
        self.assertEqual(max(vals), 300)

    def test_time_limit_marks_job_timed_out(self):
        with mock.patch('detector.jobs.run_job', side_effect=SoftTimeLimitExceeded()):
            rec = jobs.submit(self.alice, self.png_file(), 'x.png')
        job = J.objects.get(upload=rec)
        self.assertEqual((job.status, job.error_code), ('FAILED', 'TIMEOUT'))
        self.assertEqual(UploadedImage.objects.get(pk=rec.pk).processing_status, 'failed')

    def test_one_model_failing_is_a_partial_not_a_failure(self):
        from ai_models.registry import get_registry
        get_registry().detectors['xception']._forward = lambda b: (_ for _ in ()).throw(RuntimeError('model boom'))
        with self.faces_patch():
            rec = jobs.submit(self.alice, self.png_file(), 'x.png')
        job = J.objects.get(upload=rec)
        self.assertEqual((job.status, job.partial), ('COMPLETED', True))
        self.assertTrue(any('XceptionNet' in w for w in job.warnings))
        self.assertEqual(ModelResult.objects.filter(analysis__uploaded_image=rec, model_key='xception', status='error').count(), 2)

    def test_forensic_module_failure_is_a_partial_not_a_failure(self):
        with mock.patch('forensics.copy_move.run_copy_move_analysis', side_effect=RuntimeError('cv2 boom')), self.faces_patch():
            rec = jobs.submit(self.alice, self.png_file(), 'x.png')
        job = J.objects.get(upload=rec)
        self.assertEqual(job.status, 'COMPLETED')
        self.assertTrue(any('copy-move' in w for w in job.warnings), job.warnings)
        self.assertTrue(any('Forensic module' in w for w in job.warnings))
        self.assertTrue(Analysis.objects.get(uploaded_image=rec).report)

    def test_broker_down_fails_fast_with_503(self):
        with mock.patch(QUEUE_NOOP, side_effect=ConnectionError('redis down')):
            r = self.call('post', '/analyses/', self.alice,
                          data={'image': SimpleUploadedFile('p.png', make_image_bytes('PNG'), content_type='image/png')})
        self.assertEqual((r.status_code, r.json()['error']['code']), (503, 'QUEUE_UNAVAILABLE'))
        job = J.objects.get()
        self.assertEqual((job.status, job.error_code), ('FAILED', 'QUEUE_UNAVAILABLE'))
        self.assertNotIn('redis', r.content.decode().lower())


class IdempotencyAndRecoveryTests(JobTestCase):
    def test_duplicate_delivery_of_a_finished_job_changes_nothing(self):
        with self.faces_patch():
            rec = jobs.submit(self.alice, self.png_file(), 'x.png')
        before = (Analysis.objects.get(uploaded_image=rec).pk, ModelResult.objects.count())
        with self.faces_patch():
            jobs.run_job(rec.job.pk)
            jobs.run_job(rec.job.pk)
        self.assertEqual((Analysis.objects.get(uploaded_image=rec).pk, ModelResult.objects.count()), before)

    def test_restart_replaces_partial_results_of_a_previous_attempt(self):
        rec = self.submit_queued()
        stale = Analysis.objects.create(uploaded_image=rec, final_prediction='UNCERTAIN')
        self.run_and_refresh(rec)
        self.assertEqual(Analysis.objects.filter(uploaded_image=rec).count(), 1)
        self.assertFalse(Analysis.objects.filter(pk=stale.pk).exists())

    def _stuck(self, rec, attempts, status='AI_ANALYSIS', minutes=60):
        J.objects.filter(pk=rec.job.pk).update(status=status, attempts=attempts,
                                               updated_at=timezone.now() - timedelta(minutes=minutes))

    def test_reaper_requeues_then_dead_letters(self):
        a, b, fresh = self.submit_queued(), self.submit_queued(), self.submit_queued()
        self._stuck(a, attempts=1)
        self._stuck(b, attempts=settings.JOB_MAX_RETRIES + 1)
        J.objects.filter(pk=fresh.job.pk).update(status='AI_ANALYSIS')       # recently updated: must be left alone
        with self.faces_patch():
            result = jobs.reap_stuck_jobs()
        self.assertEqual(result, {'requeued': 1, 'failed': 1})
        self.assertEqual(J.objects.get(pk=a.job.pk).status, 'COMPLETED')     # eager worker re-ran it
        jb = J.objects.get(pk=b.job.pk)
        self.assertEqual((jb.status, jb.error_code), ('FAILED', 'WORKER_LOST'))
        self.assertEqual(J.objects.get(pk=fresh.job.pk).status, 'AI_ANALYSIS')

    def test_requeue_failed_reruns_dead_lettered_jobs(self):
        rec = self.submit_queued()
        jobs.fail_job(rec.job.pk, 'WORKER_LOST')
        other = self.submit_queued(self.bob); jobs.fail_job(other.job.pk, 'TIMEOUT')
        with self.faces_patch():
            n = jobs.requeue_failed(code='WORKER_LOST')
        self.assertEqual(n, 1)
        self.assertEqual(J.objects.get(pk=rec.job.pk).status, 'COMPLETED')
        self.assertEqual(J.objects.get(pk=other.job.pk).status, 'FAILED')    # other code untouched


    def test_finished_jobs_cannot_be_overwritten_by_late_updates(self):
        with self.faces_patch():
            rec = jobs.submit(self.alice, self.png_file(), 'x.png')
        jobs.fail_job(rec.job.pk, 'TIMEOUT')                                  # late failure notification
        jobs.set_stage(rec.job.pk, 'AI_ANALYSIS')
        self.assertEqual(J.objects.get(pk=rec.job.pk).status, 'COMPLETED')


class WebJobTests(JobTestCase):
    def test_upload_redirects_to_processing_page_with_real_stages(self):
        self.client.login(username='alice', password=PW)
        with mock.patch(QUEUE_NOOP):
            r = self.client.post('/auth/upload/', {'image': SimpleUploadedFile('p.png', make_image_bytes('PNG'), content_type='image/png'), 'consent': 'on'})
        rec = UploadedImage.objects.get(user=self.alice)
        self.assertRedirects(r, f'/auth/processing/{rec.pk}/')
        page = self.client.get(f'/auth/processing/{rec.pk}/').content.decode()
        for stage in J.STAGES:
            self.assertIn(f'data-stage="{stage}"', page)
        main = page[page.index('<main'):page.index('</main>')]
        self.assertNotIn('%', main)                  # no progress percentages in the page content
        st = self.client.get(f'/auth/processing/{rec.pk}/status/').json()
        self.assertEqual((st['status'], st['finished'], st['stage_total']), ('QUEUED', False, 8))
        self.assertRedirects(self.client.get(f'/auth/result/{rec.pk}/'), f'/auth/processing/{rec.pk}/')   # result waits for the job

    def test_processing_pages_are_owner_only_and_finish_to_result(self):
        rec = self.submit_queued()
        self.client.login(username='bob', password=PW)
        self.assertEqual(self.client.get(f'/auth/processing/{rec.pk}/').status_code, 404)
        self.assertEqual(self.client.get(f'/auth/processing/{rec.pk}/status/').status_code, 404)
        self.assertEqual(self.client.post(f'/auth/processing/{rec.pk}/cancel/').status_code, 404)
        self.client.login(username='alice', password=PW)
        with self.faces_patch(0):
            jobs.run_job(rec.job.pk)
        self.assertRedirects(self.client.get(f'/auth/processing/{rec.pk}/'), f'/auth/result/{rec.pk}/')
        self.assertEqual(self.client.get(f'/auth/result/{rec.pk}/').status_code, 200)

    def test_cancel_from_web_requires_post_and_csrf(self):
        rec = self.submit_queued()
        self.client.login(username='alice', password=PW)
        self.assertEqual(self.client.get(f'/auth/processing/{rec.pk}/cancel/').status_code, 405)
        from django.test import Client
        strict = Client(enforce_csrf_checks=True); strict.login(username='alice', password=PW)
        self.assertEqual(strict.post(f'/auth/processing/{rec.pk}/cancel/').status_code, 403)
        self.assertEqual(J.objects.get(pk=rec.job.pk).status, 'QUEUED')
        self.client.post(f'/auth/processing/{rec.pk}/cancel/')
        self.assertEqual(J.objects.get(pk=rec.job.pk).status, 'CANCELLED')

    def test_legacy_uploads_without_a_job_still_render(self):
        rec = self.quick_upload(self.alice)[0]
        self.client.login(username='alice', password=PW)
        self.assertEqual(self.client.get(f'/auth/result/{rec.pk}/').status_code, 200)


class WorkerConfigurationTests(JobTestCase):
    def test_celery_is_configured_for_safe_redelivery(self):
        from config.celery import app
        c = app.conf
        self.assertTrue(c.task_acks_late and c.task_reject_on_worker_lost)
        self.assertEqual(c.worker_prefetch_multiplier, 1)
        self.assertEqual(c.task_default_queue, 'analysis')
        self.assertLess(c.task_soft_time_limit, c.task_time_limit)
        from detector.tasks import analyze_upload
        self.assertEqual(analyze_upload.name, 'detector.analyze_upload')
        self.assertEqual(analyze_upload.max_retries, settings.JOB_MAX_RETRIES)

    def test_model_preload_never_crashes_a_worker(self):
        from detector.tasks import preload_models
        preload_models()
        with mock.patch('ai_models.registry.get_registry', side_effect=RuntimeError('gpu exploded')):
            preload_models()                                                 # logged, not raised

    def test_readiness_reports_broker_state(self):
        with override_settings(CELERY_TASK_ALWAYS_EAGER=False):
            class Conn:
                def __enter__(self): return self
                def __exit__(self, *a): return False
                def ensure_connection(self, **k): raise ConnectionError('refused')
            with mock.patch('config.celery.app.connection_for_read', return_value=Conn()):
                r = self.client.get(f'{BASE}/system/readiness/')
            self.assertEqual(r.status_code, 503)
            self.assertFalse(r.json()['error']['details']['queue']['ok'])
            self.assertNotIn('refused', r.content.decode())
            class Ok(Conn):
                def ensure_connection(self, **k): return None
            with mock.patch('config.celery.app.connection_for_read', return_value=Ok()):
                r = self.client.get(f'{BASE}/system/readiness/')
            self.assertEqual(r.status_code, 200)
            self.assertIn('broker reachable', r.json()['data']['checks']['queue']['info'])
