"""Phase 8 tests: structured logging + redaction, metrics endpoint, retention, account deletion."""

import io
import json
import logging
import os
from datetime import timedelta
from unittest import mock

from django.core.management import call_command
from django.test import override_settings
from django.utils import timezone

from config.logging_conf import ContextFilter, JsonFormatter, build_logging
from detector import jobs, retention
from detector.logctx import job_context
from detector.middleware import request_id_var
from detector.models import Analysis, AnalysisJob, AuditLog, UploadedImage
from detector.tests_api import PW, ApiTestCase


def fmt(msg, *args, level=logging.INFO, exc_info=None, **extra):
    rec = logging.LogRecord('t', level, __file__, 1, msg, args, exc_info)
    for k, v in extra.items():
        setattr(rec, k, v)
    ContextFilter().filter(rec)
    return json.loads(JsonFormatter().format(rec))


class LoggingTests(ApiTestCase):
    def test_json_has_correlation_ids(self):
        tok = request_id_var.set('req-12345678')
        try:
            with job_context(42):
                d = fmt('hello %s', 'world')
        finally:
            request_id_var.reset(tok)
        self.assertEqual((d['msg'], d['request_id'], d['job_id'], d['level']), ('hello world', 'req-12345678', '42', 'INFO'))
        self.assertIsNone(fmt('x')['job_id'])

    def test_redaction_of_secrets_emails_tokens_paths(self):
        jwt = 'eyJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoxfQ.abcdefghijklmnop'
        d = fmt('Authorization: Bearer %s mail bob@example.com file /srv/app/media/uploads/a.png '
                'url https://s3/x?X-Amz-Signature=deadbeef password=hunter2', jwt)
        out = d['msg']
        for leak in (jwt, 'bob@example.com', '/srv/app/media', 'deadbeef', 'hunter2'):
            self.assertNotIn(leak, out)
        self.assertIn('[PATH]', out)

    def test_extra_fields_with_secret_keys_are_redacted(self):
        d = fmt('login', password='p', refresh_token='r', info={'api_key': 'k', 'ok': 'fine'}, user_id=7)
        self.assertEqual((d['password'], d['refresh_token'], d['info']['api_key']), ('[REDACTED]',) * 3)
        self.assertEqual((d['info']['ok'], d['user_id']), ('fine', 7))

    def test_exception_logs_type_and_redacted_message_without_locals(self):
        secret_local = 'LOCALSECRET-123'  # noqa: F841
        try:
            raise RuntimeError('db at /srv/data/pw.txt failed')
        except RuntimeError:
            import sys
            d = fmt('boom', level=logging.ERROR, exc_info=sys.exc_info())
        self.assertEqual(d['exc_type'], 'RuntimeError')
        self.assertNotIn('/srv/data', d['exc'])
        self.assertNotIn('LOCALSECRET', json.dumps(d))

    def test_logging_config_builds_for_both_formats(self):
        import logging.config
        from django.conf import settings
        self.addCleanup(logging.config.dictConfig, settings.LOGGING)
        for f in ('json', 'text'):
            logging.config.dictConfig(build_logging('WARNING', f))

    def test_job_run_logs_carry_job_id(self):
        rec = self.two_face_analysis(self.alice)
        seen = []
        h = logging.Handler(); h.emit = lambda r: seen.append(r)
        h.addFilter(ContextFilter())
        lg = logging.getLogger('deepshield.test'); lg.addHandler(h); lg.setLevel(logging.INFO)
        self.addCleanup(lg.removeHandler, h)
        with job_context(rec.job.pk):
            lg.info('inside')
        self.assertEqual(seen[0].job_id, str(rec.job.pk))


class MetricsTests(ApiTestCase):
    @override_settings(METRICS_TOKEN='')
    def test_disabled_without_token_for_anonymous(self):
        self.assertEqual(self.client.get('/metrics').status_code, 404)

    @override_settings(METRICS_TOKEN='s3cret-token')
    def test_requires_token(self):
        self.assertEqual(self.client.get('/metrics').status_code, 404)
        self.assertEqual(self.client.get('/metrics', HTTP_AUTHORIZATION='Bearer wrong').status_code, 404)
        r = self.client.get('/metrics', HTTP_AUTHORIZATION='Bearer s3cret-token')
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r['Cache-Control'], 'no-store')

    @override_settings(METRICS_TOKEN='tok')
    def test_exposes_job_and_queue_metrics_without_identifiers(self):
        self.two_face_analysis(self.alice)
        with mock.patch('detector.tasks.analyze_upload.apply_async'):
            jobs.submit(self.alice, __import__('django').core.files.uploadedfile.SimpleUploadedFile(
                'q.png', open(os.path.join(self.dir, 'in.png'), 'rb').read()), 'q.png')
        body = self.client.get('/metrics', HTTP_AUTHORIZATION='Bearer tok').content.decode()
        self.assertIn('deepshield_jobs_finished_total{status="COMPLETED"}', body)
        self.assertIn('deepshield_queue_depth 1.0', body)
        self.assertIn('deepshield_job_duration_seconds_bucket', body)
        self.assertNotIn('alice', body)
        self.assertNotIn('.png', body)

    @override_settings(METRICS_TOKEN='')
    def test_staff_session_can_read(self):
        from django.contrib.auth.models import User
        staff = User.objects.create_user('ops', 'o@example.com', PW, is_staff=True)
        self.client.force_login(staff)
        self.assertEqual(self.client.get('/metrics').status_code, 200)
        self.client.force_login(self.alice)
        self.assertEqual(self.client.get('/metrics').status_code, 404)


class RetentionTests(ApiTestCase):
    def age(self, rec, days):
        UploadedImage.objects.filter(pk=rec.pk).update(uploaded_at=timezone.now() - timedelta(days=days))

    def files_of(self, rec):
        from detector import storage
        names = storage.artifact_names(rec) | {rec.image.name}
        return [n for n in names if os.path.exists(os.path.join(str(self.media_root()), n))]

    def media_root(self):
        from django.conf import settings
        return settings.MEDIA_ROOT

    @override_settings(DELETE_ANALYSIS_AFTER_DAYS=0)
    def test_disabled_policy_deletes_nothing(self):
        rec = self.two_face_analysis(self.alice); self.age(rec, 9999)
        self.assertEqual(retention.purge_expired()['analyses'], 0)
        self.assertTrue(UploadedImage.objects.filter(pk=rec.pk).exists())

    @override_settings(DELETE_ANALYSIS_AFTER_DAYS=30)
    def test_expired_analysis_and_all_files_are_deleted_fresh_one_kept(self):
        old = self.two_face_analysis(self.alice); self.age(old, 31)
        old_files = self.files_of(old)
        self.assertGreater(len(old_files), 3)
        fresh = self.two_face_analysis(self.bob); self.age(fresh, 29)
        fresh_files = self.files_of(fresh)
        r = retention.purge_expired()
        self.assertEqual(r['analyses'], 1)
        self.assertFalse(UploadedImage.objects.filter(pk=old.pk).exists())
        self.assertEqual(Analysis.objects.filter(uploaded_image_id=old.pk).count(), 0)
        self.assertEqual(AnalysisJob.objects.filter(upload_id=old.pk).count(), 0)
        for n in old_files:
            self.assertFalse(os.path.exists(os.path.join(str(self.media_root()), n)), n)
        self.assertTrue(UploadedImage.objects.filter(pk=fresh.pk).exists())
        for n in fresh_files:
            self.assertTrue(os.path.exists(os.path.join(str(self.media_root()), n)), n)

    @override_settings(DELETE_ANALYSIS_AFTER_DAYS=30)
    def test_dry_run_changes_nothing(self):
        rec = self.two_face_analysis(self.alice); self.age(rec, 40)
        self.assertEqual(retention.purge_expired(dry_run=True)['analyses'], 1)
        self.assertTrue(UploadedImage.objects.filter(pk=rec.pk).exists())

    @override_settings(DELETE_ANALYSIS_AFTER_DAYS=30)
    def test_in_flight_jobs_are_never_purged(self):
        rec = self.quick_upload(self.alice)[0]
        AnalysisJob.objects.create(upload=rec, status=AnalysisJob.AI_ANALYSIS)
        self.age(rec, 90)
        self.assertEqual(retention.purge_expired()['analyses'], 0)
        AnalysisJob.objects.filter(upload=rec).update(status=AnalysisJob.COMPLETED)
        self.assertEqual(retention.purge_expired()['analyses'], 1)

    @override_settings(DELETE_ANALYSIS_AFTER_DAYS=30, AUDIT_LOG_RETENTION_DAYS=10)
    def test_old_audit_rows_pruned(self):
        a = AuditLog.objects.create(action='login', username='x')
        AuditLog.objects.filter(pk=a.pk).update(created_at=timezone.now() - timedelta(days=11))
        b = AuditLog.objects.create(action='login', username='y')
        self.assertEqual(retention.purge_expired()['audit_rows'], 1)
        self.assertEqual(list(AuditLog.objects.values_list('pk', flat=True)), [b.pk])



class AccountDeletionTests(ApiTestCase):
    def test_delete_account_removes_user_data_files_and_tokens(self):
        rec = self.two_face_analysis(self.alice)
        from detector import storage
        files = storage.artifact_names(rec) | {rec.image.name}
        from django.conf import settings
        r = self.call('delete', '/auth/account/', self.alice, data=json.dumps({'password': PW}),
                      content_type='application/json')
        self.assertEqual(r.status_code, 204)
        from django.contrib.auth.models import User
        self.assertFalse(User.objects.filter(username='alice').exists())
        self.assertEqual(UploadedImage.objects.count(), 0)
        for n in files:
            self.assertFalse(os.path.exists(os.path.join(str(settings.MEDIA_ROOT), n)), n)
        self.assertTrue(AuditLog.objects.filter(action='account_deleted', username='alice').exists())
        # the old access token no longer works
        r2 = self.client.get('/api/v1/analyses/', HTTP_AUTHORIZATION=f"Bearer {self.tokens[self.alice.pk]['access']}")
        self.assertEqual(r2.status_code, 401)

    def test_wrong_password_deletes_nothing_and_is_rate_limited(self):
        self.quick_upload(self.alice)
        for _ in range(5):
            r = self.call('delete', '/auth/account/', self.alice, data=json.dumps({'password': 'nope'}),
                          content_type='application/json')
            self.assertEqual(r.status_code, 400)
        r = self.call('delete', '/auth/account/', self.alice, data=json.dumps({'password': PW}),
                      content_type='application/json')
        self.assertEqual(r.status_code, 429)
        self.assertEqual(UploadedImage.objects.filter(user=self.alice).count(), 1)

    def test_requires_authentication_and_leaves_other_users_alone(self):
        self.quick_upload(self.bob)
        r = self.client.delete('/api/v1/auth/account/', data=json.dumps({'password': PW}), content_type='application/json')
        self.assertEqual(r.status_code, 401)
        self.call('delete', '/auth/account/', self.alice, data=json.dumps({'password': PW}), content_type='application/json')
        self.assertEqual(UploadedImage.objects.filter(user=self.bob).count(), 1)


class ModelMonitoringTests(ApiTestCase):
    _offset = 0

    def seed(self, n, flagged_ratio, days_ago, mean=0.5, key='resnet50', status='success'):
        self._offset += 1000      # distinct face_index per call (unique per analysis+model)
        from detector.models import Analysis, ModelResult, ModelVersion
        mv, _ = ModelVersion.objects.get_or_create(key=key, version='1', weights_sha256='a' * 64, defaults={'name': key})
        rec = self.quick_upload(self.alice)[0]
        an = Analysis.objects.create(uploaded_image=rec, final_prediction='UNCERTAIN')
        ids = []
        for i in range(n):
            fl = i < n * flagged_ratio
            r = ModelResult.objects.create(analysis=an, face_index=len(ids) + self._offset, model_name=key, model_key=key, model_version=mv, status=status,
                                           prediction='FAKE' if fl else 'REAL', fake_probability=min(1, mean + (0.3 if fl else -0.3)))
            ids.append(r.pk)
        ModelResult.objects.filter(pk__in=ids).update(created_at=timezone.now() - timedelta(days=days_ago))

    def test_distribution_numbers_are_exact(self):
        from detector import model_monitoring as mm
        self.seed(40, 0.25, 1)
        r = mm.summarise(7)[0]
        self.assertEqual((r['model'], r['current']['n'], r['current']['flagged_rate'], r['current']['failure_rate']),
                         ('resnet50', 40, 0.25, 0.0))
        self.assertEqual(r['previous']['n'], 0)
        self.assertEqual(r['alerts'], [])
        self.assertFalse(r['enough_data'])

    def test_shift_in_flagged_rate_raises_alert_only_with_enough_data(self):
        from detector import model_monitoring as mm
        self.seed(40, 0.10, 10)           # previous window
        self.seed(40, 0.60, 2)            # current window
        r = mm.summarise(7)[0]
        self.assertTrue(r['enough_data'])
        self.assertTrue(any('flagged rate moved 0.10 -> 0.60' in a for a in r['alerts']), r['alerts'])

    def test_no_alert_when_stable_and_none_when_sample_small(self):
        from detector import model_monitoring as mm
        self.seed(40, 0.30, 10); self.seed(40, 0.32, 2)
        self.assertEqual(mm.summarise(7)[0]['alerts'], [])
        ModelResultCleanup = __import__('detector.models', fromlist=['ModelResult']).ModelResult
        ModelResultCleanup.objects.all().delete()
        self.seed(5, 0.0, 10); self.seed(5, 1.0, 2)
        r = mm.summarise(7)[0]
        self.assertEqual(r['alerts'], [])       # 5 samples: too small to claim drift
        self.assertFalse(r['enough_data'])

    def test_failure_rate_alert_and_command(self):
        from detector import model_monitoring as mm
        self.seed(40, 0.0, 1, status='error')
        self.assertTrue(any('failure rate' in a for a in mm.summarise(7)[0]['alerts']))
        out = io.StringIO()
        with self.assertRaises(SystemExit):
            call_command('model_report', '--fail-on-alert', stdout=out)
        self.assertIn('ALERT', out.getvalue())
        out = io.StringIO(); call_command('model_report', '--json', stdout=out)
        self.assertEqual(json.loads(out.getvalue())[0]['model'], 'resnet50')
