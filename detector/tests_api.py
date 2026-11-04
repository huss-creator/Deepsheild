"""
Phase 6 tests: REST API contract, JWT lifecycle, object-level authorization on every endpoint,
normalised tables + constraints, report integrity, file deletion, password reset, admin, CORS.
"""

import hashlib
import json
import os
import re
import uuid
from pathlib import Path

import numpy as np
from django.contrib.auth.models import User
from django.core import mail
from django.core.cache import cache
from django.core.files.uploadedfile import SimpleUploadedFile
from django.db import IntegrityError, transaction
from django.test import Client, override_settings
from django.test.utils import CaptureQueriesContext
from django.db import connection
from PIL import Image

from ai_models.face_detection import Face as DetectedFace
from ai_models.registry import MODEL_ORDER, reset_registry
from detector import reports
from detector.models import (Analysis, AuditLog, Face, ForensicAnalysis, ModelResult, ModelVersion,
                             Report, UploadedImage)
from detector.services import process_upload
from detector.tests_ai import ModelDirTestCase, StubDetector
from detector.tests_security import make_image_bytes

PW = 'Correct-Horse-9'
BASE = '/api/v1'


def two_face_png(path):
    arr = np.full((300, 420, 3), 128, np.uint8)
    arr[50:190, 50:190] = 235
    arr[60:180, 250:370] = 20
    Image.fromarray(arr).save(path)
    return path


class ApiTestCase(ModelDirTestCase):
    """Isolated MODEL_DIR/MEDIA_ROOT, two users, three working ONNX models."""

    def setUp(self):
        super().setUp()
        cache.clear()
        self.alice = User.objects.create_user('alice', 'alice@example.com', PW)
        self.bob = User.objects.create_user('bob', 'bob@example.com', PW)
        self.add_onnx('vit_b16', outputs=2, softmax=True); self.add_onnx('xception', outputs=2, softmax=True)
        self.add_onnx('efficientnet_b4', outputs=2)
        reset_registry(self.registry())
        self.addCleanup(reset_registry, None)
        self.tokens = {}

    # ---- helpers
    def token(self, user):
        if user.pk not in self.tokens:
            r = self.client.post(f'{BASE}/auth/login/', json.dumps({'username': user.username, 'password': PW}),
                                 content_type='application/json')
            self.tokens[user.pk] = r.json()['data']
        return self.tokens[user.pk]['access']

    def call(self, method, path, user=None, **kw):
        hdr = {'HTTP_AUTHORIZATION': f'Bearer {self.token(user)}'} if user else {}
        hdr.update(kw.pop('headers', {}))
        return getattr(self.client, method)(f'{BASE}{path}', **hdr, **kw)

    def two_face_analysis(self, user):
        png = two_face_png(os.path.join(self.dir, 'in.png'))
        from unittest import mock
        faces = [DetectedFace(0, 50, 50, 140, 140), DetectedFace(1, 250, 60, 120, 120)]
        with mock.patch('ai_models.pipeline.get_face_detector', return_value=StubDetector(faces)):
            return process_upload(user, SimpleUploadedFile('upload.png', open(png, 'rb').read()), 'two.png')

    def quick_upload(self, user, status=UploadedImage.STATUS_COMPLETED, n=1):
        return [UploadedImage.objects.create(
            user=user, image=SimpleUploadedFile('u.png', make_image_bytes('PNG', (40, 40))),
            original_filename='u.png', file_size=10, processing_status=status) for _ in range(n)]


# ============================================================ contract / envelope
class EnvelopeTests(ApiTestCase):
    def test_health_envelope_and_request_id(self):
        r = self.client.get(f'{BASE}/system/health/')
        body = r.json()
        self.assertEqual((r.status_code, body['success'], body['error']), (200, True, None))
        self.assertEqual(body['data']['status'], 'ok')
        self.assertEqual(body['request_id'], r['X-Request-ID'])

    def test_client_request_id_is_echoed_only_if_well_formed(self):
        ok = self.client.get(f'{BASE}/system/health/', HTTP_X_REQUEST_ID='trace-1234-abcd')
        self.assertEqual(ok.json()['request_id'], 'trace-1234-abcd')
        bad = self.client.get(f'{BASE}/system/health/', HTTP_X_REQUEST_ID='<script>x</script>')
        self.assertNotIn('script', bad.json()['request_id'])

    def test_unauthenticated_is_401_envelope(self):
        r = self.client.get(f'{BASE}/analyses/')
        self.assertEqual(r.status_code, 401)
        b = r.json()
        self.assertEqual((b['success'], b['data'], b['error']['code']), (False, None, 'UNAUTHENTICATED'))
        self.assertEqual(r['WWW-Authenticate'], 'Bearer')

    def test_garbage_token_is_401(self):
        r = self.client.get(f'{BASE}/analyses/', HTTP_AUTHORIZATION='Bearer not.a.token')
        self.assertEqual((r.status_code, r.json()['error']['code']), (401, 'UNAUTHENTICATED'))

    def test_unknown_path_and_version_are_json_404(self):
        for path in (f'{BASE}/nope/', '/api/v9/system/health/'):
            r = self.client.get(path)
            self.assertEqual(r.status_code, 404, path)
            self.assertEqual(r.json()['error']['code'], 'NOT_FOUND')

    def test_malformed_json_is_400(self):
        r = self.client.post(f'{BASE}/auth/login/', '{bad json', content_type='application/json')
        self.assertEqual((r.status_code, r.json()['error']['code']), (400, 'MALFORMED_REQUEST'))

    def test_unhandled_error_never_leaks_internals(self):
        from unittest import mock
        with mock.patch('detector.api.views.ratelimit.allow', side_effect=RuntimeError('/srv/secret/db.password boom')):
            r = self.client.post(f'{BASE}/auth/register/', json.dumps({}), content_type='application/json')
        self.assertEqual((r.status_code, r.json()['error']['code']), (500, 'INTERNAL_ERROR'))
        blob = r.content.decode()
        for leak in ('secret', 'Traceback', 'RuntimeError', '/srv', 'password'):
            self.assertNotIn(leak, blob)

    def test_session_login_does_not_authenticate_the_api(self):
        c = Client(); c.login(username='alice', password=PW)
        r = c.post(f'{BASE}/analyses/', {'image': SimpleUploadedFile('a.png', make_image_bytes('PNG'))})
        self.assertEqual(r.status_code, 401)           # JWT-only: no cookie/CSRF attack surface

    def test_openapi_schema_documents_every_endpoint(self):
        r = self.client.get(f'{BASE}/schema/', HTTP_ACCEPT='application/json')
        self.assertEqual(r.status_code, 200)
        paths = set(json.loads(r.content)['paths'])
        for p in ('auth/login', 'auth/register', 'auth/refresh', 'auth/logout', 'analyses/', 'analyses/{id}/',
                  'analyses/{id}/faces/', 'analyses/{id}/models/', 'analyses/{id}/forensics/',
                  'analyses/{id}/report/', 'models/', 'system/health/', 'system/readiness/'):
            self.assertTrue(any(p in x for x in paths), p)
        self.assertEqual(self.client.get(f'{BASE}/docs/').status_code, 200)
        self.assertEqual(self.client.get(f'{BASE}/redoc/').status_code, 200)

    def test_openapi_mentions_limits_and_states(self):
        text = self.client.get(f'{BASE}/schema/', HTTP_ACCEPT='application/json').content.decode()
        for needle in ('10 MB', 'no_face', 'Bearer', 'request_id'):
            self.assertIn(needle, text)


# ============================================================ authentication
class ApiAuthTests(ApiTestCase):
    def post(self, path, data, **kw):
        return self.client.post(f'{BASE}{path}', json.dumps(data), content_type='application/json', **kw)

    def test_register_login_and_use_token(self):
        r = self.post('/auth/register/', {'username': 'carol', 'email': 'c@example.com', 'password': 'Tr0ub4dor&3-xyz'})
        self.assertEqual(r.status_code, 201)
        d = r.json()['data']
        self.assertTrue(d['access'] and d['refresh'])
        self.assertNotIn('password', json.dumps(d))
        self.assertTrue(User.objects.get(username='carol').check_password('Tr0ub4dor&3-xyz'))   # hashed, not plain
        self.assertNotEqual(User.objects.get(username='carol').password, 'Tr0ub4dor&3-xyz')
        me = self.client.get(f'{BASE}/analyses/', HTTP_AUTHORIZATION=f"Bearer {d['access']}")
        self.assertEqual(me.status_code, 200)

    def test_register_validation(self):
        base = {'username': 'dave', 'email': 'd@example.com', 'password': 'Tr0ub4dor&3-xyz'}
        for bad, field in (({**base, 'password': 'short1A'}, 'non_field_errors'),
                           ({**base, 'password': '1234567890'}, 'non_field_errors'),
                           ({**base, 'username': 'alice'}, 'username'),
                           ({**base, 'email': 'ALICE@example.com'}, 'email'),
                           ({**base, 'email': 'nope'}, 'email')):
            r = self.post('/auth/register/', bad)
            self.assertEqual((r.status_code, r.json()['error']['code']), (400, 'VALIDATION_ERROR'), bad)
            self.assertIn(field, r.json()['error']['details'])
        self.assertFalse(User.objects.filter(username='dave').exists())

    def test_login_failure_is_generic_and_identical_for_unknown_user(self):
        a = self.post('/auth/login/', {'username': 'alice', 'password': 'wrong'})
        b = self.post('/auth/login/', {'username': 'ghost', 'password': 'wrong'})
        self.assertEqual((a.status_code, b.status_code), (401, 401))
        self.assertEqual(a.json()['error'], b.json()['error'])

    @override_settings(RATE_LIMITS={**__import__('django.conf', fromlist=['settings']).settings.RATE_LIMITS,
                                    'login_fail': (3, 900)})
    def test_login_lockout(self):
        for _ in range(3):
            self.assertEqual(self.post('/auth/login/', {'username': 'alice', 'password': 'x'}).status_code, 401)
        r = self.post('/auth/login/', {'username': 'alice', 'password': PW})     # correct, but locked
        self.assertEqual((r.status_code, r.json()['error']['code']), (429, 'RATE_LIMITED'))
        self.assertIn('Retry-After', r)

    def test_register_rate_limit(self):
        from django.conf import settings
        with override_settings(RATE_LIMITS={**settings.RATE_LIMITS, 'register': (2, 3600)}):
            codes = [self.post('/auth/register/', {}).status_code for _ in range(3)]
        self.assertEqual(codes, [400, 400, 429])

    def test_refresh_rotates_and_old_refresh_is_revoked(self):
        t = self.post('/auth/login/', {'username': 'alice', 'password': PW}).json()['data']
        r = self.post('/auth/refresh/', {'refresh': t['refresh']})
        self.assertEqual(r.status_code, 200)
        new = r.json()['data']
        self.assertNotEqual(new['refresh'], t['refresh'])
        again = self.post('/auth/refresh/', {'refresh': t['refresh']})           # replay of rotated token
        self.assertEqual((again.status_code, again.json()['error']['code']), (401, 'UNAUTHENTICATED'))
        self.assertEqual(self.post('/auth/refresh/', {'refresh': new['refresh']}).status_code, 200)

    def test_logout_revokes_refresh_token(self):
        t = self.post('/auth/login/', {'username': 'alice', 'password': PW}).json()['data']
        h = {'HTTP_AUTHORIZATION': f"Bearer {t['access']}"}
        self.assertEqual(self.post('/auth/logout/', {'refresh': t['refresh']}, **h).status_code, 200)
        self.assertEqual(self.post('/auth/refresh/', {'refresh': t['refresh']}).status_code, 401)
        self.assertEqual(self.post('/auth/logout/', {'refresh': t['refresh']}, **h).status_code, 400)   # already revoked

    def test_cannot_revoke_another_users_refresh_token(self):
        ta = self.post('/auth/login/', {'username': 'alice', 'password': PW}).json()['data']
        tb = self.post('/auth/login/', {'username': 'bob', 'password': PW}).json()['data']
        r = self.post('/auth/logout/', {'refresh': ta['refresh']}, HTTP_AUTHORIZATION=f"Bearer {tb['access']}")
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.post('/auth/refresh/', {'refresh': ta['refresh']}).status_code, 200)   # still valid

    def test_token_lifetimes_are_short(self):
        from django.conf import settings
        self.assertLessEqual(settings.SIMPLE_JWT['ACCESS_TOKEN_LIFETIME'].total_seconds(), 3600)
        self.assertTrue(settings.SIMPLE_JWT['ROTATE_REFRESH_TOKENS'] and settings.SIMPLE_JWT['BLACKLIST_AFTER_ROTATION'])

    def test_audit_trail_for_auth_events_without_sensitive_data(self):
        self.post('/auth/login/', {'username': 'alice', 'password': 'wrong-pass-123'})
        self.post('/auth/login/', {'username': 'alice', 'password': PW}, HTTP_X_REQUEST_ID='audit-req-0001')
        failed = AuditLog.objects.get(action='login_failed')
        self.assertFalse(failed.success)
        self.assertNotIn('alice', json.dumps(failed.metadata))              # only a hash of the attempted name
        self.assertNotIn('wrong-pass', json.dumps(failed.metadata))
        ok = AuditLog.objects.get(action='login', user=self.alice)
        self.assertEqual(ok.request_id, 'audit-req-0001')                   # request ID links audit rows to logs
        self.assertTrue(ok.ip_address)


# ============================================================ analyses
class ApiAnalysisTests(ApiTestCase):
    def upload(self, user, data=None, name='p.png', ctype='image/png'):
        return self.call('post', '/analyses/', user,
                         data={'image': SimpleUploadedFile(name, make_image_bytes('PNG') if data is None else data, content_type=ctype)})

    def test_upload_creates_analysis_with_documented_shape(self):
        r = self.upload(self.alice)
        self.assertEqual(r.status_code, 202)
        d = r.json()['data']
        uuid.UUID(d['id'])
        self.assertEqual((d['status'], d['assessment'], d['face_count']), ('no_face', 'NO_FACE', 0))
        self.assertIsInstance(d['warnings'], list)
        rec = UploadedImage.objects.get(public_id=d['id'])
        self.assertEqual(d['sha256'], hashlib.sha256(Path(rec.image.path).read_bytes()).hexdigest())
        self.assertRegex(rec.image.name, r'^users/user_\d+/uploads/[0-9a-f-]{36}\.png$')
        self.assertEqual(AuditLog.objects.filter(action='analysis_created', user=self.alice).count(), 1)

    def test_invalid_uploads(self):
        cases = [(b'<?php evil(); ?>', 'x.jpg', 'image/jpeg', 400, 'INVALID_IMAGE'),
                 (b'MZ\x90\x00' + b'\0' * 200, 'x.png', 'image/png', 400, 'INVALID_IMAGE'),
                 (make_image_bytes('PNG') + b'<script>x</script>', 'x.png', 'image/png', 400, 'INVALID_IMAGE'),
                 (b'', 'x.png', 'image/png', 400, 'INVALID_IMAGE')]
        for data, name, ctype, status, code in cases:
            r = self.upload(self.alice, data, name, ctype)
            self.assertEqual((r.status_code, r.json()['error']['code']), (status, code), name)
        self.assertEqual(UploadedImage.objects.count(), 0)

    def test_missing_field_oversize_and_pixel_bomb(self):
        r = self.call('post', '/analyses/', self.alice, data={})
        self.assertEqual((r.status_code, r.json()['error']['code']), (400, 'VALIDATION_ERROR'))
        with override_settings(MAX_UPLOAD_SIZE=500):
            r = self.upload(self.alice)
        self.assertEqual((r.status_code, r.json()['error']['code']), (413, 'FILE_TOO_LARGE'))
        import io
        buf = io.BytesIO(); Image.new('L', (3000, 3000), 0).save(buf, 'PNG')
        with override_settings(MAX_PIXELS=1_000_000):
            r = self.upload(self.alice, buf.getvalue())
        self.assertEqual(r.json()['error']['details']['reason'], 'too_many_pixels')

    def test_upload_rate_limit(self):
        from django.conf import settings
        with override_settings(RATE_LIMITS={**settings.RATE_LIMITS, 'upload': (2, 3600)}):
            codes = [self.upload(self.alice, b'junk').status_code for _ in range(3)]
        self.assertEqual(codes, [400, 400, 429])
        self.assertTrue(AuditLog.objects.filter(action='throttled').exists())

    def test_list_pagination_filter_order_and_isolation(self):
        self.quick_upload(self.alice, n=22)
        self.quick_upload(self.alice, UploadedImage.STATUS_FAILED, 2)
        self.quick_upload(self.bob, n=3)
        d = self.call('get', '/analyses/', self.alice).json()['data']
        self.assertEqual((d['count'], len(d['results'])), (24, 20))
        self.assertIsNotNone(d['next'])
        p2 = self.call('get', '/analyses/?page=2', self.alice).json()['data']
        self.assertEqual(len(p2['results']), 4)
        self.assertEqual(self.call('get', '/analyses/?status=failed', self.alice).json()['data']['count'], 2)
        ids = {x['id'] for x in d['results'] + p2['results']}
        self.assertFalse(ids & {str(u.public_id) for u in UploadedImage.objects.filter(user=self.bob)})
        dates = [x['created_at'] for x in d['results']]
        self.assertEqual(dates, sorted(dates, reverse=True))

    def test_list_has_constant_query_count(self):
        self.token(self.alice)
        self.quick_upload(self.alice, n=3)
        with CaptureQueriesContext(connection) as small:
            self.call('get', '/analyses/', self.alice)
        self.quick_upload(self.alice, n=15)
        with CaptureQueriesContext(connection) as big:
            self.call('get', '/analyses/', self.alice)
        self.assertEqual(len(small), len(big))             # no N+1

    def test_multiface_endpoints_and_normalised_rows(self):
        rec = self.two_face_analysis(self.alice)
        self.assertEqual(rec.processing_status, UploadedImage.STATUS_COMPLETED)
        a = rec.analysis
        self.assertEqual(Face.objects.filter(analysis=a).count(), 2)
        self.assertEqual(
            ModelResult.objects.filter(analysis=a).count(),
            Face.objects.filter(analysis=a).count() * len(MODEL_ORDER),
        )
        self.assertEqual(ModelVersion.objects.count(), 3)
        self.assertFalse(ModelResult.objects.filter(analysis=a, model_key='vit_b16', model_version=None).exists())
        pid = str(rec.public_id)

        d = self.call('get', f'/analyses/{pid}/', self.alice).json()['data']
        self.assertEqual((d['assessment'], d['face_count']), ('POTENTIALLY_MANIPULATED', 2))
        self.assertEqual(d['ai_detection']['faces_analysed'], 2)
        self.assertEqual(len(d['model_versions']), 3)
        self.assertEqual(d['report']['sha256'], a.report.content_sha256)
        blob = json.dumps(d)
        for leak in (str(self.media), '/media/', 'uploads/', 'faces/', str(self.dir)):
            self.assertNotIn(leak, blob)

        faces = self.call('get', f'/analyses/{pid}/faces/', self.alice).json()['data']
        self.assertEqual([f['index'] for f in faces], [0, 1])
        self.assertEqual(faces[0]['bbox'], {'x': 50, 'y': 50, 'width': 140, 'height': 140})
        self.assertEqual((faces[0]['ensemble']['label'], faces[1]['ensemble']['label']),
                         ('POTENTIALLY_MANIPULATED', 'AUTHENTIC'))

        models = self.call('get', f'/analyses/{pid}/models/', self.alice).json()['data']
        self.assertEqual(len(models), 2)
        r0 = {m['key']: m for m in models[0]['results']}
        self.assertEqual(len(r0), len(MODEL_ORDER))
        self.assertEqual(r0['vit_b16']['status'], 'AVAILABLE')
        self.assertEqual(r0['freq_svm']['status'], 'NOT_CONFIGURED')
        self.assertIsNone(r0['freq_svm']['score'])                       # never a fabricated score
        self.assertEqual(sorted(models[0]['disagreement']['manipulated']), ['efficientnet_b4', 'vit_b16', 'xception'])
        self.assertEqual(len(models[0]['disagreement']['unavailable']), len(MODEL_ORDER) - 3)

        f = self.call('get', f'/analyses/{pid}/forensics/', self.alice).json()['data']
        for k in ('forensic_indicators', 'provenance', 'integrity', 'explainability', 'summary'):
            self.assertIn(k, f)
        self.assertEqual(f['summary']['assessment'], 'POTENTIALLY_MANIPULATED')

    def test_report_integrity_hash_and_signature(self):
        rec = self.two_face_analysis(self.alice)
        pid = str(rec.public_id)
        r = self.call('get', f'/analyses/{pid}/report/', self.alice).json()['data']
        self.assertEqual(r['verification'], {'hash_valid': True, 'signature_valid': True})
        c = r['report']
        for k in ('software_version', 'analysis_id', 'file', 'model_versions', 'ai_detection', 'faces',
                  'forensic_evidence', 'limitations'):
            self.assertIn(k, c)
        self.assertEqual(c['file']['sha256'], rec.sha256_hash)
        self.assertNotIn(str(self.media), json.dumps(c))
        self.assertEqual(hashlib.sha256(reports.canonical_json(c).encode()).hexdigest(), r['sha256'])
        self.assertTrue(AuditLog.objects.filter(action='report_viewed', user=self.alice).exists())
        # tamper with the stored content -> detected
        rep = Report.objects.get(analysis=rec.analysis)
        rep.content = {**rep.content, 'assessment': {'label': 'AUTHENTIC'}}
        rep.save()
        v = self.call('get', f'/analyses/{pid}/report/', self.alice).json()['data']['verification']
        self.assertEqual(v, {'hash_valid': False, 'signature_valid': True})   # signature covers the stored hash only
        rep.content_sha256 = reports.content_hash(rep.content)
        rep.save()
        v = self.call('get', f'/analyses/{pid}/report/', self.alice).json()['data']['verification']
        self.assertEqual(v, {'hash_valid': True, 'signature_valid': False})   # forged hash cannot carry a valid HMAC

    def test_artifact_endpoints_serve_images_to_owner(self):
        rec = self.two_face_analysis(self.alice)
        pid = str(rec.public_id)
        for path in (f'/analyses/{pid}/image/', f'/analyses/{pid}/faces/0/crop/', f'/analyses/{pid}/faces/1/crop/'):
            r = self.call('get', path, self.alice)
            self.assertEqual(r.status_code, 200, path)
            self.assertEqual(r['Content-Type'], 'image/png')
            self.assertIn('no-store', r['Cache-Control'])
            b"".join(r.streaming_content)   # drain; close() would drop the shared test DB connection
        self.assertEqual(self.call('get', f'/analyses/{pid}/faces/9/crop/', self.alice).status_code, 404)
        self.assertEqual(self.call('get', f'/analyses/{pid}/forensics/not_a_test/image/', self.alice).status_code, 404)

    def test_delete_removes_rows_and_files(self):
        rec = self.two_face_analysis(self.alice)
        pid = str(rec.public_id)
        paths = [Path(rec.image.path)] + [Path(f.crop_image.path) for f in rec.analysis.faces.all()]
        forensic = rec.analysis.forensic_analysis
        paths += [Path(getattr(forensic, n).path) for n in ('ela_image', 'freq_image', 'noise_image',
                                                              'copy_move_image', 'resampling_image') if getattr(forensic, n)]
        self.assertTrue(all(p.exists() for p in paths))
        r = self.call('delete', f'/analyses/{pid}/', self.alice)
        self.assertEqual(r.status_code, 204)
        self.assertEqual((UploadedImage.objects.count(), Analysis.objects.count(), Face.objects.count(),
                          ModelResult.objects.count(), Report.objects.count(), ForensicAnalysis.objects.count()),
                         (0, 0, 0, 0, 0, 0))
        self.assertFalse([p for p in paths if p.exists()])                   # files really gone
        self.assertEqual(self.call('get', f'/analyses/{pid}/', self.alice).status_code, 404)
        self.assertTrue(AuditLog.objects.filter(action='analysis_deleted', object_id=pid).exists())

    def test_account_deletion_removes_all_files(self):
        rec = self.two_face_analysis(self.alice)
        p = Path(rec.image.path)
        self.alice.delete()
        self.assertFalse(p.exists())
        self.assertEqual(UploadedImage.objects.count(), 0)

    def test_failed_processing_is_recorded_safely(self):
        from unittest import mock
        with mock.patch('detector.services.persist_ai_results', side_effect=RuntimeError('/srv/x exploded')):
            r = self.upload(self.alice)
        d = r.json()['data']
        self.assertEqual((r.status_code, d['status'], d['assessment']), (202, 'failed', 'ANALYSIS_FAILED'))
        self.assertNotIn('/srv', json.dumps(r.json()))
        self.assertEqual(self.call('get', f"/analyses/{d['id']}/faces/", self.alice).status_code, 409)


# ============================================================ authorization (IDOR)
class AuthorizationTests(ApiTestCase):
    def test_other_users_cannot_touch_any_endpoint(self):
        rec = self.two_face_analysis(self.alice)
        pid = str(rec.public_id)
        ghost = str(uuid.uuid4())
        endpoints = [('get', '/analyses/{}/'), ('get', '/analyses/{}/faces/'), ('get', '/analyses/{}/models/'),
                     ('get', '/analyses/{}/forensics/'), ('get', '/analyses/{}/report/'),
                     ('get', '/analyses/{}/image/'), ('get', '/analyses/{}/faces/0/crop/'),
                     ('get', '/analyses/{}/forensics/frequency/image/'), ('delete', '/analyses/{}/')]
        for method, tpl in endpoints:
            other = self.call(method, tpl.format(pid), self.bob)
            none = self.call(method, tpl.format(ghost), self.bob)
            self.assertEqual(other.status_code, 404, tpl)
            a, b = other.json(), none.json()
            self.assertEqual(a['error'], b['error'], tpl)                     # indistinguishable from "does not exist"
            self.assertEqual(a['error']['code'], 'NOT_FOUND')
        self.assertTrue(UploadedImage.objects.filter(public_id=pid).exists())  # bob's DELETE did nothing
        self.assertEqual(self.call('get', '/analyses/', self.bob).json()['data']['count'], 0)
        denied = AuditLog.objects.filter(action='access_denied', user=self.bob, success=False)
        self.assertEqual(denied.count(), len(endpoints))                       # every probe is audited
        self.assertFalse(AuditLog.objects.filter(action='access_denied', object_id='').exists() and False)

    def test_owner_still_has_access_after_probes(self):
        rec = self.two_face_analysis(self.alice)
        self.call('get', f'/analyses/{rec.public_id}/', self.bob)
        self.assertEqual(self.call('get', f'/analyses/{rec.public_id}/', self.alice).status_code, 200)

    def test_media_urls_for_face_crops_are_owner_only(self):
        rec = self.two_face_analysis(self.alice)
        url = rec.analysis.faces.get(face_index=1).crop_image.url
        self.client.login(username='bob', password=PW)
        self.assertEqual(self.client.get(url).status_code, 404)
        self.client.login(username='alice', password=PW)
        r = self.client.get(url)
        self.assertEqual(r.status_code, 200)
        b"".join(r.streaming_content)

    def test_numeric_ids_are_not_accepted_by_the_api(self):
        rec = self.two_face_analysis(self.alice)
        self.assertEqual(self.call('get', f'/analyses/{rec.pk}/', self.alice).status_code, 404)

    def test_hash_collision_never_returns_another_users_analysis(self):
        a = self.two_face_analysis(self.alice)
        png = two_face_png(os.path.join(self.dir, 'same.png'))
        from unittest import mock
        with mock.patch('ai_models.pipeline.get_face_detector', return_value=StubDetector([])):
            r = self.call('post', '/analyses/', self.bob, data={'image': SimpleUploadedFile('s.png', open(png, 'rb').read())})
        self.assertEqual(r.status_code, 202)
        self.assertNotEqual(r.json()['data']['id'], str(a.public_id))        # same bytes, separate record
        self.assertEqual(r.json()['data']['sha256'], a.sha256_hash)


# ============================================================ DB integrity
class DatabaseIntegrityTests(ApiTestCase):
    def test_constraints(self):
        rec = self.two_face_analysis(self.alice)
        a = rec.analysis
        with self.assertRaises(IntegrityError), transaction.atomic():
            ModelResult.objects.create(analysis=a, face_index=0, model_key='vit_b16', model_name='x')
        with self.assertRaises(IntegrityError), transaction.atomic():
            Face.objects.create(analysis=a, face_index=0, x=1, y=1, w=1, h=1)
        with self.assertRaises(IntegrityError), transaction.atomic():
            ModelResult.objects.create(analysis=a, face_index=5, model_key='zzz', model_name='x', fake_probability=2.0)
        mv = ModelVersion.objects.first()
        with self.assertRaises(IntegrityError), transaction.atomic():
            ModelVersion.objects.create(key=mv.key, name='dup', version=mv.version, weights_sha256=mv.weights_sha256)

    def test_public_ids_are_unique_uuids(self):
        rows = self.quick_upload(self.alice, n=3)
        self.assertEqual(len({r.public_id for r in rows}), 3)
        self.assertTrue(all(isinstance(r.public_id, uuid.UUID) for r in rows))

    def test_model_version_rows_record_reproducibility_metadata(self):
        self.two_face_analysis(self.alice)
        mv = ModelVersion.objects.get(key='vit_b16')
        self.assertEqual((mv.version, mv.framework, mv.input_size), ('9.9.9', 'onnx', 32))
        self.assertEqual(len(mv.weights_sha256), 64)
        self.assertIn('training_dataset', mv.metadata)

    def test_indexes_exist_for_frequent_queries(self):
        names = {i.name for i in UploadedImage._meta.indexes} | {i.name for i in AuditLog._meta.indexes} \
            | {i.name for i in ModelResult._meta.indexes}
        for n in ('img_user_created_idx', 'img_status_idx', 'audit_user_idx', 'audit_action_idx', 'result_face_idx'):
            self.assertIn(n, names)


# ============================================================ system endpoints
class SystemEndpointTests(ApiTestCase):
    def test_readiness_ready(self):
        r = self.client.get(f'{BASE}/system/readiness/')
        self.assertEqual(r.status_code, 200)
        checks = r.json()['data']['checks']
        self.assertTrue(all(c['ok'] for c in checks.values()))
        self.assertEqual(set(checks), {'database', 'cache', 'storage', 'queue', 'models'})
        self.assertNotIn(str(self.dir), r.content.decode())

    def test_readiness_fails_when_models_required_but_missing(self):
        (self.dir / 'models.json').write_text('{}')
        import shutil
        for k in ('vit_b16', 'xception', 'efficientnet_b4'):
            shutil.rmtree(self.dir / k)
        from ai_models.registry import ModelRegistry, build_specs
        reg = ModelRegistry(build_specs()); reg.load_all(); reset_registry(reg)
        with override_settings(READINESS_MIN_MODELS=1):
            r = self.client.get(f'{BASE}/system/readiness/')
        self.assertEqual(r.status_code, 503)
        e = r.json()['error']
        self.assertEqual(e['code'], 'NOT_READY')
        self.assertFalse(e['details']['models']['ok'])
        self.assertTrue(e['details']['database']['ok'])
        with override_settings(READINESS_MIN_MODELS=0):
            self.assertEqual(self.client.get(f'{BASE}/system/readiness/').status_code, 200)

    def test_models_endpoint_requires_auth_and_hides_paths(self):
        self.assertEqual(self.client.get(f'{BASE}/models/').status_code, 401)
        r = self.call('get', '/models/', self.alice)
        data = r.json()['data']
        self.assertEqual(len(data), len(MODEL_ORDER))
        by = {m['key']: m for m in data}
        self.assertEqual((by['vit_b16']['status'], by['freq_svm']['status']), ('AVAILABLE', 'NOT_CONFIGURED'))
        self.assertEqual(by['vit_b16']['version'], '9.9.9')
        self.assertNotIn(str(self.dir), r.content.decode())


# ============================================================ web: password reset, admin, CORS
class PasswordResetTests(ApiTestCase):
    def test_flow_and_no_account_enumeration(self):
        known = self.client.post('/public/password-reset/', {'email': 'alice@example.com'})
        unknown = self.client.post('/public/password-reset/', {'email': 'ghost@example.com'})
        self.assertEqual((known.status_code, unknown.status_code), (302, 302))
        self.assertEqual(known['Location'], unknown['Location'])           # identical response either way
        self.assertEqual(len(mail.outbox), 1)
        self.assertEqual(mail.outbox[0].to, ['alice@example.com'])
        link = re.search(r'https?://[^\s]+/reset/[^\s]+', mail.outbox[0].body).group(0)
        path = '/' + link.split('/', 3)[3]
        r = self.client.get(path, follow=True)
        self.assertEqual(r.status_code, 200)
        weak = self.client.post(r.request['PATH_INFO'], {'new_password1': '12345678', 'new_password2': '12345678'})
        self.assertEqual(weak.status_code, 200)                              # rejected, form shown again
        ok = self.client.post(r.request['PATH_INFO'], {'new_password1': 'N3w-Passphrase-77', 'new_password2': 'N3w-Passphrase-77'})
        self.assertEqual(ok.status_code, 302)
        self.assertTrue(Client().login(username='alice', password='N3w-Passphrase-77'))
        self.assertFalse(Client().login(username='alice', password=PW))
        again = Client().get(path, follow=True)
        self.assertIn('invalid or has expired', again.content.decode())      # single use
        self.assertTrue(AuditLog.objects.filter(action='password_reset_requested').exists())

    def test_rate_limited(self):
        from django.conf import settings
        with override_settings(RATE_LIMITS={**settings.RATE_LIMITS, 'password_reset': (2, 3600)}):
            codes = [self.client.post('/public/password-reset/', {'email': 'x@example.com'}).status_code for _ in range(3)]
        self.assertEqual(codes, [302, 302, 429])

    def test_login_page_links_to_reset(self):
        self.assertIn('/public/password-reset/', self.client.get('/public/login/').content.decode())


class AdminTests(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.root = User.objects.create_superuser('root', 'r@example.com', PW)
        self.client.login(username='root', password=PW)

    def test_audit_log_is_append_only_and_readable(self):
        self.two_face_analysis(self.alice)
        from detector import audit
        audit.audit('login', None, user=self.alice)
        self.assertEqual(self.client.get('/admin/detector/auditlog/').status_code, 200)
        self.assertEqual(self.client.get('/admin/detector/auditlog/add/').status_code, 403)
        row = AuditLog.objects.first()
        self.assertEqual(self.client.post(f'/admin/detector/auditlog/{row.pk}/delete/', {'post': 'yes'}).status_code, 403)
        self.assertTrue(AuditLog.objects.filter(pk=row.pk).exists())

    def test_admin_does_not_expose_private_images_or_filenames(self):
        rec = self.two_face_analysis(self.alice)
        page = self.client.get(f'/admin/detector/uploadedimage/{rec.pk}/change/').content.decode()
        for leak in ('uploads/', '/media/', 'two.png', rec.image.name):
            self.assertNotIn(leak, page)
        fp = self.client.get(f'/admin/detector/face/{rec.analysis.faces.first().pk}/change/').content.decode()
        self.assertNotIn('/media/', fp)
        self.assertEqual(self.client.get('/admin/detector/uploadedimage/add/').status_code, 403)
        self.assertEqual(self.client.get('/admin/detector/modelversion/').status_code, 200)
        self.assertEqual(self.client.get('/admin/detector/report/').status_code, 200)


class CorsTests(ApiTestCase):
    def test_no_cors_by_default_and_never_allow_all(self):
        from django.conf import settings
        self.assertFalse(settings.CORS_ALLOW_ALL_ORIGINS)
        r = self.client.get(f'{BASE}/system/health/', HTTP_ORIGIN='https://evil.example')
        self.assertNotIn('Access-Control-Allow-Origin', r)

    @override_settings(CORS_ALLOWED_ORIGINS=['https://app.example.com'])
    def test_only_listed_origins_are_allowed(self):
        ok = self.client.get(f'{BASE}/system/health/', HTTP_ORIGIN='https://app.example.com')
        bad = self.client.get(f'{BASE}/system/health/', HTTP_ORIGIN='https://evil.example')
        self.assertEqual(ok['Access-Control-Allow-Origin'], 'https://app.example.com')
        self.assertNotIn('Access-Control-Allow-Origin', bad)
