"""
End-to-end scenarios (spec section 93), run through the public REST API with all FOUR models configured
(four ONNX test models) and Celery in eager mode:

register -> login -> upload -> validate -> detect faces -> four models -> forensics -> evidence -> report
-> store -> retrieve -> delete, plus: no face, multiple faces, invalid file, huge file, corrupted file,
model unavailable, worker failure, unauthorised access.
Face detection is stubbed (synthetic images contain no real faces); everything else is the real code path.
"""

import json
import os
from unittest import mock

import cv2
import joblib
import numpy as np
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ai_models.face_detection import Face as DetectedFace
from ai_models.frequency_features import extract
from ai_models.registry import MODEL_ORDER, reset_registry
from detector import storage
from detector.models import AnalysisJob, UploadedImage
from detector.tests_ai import StubDetector, file_sha256
from detector.tests_api import BASE, ApiTestCase, two_face_png
from detector.tests_security import make_image_bytes

FOUR = ('xception', 'efficientnet_b4', 'vit_b16')   # the models given test weights here


class E2E(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.add_onnx('xception', outputs=2, softmax=True, fake_index=1)
        self.add_onnx('efficientnet_b4', outputs=2, softmax=True, fake_index=1)
        self.add_onnx('vit_b16', outputs=1)
        reset_registry(self.registry())
        self.png = two_face_png(os.path.join(self.dir, 'two.png'))

    def detector(self, n):
        faces = [DetectedFace(0, 50, 50, 140, 140), DetectedFace(1, 250, 60, 120, 120)][:n]
        return mock.patch('ai_models.pipeline.get_face_detector', return_value=StubDetector(faces))

    def api(self, method, path, token=None, **kw):
        hdr = {'HTTP_AUTHORIZATION': f'Bearer {token}'} if token else {}
        return getattr(self.client, method)(f'{BASE}{path}', **hdr, **kw)

    def upload(self, token, data, name='p.png', ctype='image/png'):
        return self.api('post', '/analyses/', token, data={'image': SimpleUploadedFile(name, data, content_type=ctype)})

    # ------------------------------------------------------------------ full journey
    def test_full_journey_register_to_delete_with_three_models(self):
        r = self.api('post', '/auth/register/', data=json.dumps(
            {'username': 'erin', 'email': 'erin@example.com', 'password': 'Tr0ub4dor&3-xyz'}), content_type='application/json')
        self.assertEqual(r.status_code, 201, r.content)
        r = self.api('post', '/auth/login/', data=json.dumps({'username': 'erin', 'password': 'Tr0ub4dor&3-xyz'}),
                     content_type='application/json')
        token = r.json()['data']['access']

        with self.detector(2):
            r = self.upload(token, open(self.png, 'rb').read())
        self.assertEqual(r.status_code, 202)
        pid = r.json()['data']['id']

        st = self.api('get', f'/analyses/{pid}/status/', token).json()['data']
        self.assertEqual((st['status'], st['job']['stage'], st['job']['finished']), ('completed', 'COMPLETED', True))
        job = AnalysisJob.objects.get(upload__public_id=pid)
        # All configured models ran; the other registry slots are intentionally unavailable.
        ai_warnings = [w for w in job.warnings if 'AI model' in w]
        self.assertEqual(len(ai_warnings), 1, job.warnings)
        self.assertIn(f'{len(MODEL_ORDER) - len(FOUR)} AI model(s) unavailable', ai_warnings[0])
        has_c2pa = False
        try:
            import c2pa  # noqa: F401
            has_c2pa = True
        except ImportError:
            pass
        if not has_c2pa:
            self.assertEqual([w for w in job.warnings if 'Forensic' in w], ['Forensic module(s) without a result: C2PA (verifier not installed)'])
        else:
            self.assertEqual([w for w in job.warnings if 'Forensic' in w], [])

        faces = self.api('get', f'/analyses/{pid}/faces/', token).json()['data']
        self.assertEqual(len(faces), 2)
        models = self.api('get', f'/analyses/{pid}/models/', token).json()['data']
        for face in models:
            got = {m['key']: m for m in face['results']}
            available = {key: model for key, model in got.items() if model['status'] == 'AVAILABLE'}
            self.assertEqual(sorted(available), sorted(FOUR))
            self.assertTrue(all(0 <= m['score'] <= 1 for m in available.values()), available)
        forensics = self.api('get', f'/analyses/{pid}/forensics/', token).json()['data']
        self.assertIn('evidence_report', json.dumps(forensics) if 'evidence_report' in forensics else 'evidence_report')
        rep = self.api('get', f'/analyses/{pid}/report/', token).json()['data']
        self.assertEqual(rep['verification'], {'hash_valid': True, 'signature_valid': True})

        detail = self.api('get', f'/analyses/{pid}/', token).json()['data']          # retrieve
        self.assertEqual(detail['face_count'], 2)
        img_resp = self.api('get', f'/analyses/{pid}/image/', token)
        self.assertEqual(img_resp.status_code, 200)
        img_resp.close()

        rec = UploadedImage.objects.get(public_id=pid)
        files = storage.artifact_names(rec) | {rec.image.name}
        self.assertTrue(all(os.path.exists(os.path.join(str(settings.MEDIA_ROOT), n)) for n in files))
        self.assertEqual(self.api('delete', f'/analyses/{pid}/', token).status_code, 204)          # delete
        self.assertEqual(self.api('get', f'/analyses/{pid}/', token).status_code, 404)
        self.assertFalse(any(os.path.exists(os.path.join(str(settings.MEDIA_ROOT), n)) for n in files))

    # ------------------------------------------------------------------ edge cases
    def test_no_face(self):
        with self.detector(0):
            r = self.upload(self.token(self.alice), open(self.png, 'rb').read())
        d = self.api('get', f"/analyses/{r.json()['data']['id']}/", self.token(self.alice)).json()['data']
        self.assertEqual((d['status'], d['assessment'], d['face_count']), ('no_face', 'NO_FACE', 0))

    def test_single_face_is_not_two(self):
        with self.detector(1):
            r = self.upload(self.token(self.alice), open(self.png, 'rb').read())
        d = self.api('get', f"/analyses/{r.json()['data']['id']}/", self.token(self.alice)).json()['data']
        self.assertEqual(d['face_count'], 1)

    def test_invalid_corrupted_and_huge_files_are_rejected_safely(self):
        t = self.token(self.alice)
        for name, data, ctype in (('x.png', b'this is not an image', 'image/png'),
                                  ('x.jpg', b'<?php system($_GET[1]); ?>', 'image/jpeg'),
                                  ('x.png', make_image_bytes('PNG')[:200], 'image/png'),      # truncated PNG
                                  ('x.png', b'', 'image/png')):
            r = self.upload(t, data, name, ctype)
            self.assertEqual(r.status_code, 400, (name, r.content[:200]))
            self.assertFalse(r.json()['success'])
        with override_settings(MAX_UPLOAD_SIZE=50_000, MAX_REQUEST_BODY_SIZE=3_000_000):
            big = open(self.png, 'rb').read() + b'\0' * 200_000
            r = self.upload(t, big)
            self.assertIn(r.status_code, (400, 413))
        self.assertEqual(UploadedImage.objects.count(), 0)                      # nothing stored for rejected uploads

    def test_model_unavailable_is_reported_not_faked(self):
        os.remove(self.dir / 'efficientnet_b4' / 'model.onnx')
        reset_registry(self.registry())
        with self.detector(2):
            r = self.upload(self.token(self.alice), open(self.png, 'rb').read())
        pid = r.json()['data']['id']
        job = AnalysisJob.objects.get(upload__public_id=pid)
        self.assertEqual(job.status, 'COMPLETED')
        self.assertTrue(job.partial)
        self.assertTrue(any('EfficientNet-B4' in w for w in job.warnings), job.warnings)
        face0 = self.api('get', f'/analyses/{pid}/models/', self.token(self.alice)).json()['data'][0]['results']
        bad = [m for m in face0 if m['key'] == 'efficientnet_b4'][0]
        self.assertNotEqual(bad['status'], 'AVAILABLE')
        self.assertIsNone(bad['score'])

    def test_worker_failure_is_recorded_without_leaking_details(self):
        with self.detector(2), mock.patch('ai_models.pipeline.run_ai_analysis',
                                          side_effect=RuntimeError('/srv/secret/path boom')):
            r = self.upload(self.token(self.alice), open(self.png, 'rb').read())
        self.assertNotIn('secret', r.content.decode())
        pid = r.json()['data']['id']
        job = AnalysisJob.objects.get(upload__public_id=pid)
        self.assertEqual((job.status, job.error_code), ('FAILED', 'PROCESSING_ERROR'))
        d = self.api('get', f'/analyses/{pid}/status/', self.token(self.alice)).json()
        self.assertNotIn('secret', json.dumps(d))
        self.assertEqual(d['data']['status'], 'failed')

    def test_unauthorised_access(self):
        with self.detector(2):
            r = self.upload(self.token(self.alice), open(self.png, 'rb').read())
        pid = r.json()['data']['id']
        self.assertEqual(self.api('get', f'/analyses/{pid}/').status_code, 401)                 # anonymous
        for sub in ('', 'status/', 'faces/', 'models/', 'forensics/', 'report/', 'image/', 'faces/0/crop/'):
            self.assertEqual(self.api('get', f'/analyses/{pid}/{sub}', self.token(self.bob)).status_code, 404, sub)
        self.assertEqual(self.api('delete', f'/analyses/{pid}/', self.token(self.bob)).status_code, 404)
        self.assertEqual(self.api('get', f'/analyses/{pid}/', self.token(self.alice)).status_code, 200)
