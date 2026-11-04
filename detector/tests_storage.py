"""Phase 8 tests: S3-compatible private storage (moto), signed delivery, deletion, local passthrough."""

import os
from unittest import mock

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings

from detector import storage
from detector.models import Face
from detector.tests_api import ApiTestCase, two_face_png
from detector.tests_ai import StubDetector
from ai_models.face_detection import Face as DetectedFace
from detector.services import process_upload

class LocalStorageTests(ApiTestCase):
    def test_local_backend_is_passthrough(self):
        self.assertFalse(storage.uses_remote_storage())
        rec = self.quick_upload(self.alice)[0]
        self.assertEqual(storage.publish_artifacts(rec), 0)
        with storage.local_copy(rec.image) as p:
            self.assertEqual(p, rec.image.path)
