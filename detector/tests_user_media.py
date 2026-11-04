import io
import os
import tempfile

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from detector.media import user_owns_media
from detector.models import UploadedImage, user_media_prefix


def _png():
    from PIL import Image
    b = io.BytesIO()
    Image.new('RGB', (8, 8), 'red').save(b, 'PNG')
    return b.getvalue()


class PerUserMediaTests(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ctx = override_settings(MEDIA_ROOT=self.tmp.name)
        self.ctx.enable()
        self.a = User.objects.create_user('alice', password='x')
        self.b = User.objects.create_user('bob', password='x')

    def tearDown(self):
        self.ctx.disable()
        self.tmp.cleanup()

    def _up(self, user):
        return UploadedImage.objects.create(
            user=user, original_filename='p.png', file_size=1,
            image=SimpleUploadedFile('p.png', _png(), content_type='image/png'))

    def test_separate_folders(self):
        ua, ub = self._up(self.a), self._up(self.b)
        self.assertTrue(ua.image.name.startswith(user_media_prefix(self.a.pk, 'uploads') + '/'))
        self.assertTrue(ub.image.name.startswith(user_media_prefix(self.b.pk, 'uploads') + '/'))
        self.assertNotEqual(os.path.dirname(ua.image.name), os.path.dirname(ub.image.name))

    def test_ownership_checks_prefix(self):
        ua = self._up(self.a)
        self.assertTrue(user_owns_media(self.a, ua.image.name))
        self.assertFalse(user_owns_media(self.b, ua.image.name))
        self.assertFalse(user_owns_media(self.b, f'users/user_{self.a.pk}/uploads/x.png'))

    def test_submit_stores_hash(self):
        import hashlib
        from unittest import mock
        from detector import jobs
        data = _png()
        f = SimpleUploadedFile('p.png', data, content_type='image/png')
        with mock.patch.object(jobs, 'enqueue'):
            rec = jobs.submit(self.a, f, 'p.png')
        self.assertEqual(rec.sha256_hash, hashlib.sha256(data).hexdigest())

    def test_delete_account_removes_folder(self):
        from detector.retention import delete_account
        ua = self._up(self.a)
        folder = os.path.join(self.tmp.name, 'users', f'user_{self.a.pk}')
        self.assertTrue(os.path.isdir(folder))
        delete_account(self.a)
        self.assertFalse(os.path.exists(folder))
        self.assertFalse(UploadedImage.objects.filter(pk=ua.pk).exists())
