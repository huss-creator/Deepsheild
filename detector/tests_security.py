"""
Phase 3 security tests: upload validation, media authorization, redirects,
CSRF, logout, rate limiting, request limits, error leakage, EXIF redaction,
and production-settings safety.
"""

import io
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
from django.conf import settings
from django.contrib.auth.models import User
from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.core.files.uploadedfile import SimpleUploadedFile
from django.http import Http404, HttpResponse
from django.test import Client, RequestFactory, TestCase, override_settings
from django.urls import reverse
from PIL import Image

from detector.forms import ImageUploadForm
from detector.image_validation import validate_image_bytes
from detector.media import protected_media
from detector.middleware import RequestSizeLimitMiddleware
from detector.models import UploadedImage
from forensics.metadata import extract_metadata

BASE_DIR = Path(settings.BASE_DIR)


def make_image_bytes(fmt='PNG', size=(200, 200), mode='RGB'):
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(arr).convert(mode)
    buf = io.BytesIO()
    img.save(buf, fmt)
    return buf.getvalue()


class TempMediaTestCase(TestCase):
    """Each test class gets an isolated MEDIA_ROOT and a clean rate-limit cache."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._media = tempfile.mkdtemp(prefix='ds_media_')
        cls._override = override_settings(MEDIA_ROOT=cls._media)
        cls._override.enable()

    @classmethod
    def tearDownClass(cls):
        cls._override.disable()
        shutil.rmtree(cls._media, ignore_errors=True)
        super().tearDownClass()

    def setUp(self):
        cache.clear()
        self.alice = User.objects.create_user('alice', 'a@example.com', 'Correct-Horse-9')
        self.bob = User.objects.create_user('bob', 'b@example.com', 'Correct-Horse-9')

    def make_upload(self, user, data=None, name='x.png'):
        return UploadedImage.objects.create(
            user=user, image=SimpleUploadedFile(name, data or make_image_bytes()),
            original_filename=name, file_size=10)


# ------------------------------------------------------------------ validation
class ImageValidationTests(TestCase):
    def test_valid_png_and_jpeg(self):
        png = validate_image_bytes(make_image_bytes('PNG'))
        jpg = validate_image_bytes(make_image_bytes('JPEG'))
        self.assertEqual((png.format, png.extension), ('PNG', 'png'))
        self.assertEqual((jpg.format, jpg.extension), ('JPEG', 'jpg'))
        self.assertEqual((png.width, png.height), (200, 200))

    def test_rejects_non_image_content(self):
        for payload in (b'hello world', b'MZ\x90\x00' + b'\0' * 100, b'\x7fELF' + b'\0' * 100,
                        b'<?php system($_GET[0]); ?>', b'GIF89a' + b'\0' * 100):
            with self.assertRaises(ValidationError, msg=payload[:8]):
                validate_image_bytes(payload)

    def test_rejects_empty(self):
        with self.assertRaises(ValidationError):
            validate_image_bytes(b'')

    def test_rejects_truncated_image(self):
        data = make_image_bytes('PNG')
        with self.assertRaises(ValidationError) as ctx:
            validate_image_bytes(data[: len(data) // 2])
        self.assertEqual(ctx.exception.code, 'corrupt')

    @override_settings(MAX_UPLOAD_SIZE=1000)
    def test_rejects_oversized_file(self):
        with self.assertRaises(ValidationError) as ctx:
            validate_image_bytes(make_image_bytes('PNG'))
        self.assertEqual(ctx.exception.code, 'too_large')

    @override_settings(MAX_PIXELS=1_000_000)
    def test_rejects_pixel_bomb(self):
        # 3000x3000 solid PNG is tiny on disk but 9M pixels when decoded
        buf = io.BytesIO()
        Image.new('L', (3000, 3000), 0).save(buf, 'PNG')
        self.assertLess(len(buf.getvalue()), 100_000)
        with self.assertRaises(ValidationError) as ctx:
            validate_image_bytes(buf.getvalue())
        self.assertEqual(ctx.exception.code, 'too_many_pixels')

    @override_settings(MAX_IMAGE_WIDTH=100)
    def test_rejects_excess_width(self):
        with self.assertRaises(ValidationError):
            validate_image_bytes(make_image_bytes('PNG', (200, 50)))

    def test_rejects_tiny_image(self):
        with self.assertRaises(ValidationError) as ctx:
            validate_image_bytes(make_image_bytes('PNG', (8, 8)))
        self.assertEqual(ctx.exception.code, 'too_small')

    def test_rejects_polyglot_trailers(self):
        for fmt in ('JPEG', 'PNG'):
            for trailer in (b'<?php echo 1; ?>', b'<script>alert(1)</script>', b'PK\x03\x04zip'):
                with self.assertRaises(ValidationError, msg=f'{fmt} {trailer[:6]}') as ctx:
                    validate_image_bytes(make_image_bytes(fmt) + trailer)
                self.assertEqual(ctx.exception.code, 'suspicious')

    def test_benign_trailing_bytes_not_rejected(self):
        # Some cameras append vendor trailers; plain binary padding is allowed.
        validate_image_bytes(make_image_bytes('JPEG') + b'\x00' * 32)

    def test_rejects_animated_png(self):
        frames = [Image.new('RGB', (64, 64), c) for c in ('red', 'blue')]
        buf = io.BytesIO()
        frames[0].save(buf, 'PNG', save_all=True, append_images=frames[1:])
        with self.assertRaises(ValidationError):
            validate_image_bytes(buf.getvalue())

    def test_extension_and_type_are_not_trusted(self):
        """A PHP payload named photo.jpg with image/jpeg must fail; real PNG named
        evil.php is accepted but stored with a server-chosen .png name."""
        bad = SimpleUploadedFile('photo.jpg', b'<?php evil(); ?>', content_type='image/jpeg')
        self.assertFalse(ImageUploadForm({}, {'image': bad}).is_valid())

        good = SimpleUploadedFile('evil.php', make_image_bytes('PNG'), content_type='text/x-php')
        form = ImageUploadForm({'consent': 'on'}, {'image': good})
        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(form.cleaned_data['image'].name, 'upload.png')

    def test_display_name_sanitised(self):
        f = SimpleUploadedFile('../../etc/pass\x00wd\n.png', make_image_bytes('PNG'))
        form = ImageUploadForm({'consent': 'on'}, {'image': f})
        self.assertTrue(form.is_valid(), form.errors)
        self.assertNotIn('/', form.original_filename)
        self.assertNotIn('\x00', form.original_filename)
        self.assertLessEqual(len(form.original_filename), 255)

    def test_error_messages_do_not_leak_internals(self):
        with self.assertRaises(ValidationError) as ctx:
            validate_image_bytes(b'\xff\xd8\xff' + b'garbage' * 50)
        msg = ' '.join(ctx.exception.messages)
        for leak in ('Traceback', 'PIL', '/', 'OSError'):
            self.assertNotIn(leak, msg)


# ----------------------------------------------------------- upload view e2e
class UploadViewTests(TempMediaTestCase):
    def setUp(self):
        super().setUp()
        self.client.login(username='alice', password='Correct-Horse-9')

    def _post(self, data, name='pic.png', ctype='image/png'):
        return self.client.post(reverse('upload'),
                                {'image': SimpleUploadedFile(name, data, content_type=ctype), 'consent': 'on'})

    def test_valid_upload_uses_server_generated_filename(self):
        resp = self._post(make_image_bytes('PNG'), name='../../evil name.php')
        self.assertEqual(resp.status_code, 302)
        rec = UploadedImage.objects.get(user=self.alice)
        self.assertRegex(rec.image.name, r'^users/user_\d+/uploads/[0-9a-f-]{36}\.png$')
        self.assertTrue(os.path.realpath(rec.image.path).startswith(os.path.realpath(self._media)))
        self.assertNotIn('..', rec.original_filename)
        # random noise has no face: the models must NOT be run on the whole image
        self.assertEqual(rec.processing_status, UploadedImage.STATUS_NO_FACE)
        self.assertEqual(rec.analysis.final_prediction, 'NO_FACE')
        self.assertEqual(rec.analysis.model_results.count(), 0)

    def test_malicious_upload_rejected_and_not_stored(self):
        resp = self._post(b'<?php system($_GET["c"]); ?>', name='shell.jpg', ctype='image/jpeg')
        self.assertEqual(resp.status_code, 200)            # form re-rendered with error
        self.assertEqual(UploadedImage.objects.count(), 0)
        self.assertFalse(list(Path(self._media).rglob('*.php')))

    def test_processing_failure_does_not_leak_exception_text(self):
        with mock.patch('ai_models.pipeline.run_ai_analysis',
                        side_effect=RuntimeError('/srv/secret/path/model.h5 exploded')):
            resp = self._post(make_image_bytes('PNG'), name='ok.png')
        rec = UploadedImage.objects.get(user=self.alice)
        self.assertEqual(rec.processing_status, UploadedImage.STATUS_FAILED)
        self.assertNotIn('secret', rec.error_message)
        page = self.client.get(resp['Location'], follow=True).content.decode()
        self.assertNotIn('/srv/secret', page)

    @override_settings(RATE_LIMITS={**settings.RATE_LIMITS, 'upload': (2, 3600)})
    def test_upload_rate_limit(self):
        codes = [self._post(b'not an image').status_code for _ in range(3)]
        self.assertEqual(codes, [200, 200, 429])


# ------------------------------------------------------- media authorization
class ProtectedMediaTests(TempMediaTestCase):
    def test_anonymous_redirected_to_login(self):
        up = self.make_upload(self.alice)
        resp = self.client.get(up.image.url)
        self.assertEqual(resp.status_code, 302)
        self.assertIn('/public/login/', resp['Location'])

    def test_owner_can_fetch_with_private_headers(self):
        up = self.make_upload(self.alice)
        self.client.login(username='alice', password='Correct-Horse-9')
        resp = self.client.get(up.image.url)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp['Content-Type'], 'image/png')
        self.assertIn('no-store', resp['Cache-Control'])
        self.assertEqual(resp['X-Content-Type-Options'], 'nosniff')
        b"".join(resp.streaming_content)

    def test_other_user_gets_404_for_upload(self):
        up = self.make_upload(self.alice)
        self.client.login(username='bob', password='Correct-Horse-9')
        self.assertEqual(self.client.get(up.image.url).status_code, 404)

    def test_unreferenced_file_on_disk_is_404(self):
        Path(self._media, 'uploads').mkdir(exist_ok=True)
        Path(self._media, 'uploads', 'orphan.png').write_bytes(make_image_bytes('PNG'))
        self.client.login(username='alice', password='Correct-Horse-9')
        self.assertEqual(self.client.get('/media/uploads/orphan.png').status_code, 404)

    def test_path_traversal_rejected(self):
        rf = RequestFactory()
        for evil in ('../db.sqlite3', 'uploads/../../etc/passwd', '/etc/passwd',
                     'uploads/..\\..\\secret', 'uploads/x\x00.png'):
            req = rf.get('/media/x')
            req.user = self.alice
            with self.assertRaises(Http404, msg=evil):
                protected_media(req, evil)

    def test_x_accel_redirect_when_configured(self):
        up = self.make_upload(self.alice)
        self.client.login(username='alice', password='Correct-Horse-9')
        with override_settings(MEDIA_X_ACCEL_PREFIX='/_protected_media/'):
            resp = self.client.get(up.image.url)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp['X-Accel-Redirect'], '/_protected_media/' + up.image.name)
        self.assertEqual(resp.content, b'')

    def test_result_detail_report_are_owner_only(self):
        up = self.make_upload(self.alice)
        self.client.login(username='bob', password='Correct-Horse-9')
        for name in ('result', 'detail', 'report'):
            self.assertEqual(self.client.get(reverse(name, args=[up.pk])).status_code, 404, name)


# ----------------------------------------------------------- auth hardening
class AuthHardeningTests(TempMediaTestCase):
    def _login(self, **extra):
        return self.client.post(reverse('login'), {'username': 'alice',
                                                   'password': 'Correct-Horse-9', **extra})

    def test_open_redirect_blocked(self):
        for evil in ('https://evil.example/', '//evil.example', 'http://evil.example/x',
                     'javascript:alert(1)'):
            self.client.logout()
            resp = self._login(next=evil)
            self.assertEqual(resp.status_code, 302)
            self.assertEqual(resp['Location'], reverse('upload'), evil)

    def test_safe_next_is_honoured(self):
        resp = self._login(next='/auth/history/')
        self.assertEqual(resp['Location'], '/auth/history/')

    def test_login_form_carries_next(self):
        page = self.client.get(reverse('login') + '?next=/auth/history/').content.decode()
        self.assertIn('name="next" value="/auth/history/"', page)

    def test_logout_requires_post(self):
        self.client.login(username='alice', password='Correct-Horse-9')
        self.assertEqual(self.client.get(reverse('logout')).status_code, 405)
        self.assertIn('_auth_user_id', self.client.session)
        self.assertEqual(self.client.post(reverse('logout')).status_code, 302)
        self.assertNotIn('_auth_user_id', self.client.session)

    def test_csrf_enforced_on_login_and_logout(self):
        strict = Client(enforce_csrf_checks=True)
        self.assertEqual(strict.post(reverse('login'), {'username': 'alice',
                                                        'password': 'Correct-Horse-9'}).status_code, 403)
        strict.force_login(self.alice)
        self.assertEqual(strict.post(reverse('logout')).status_code, 403)

    @override_settings(RATE_LIMITS={**settings.RATE_LIMITS, 'login_fail': (3, 900)})
    def test_login_lockout_after_repeated_failures(self):
        for _ in range(3):
            r = self.client.post(reverse('login'), {'username': 'alice', 'password': 'wrong'})
            self.assertEqual(r.status_code, 200)
        r = self._login()                      # correct password, but locked out
        self.assertEqual(r.status_code, 429)
        self.assertIn('Retry-After', r)
        self.assertNotIn('_auth_user_id', self.client.session)

    @override_settings(RATE_LIMITS={**settings.RATE_LIMITS, 'login_fail': (3, 900)})
    def test_lockout_is_per_username(self):
        for _ in range(3):
            self.client.post(reverse('login'), {'username': 'alice', 'password': 'wrong'})
        r = self.client.post(reverse('login'), {'username': 'bob', 'password': 'Correct-Horse-9'})
        self.assertEqual(r.status_code, 302)

    @override_settings(RATE_LIMITS={**settings.RATE_LIMITS, 'register': (2, 3600)})
    def test_registration_rate_limit(self):
        codes = [self.client.post(reverse('register'), {}).status_code for _ in range(3)]
        self.assertEqual(codes, [200, 200, 429])

    def test_spoofed_forwarded_for_ignored_without_trusted_proxy(self):
        from detector.ratelimit import client_ip
        req = RequestFactory().get('/', REMOTE_ADDR='10.0.0.5', HTTP_X_FORWARDED_FOR='1.2.3.4')
        self.assertEqual(client_ip(req), '10.0.0.5')
        with override_settings(NUM_PROXIES=1):
            self.assertEqual(client_ip(req), '1.2.3.4')

    def test_short_password_rejected(self):
        r = self.client.post(reverse('register'), {
            'username': 'newuser', 'email': 'n@example.com',
            'password1': 'Ab1!xyz', 'password2': 'Ab1!xyz'})
        self.assertEqual(r.status_code, 200)
        self.assertFalse(User.objects.filter(username='newuser').exists())


# ---------------------------------------------------------- request limits
class RequestSizeTests(TestCase):
    def test_oversized_content_length_rejected_before_parsing(self):
        mw = RequestSizeLimitMiddleware(lambda r: HttpResponse('ok'))
        rf = RequestFactory()
        big = rf.post('/auth/upload/', {}, CONTENT_LENGTH=str(settings.MAX_REQUEST_BODY_SIZE + 1))
        self.assertEqual(mw(big).status_code, 413)
        bad = rf.post('/auth/upload/', {}, CONTENT_LENGTH='abc')
        self.assertEqual(mw(bad).status_code, 400)
        ok = rf.post('/auth/upload/', {}, CONTENT_LENGTH='100')
        self.assertEqual(mw(ok).status_code, 200)


# ------------------------------------------------------------ EXIF privacy
class MetadataRedactionTests(TestCase):
    def test_gps_and_identifying_tags_are_not_stored(self):
        exif = Image.Exif()
        exif[0x010F] = 'TestCam'                  # Make
        exif[0x0110] = 'Model-X'                  # Model
        exif[0x013B] = 'Jane Q. Private'          # Artist
        exif[0x010E] = 'Photo of my home'         # ImageDescription
        exif[0x8825] = {1: 'N', 2: (31.0, 30.0, 0.0), 3: 'E', 4: (74.0, 20.0, 0.0)}
        path = os.path.join(tempfile.mkdtemp(), 'gps.jpg')
        Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(path, 'JPEG', exif=exif)
        self.addCleanup(shutil.rmtree, os.path.dirname(path), True)

        # precondition: the sensitive data really is in the file
        raw = Image.open(path).getexif()
        self.assertEqual(raw.get(0x013B), 'Jane Q. Private')
        self.assertTrue(raw.get_ifd(0x8825))

        meta = extract_metadata(path)
        blob = meta['meta_raw_json'] + meta['meta_note'] + meta['meta_camera_make']
        self.assertNotIn('Jane', blob)
        self.assertNotIn('my home', blob)
        self.assertNotIn('GPSInfo', meta['meta_raw_json'])
        self.assertNotIn('31.0', meta['meta_raw_json'])
        self.assertIn('GPS', meta['meta_note'])            # presence disclosed, values not
        self.assertEqual(meta['meta_camera_make'], 'TestCam')   # non-sensitive kept


# ----------------------------------------------------- settings
class SettingsTests(TestCase):
    def test_no_secret_key_hardcoded_in_source(self):
        for py in (BASE_DIR / 'config' / 'settings').glob('*.py'):
            text = py.read_text()
            self.assertNotIn('django-deepshield-secret', text)
            self.assertNotIn('django-insecure', text.replace("startswith('django-insecure')", ''))
