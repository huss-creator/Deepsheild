"""Tests for landing, registration, login and logout (Member 3)."""

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

PASSWORD = 'Str0ng!Pass_2026'


class PublicPages(TestCase):
    def test_pages_open(self):
        for name in ('home', 'login', 'register'):
            self.assertEqual(self.client.get(reverse(name)).status_code, 200)


class Registration(TestCase):
    def data(self, **extra):
        return {'username': 'ali', 'email': 'ali@example.com', 'password1': PASSWORD, 'password2': PASSWORD, **extra}

    def test_register_creates_user_with_hashed_password(self):
        r = self.client.post(reverse('register'), self.data())
        self.assertRedirects(r, reverse('login'))
        user = User.objects.get(username='ali')
        self.assertNotEqual(user.password, PASSWORD)
        self.assertTrue(user.check_password(PASSWORD))

    def test_duplicate_email_rejected(self):
        User.objects.create_user('first', 'ali@example.com', PASSWORD)
        r = self.client.post(reverse('register'), self.data())
        self.assertEqual(r.status_code, 200)
        self.assertFalse(User.objects.filter(username='ali').exists())

    def test_password_mismatch_rejected(self):
        r = self.client.post(reverse('register'), self.data(password2='different'))
        self.assertEqual(r.status_code, 200)
        self.assertFalse(User.objects.filter(username='ali').exists())


class LoginLogout(TestCase):
    def setUp(self):
        User.objects.create_user('sara', 'sara@example.com', PASSWORD)

    def test_login_ok_then_logout(self):
        self.assertRedirects(self.client.post(reverse('login'), {'username': 'sara', 'password': PASSWORD}), reverse('dashboard'))
        self.assertContains(self.client.get(reverse('dashboard')), 'sara')
        self.assertRedirects(self.client.get(reverse('logout')), reverse('home'))

    def test_wrong_password(self):
        r = self.client.post(reverse('login'), {'username': 'sara', 'password': 'nope'})
        self.assertContains(r, 'Invalid username or password')

    def test_next_to_other_site_is_ignored(self):
        r = self.client.post(reverse('login') + '?next=https://evil.example/', {'username': 'sara', 'password': PASSWORD})
        self.assertRedirects(r, reverse('dashboard'))


class Dashboard(TestCase):
    def test_guest_is_sent_to_login(self):
        r = self.client.get(reverse('dashboard'))
        self.assertRedirects(r, reverse('login') + '?next=' + reverse('dashboard'))

    def test_logged_in_user_sees_their_details(self):
        User.objects.create_user('sara', 'sara@example.com', PASSWORD)
        self.client.login(username='sara', password=PASSWORD)
        r = self.client.get(reverse('dashboard'))
        self.assertContains(r, 'sara@example.com')
        self.assertContains(r, 'Welcome back')

    def test_login_with_next_goes_to_the_requested_page(self):
        User.objects.create_user('sara', 'sara@example.com', PASSWORD)
        r = self.client.post(reverse('login') + '?next=/dashboard/', {'username': 'sara', 'password': PASSWORD})
        self.assertRedirects(r, '/dashboard/')


# ----------------------------------------------------------------- Analyze page (interface only)
import struct
import zlib

from django.core.files.uploadedfile import SimpleUploadedFile


def _png_bytes():
    """A valid 1x1 PNG built by hand (no Pillow needed)."""
    def chunk(t, d):
        c = struct.pack('>I', len(d)) + t + d
        return c + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(b'\x00\x0a\x78\xc8')) + chunk(b'IEND', b''))


class AnalyzePage(TestCase):
    def setUp(self):
        User.objects.create_user('sara', 'sara@example.com', PASSWORD)

    def test_guest_is_sent_to_login(self):
        self.assertRedirects(self.client.get(reverse('analyze')), '/login/?next=/analyze/')

    def test_page_opens_when_logged_in(self):
        self.client.login(username='sara', password=PASSWORD)
        self.assertContains(self.client.get(reverse('analyze')), 'Analyze an image')

    def test_valid_image_goes_to_result_page(self):
        self.client.login(username='sara', password=PASSWORD)
        r = self.client.post(reverse('analyze'), {'image': SimpleUploadedFile('a.png', _png_bytes(), 'image/png')})
        self.assertRedirects(r, reverse('result'))

    def test_fake_image_file_is_rejected(self):
        self.client.login(username='sara', password=PASSWORD)
        r = self.client.post(reverse('analyze'), {'image': SimpleUploadedFile('a.png', b'hello world, not an image', 'image/png')})
        self.assertContains(r, 'Only JPG, PNG or WEBP')
        self.assertNotIn('last_upload', self.client.session)


class ResultPage(TestCase):
    def setUp(self):
        User.objects.create_user('sara', 'sara@example.com', PASSWORD)

    def test_guest_is_sent_to_login(self):
        self.assertRedirects(self.client.get(reverse('result')), '/login/?next=/result/')

    def test_without_upload_goes_back_to_analyze(self):
        self.client.login(username='sara', password=PASSWORD)
        self.assertRedirects(self.client.get(reverse('result')), reverse('analyze'))

    def test_result_shows_real_hash_and_sample_label(self):
        import hashlib
        data = _png_bytes()
        self.client.login(username='sara', password=PASSWORD)
        self.client.post(reverse('analyze'), {'image': SimpleUploadedFile('face.png', data, 'image/png')})
        r = self.client.get(reverse('result'))
        self.assertContains(r, 'Interface preview')
        self.assertContains(r, 'face.png')
        self.assertContains(r, hashlib.sha256(data).hexdigest())
        self.assertContains(r, 'EfficientNet-B4')

    def test_image_itself_is_not_stored_in_session(self):
        self.client.login(username='sara', password=PASSWORD)
        self.client.post(reverse('analyze'), {'image': SimpleUploadedFile('a.png', _png_bytes(), 'image/png')})
        self.assertNotIn(b'PNG'.decode(), str(self.client.session.get('last_upload')))
