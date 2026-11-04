"""Extra adversarial probes: cross-user access on every id-based page, state-changing GETs,
security headers, reflected/stored XSS, and per-user media isolation."""

from django.urls import reverse

from detector.models import UploadedImage
from detector.tests_security import TempMediaTestCase


class CrossUserAccessTests(TempMediaTestCase):
    def setUp(self):
        super().setUp()
        self.up = self.make_upload(self.alice)
        self.client.login(username='bob', password='Correct-Horse-9')

    def test_every_id_page_is_404_or_redirect_for_other_user(self):
        pk = self.up.pk
        for name in ('result', 'detail', 'report', 'processing', 'job_status'):
            r = self.client.get(reverse(name, args=[pk]))
            self.assertIn(r.status_code, (302, 403, 404), name)
            if r.status_code == 302:
                self.assertNotIn(str(pk), r['Location'].replace('login', ''), name)
        for name in ('cancel_job', 'delete_analysis'):
            r = self.client.post(reverse(name, args=[pk]))
            self.assertIn(r.status_code, (403, 404), name)
        self.assertTrue(UploadedImage.objects.filter(pk=pk).exists())

    def test_state_changing_endpoints_reject_get(self):
        for name, args in (('delete_analysis', [self.up.pk]), ('cancel_job', [self.up.pk]),
                           ('delete_account', []), ('logout', [])):
            r = self.client.get(reverse(name, args=args))
            self.assertIn(r.status_code, (302, 404, 405), name)
        self.assertTrue(UploadedImage.objects.filter(pk=self.up.pk).exists())

    def test_stored_files_live_under_owner_folder(self):
        self.assertTrue(self.up.image.name.startswith(f'users/user_{self.alice.id}/uploads/'))
        r = self.client.get('/media/' + self.up.image.name)
        self.assertIn(r.status_code, (403, 404))


class HeaderAndXssTests(TempMediaTestCase):
    def test_security_headers_present(self):
        r = self.client.get('/')
        self.assertEqual(r.headers.get('X-Content-Type-Options'), 'nosniff')
        csp = r.headers.get('Content-Security-Policy', '')
        self.assertIn("default-src", csp)
        self.assertNotIn("'unsafe-eval'", csp)
        self.assertIn(r.headers.get('X-Frame-Options', 'DENY').upper(), ('DENY', 'SAMEORIGIN'))
        self.assertTrue(r.headers.get('Referrer-Policy'))

    def test_filename_xss_is_escaped_on_history(self):
        payload = '<script>alert(1)</script>.png'
        up = self.make_upload(self.alice, name='x.png')
        UploadedImage.objects.filter(pk=up.pk).update(original_filename=payload)
        self.client.login(username='alice', password='Correct-Horse-9')
        for name in ('history',):
            html = self.client.get(reverse(name)).content.decode()
            self.assertNotIn('<script>alert(1)</script>', html, name)

    def test_login_next_reflection_is_escaped(self):
        html = self.client.get(reverse('login') + '?next="><script>alert(1)</script>').content.decode()
        self.assertNotIn('<script>alert(1)</script>', html)

    def test_no_user_enumeration_on_password_reset(self):
        a = self.client.post(reverse('password_reset'), {'email': 'a@example.com'})
        b = self.client.post(reverse('password_reset'), {'email': 'nobody@example.com'})
        self.assertEqual(a.status_code, b.status_code)
        self.assertEqual(a.get('Location'), b.get('Location'))
