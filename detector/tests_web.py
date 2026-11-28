"""Phase 8/9 web tests: consent enforcement, privacy page, web deletion, account deletion, CSP."""

import os
import re

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from detector import storage
from detector.models import UploadedImage
from detector.tests_api import PW, ApiTestCase
from detector.tests_security import make_image_bytes


class ThemeToggleTests(ApiTestCase):
    def test_guests_are_locked_to_dark_and_have_no_toggle(self):
        html = self.client.get(reverse('home')).content.decode()
        self.assertNotIn('id="themeToggle"', html)
        self.assertIn('data-lock-theme="dark"', html)

    def test_toggle_is_shown_to_signed_in_users_outside_the_collapsible_nav(self):
        self.client.force_login(self.alice)
        html = self.client.get(reverse('upload')).content.decode()
        toggle = '<button type="button" id="themeToggle"'
        nav = '<div class="collapse navbar-collapse order-lg-1" id="navbarNav">'
        self.assertEqual(html.count('id="themeToggle"'), 1)
        self.assertLess(html.index(toggle), html.index(nav))
        self.assertNotIn('data-lock-theme', html)

    def test_auth_forms_retain_their_card_and_submission_controls(self):
        login = self.client.get(reverse('login')).content.decode()
        register = self.client.get(reverse('register')).content.decode()
        self.assertIn('class="ds-auth-card"', login)
        self.assertIn('id="loginForm"', login)
        self.assertIn('class="ds-auth-card"', register)
        self.assertIn('id="registerForm"', register)

    def test_upload_page_keeps_file_and_submit_controls_in_landscape_layout(self):
        self.client.force_login(self.alice)
        html = self.client.get(reverse('upload')).content.decode()
        self.assertIn('ds-main--upload', html)
        self.assertIn('class="ds-upload-form"', html)
        self.assertIn('id="dropzone"', html)
        self.assertIn('id="imageUpload"', html)
        self.assertIn('id="submitBtn"', html)
        self.assertIn('id="scanningState"', html)
        self.assertIn('What happens after you upload?', html)


class PublicNavigationTests(ApiTestCase):
    def test_public_nav_matches_landing_section_order(self):
        html = self.client.get(reverse('home')).content.decode()
        nav = html[:html.index('</nav>')]
        section_links = (
            f'href="{reverse("home")}#how-it-works"',
            f'href="{reverse("home")}#models"',
            f'href="{reverse("home")}#algorithms"',
        )
        positions = [nav.index(link) for link in section_links]
        self.assertEqual(positions, sorted(positions))

    def test_homepage_redesign_preserves_copy_routes_and_face_scan(self):
        html = self.client.get(reverse('home')).content.decode()
        for content in (
            'Detect Deepfakes.',
            'Understand the Evidence.',
            'AI-Powered Image Forensics',
            'available model outputs with forensic checks',
            'not proof that an image is authentic or manipulated.',
            'not a probability.',
            'How DeepShield Works',
            'AI Models',
            'Forensic Algorithms',
            'System Features',
            'Academic Research Disclaimer',
            'Up to 5 AI Models',
            'Model Scores',
            'SHA-256 Hash',
            'Explanation Maps',
            'Ensemble Decision',
            'Analysis History',
        ):
            self.assertIn(content, html)
        self.assertRegex(html, r'Upload a JPG or PNG portrait photo \(up to [\d.]+ MB\)\.')
        self.assertIn('class="ds-home"', html)
        self.assertIn('static/images/home-face.png', html)
        self.assertIn('ds-home-face-mesh', html)
        self.assertIn('Illustrative visual only — this image has not been analyzed.', html)
        self.assertNotIn('FACE-SCAN VISUALIZATION', html)
        self.assertRegex(html, r'class="[^"]*\bds-home-visual-note\b[^"]*"')
        self.assertIn('Analysis features orbiting the illustrative face scan', html)
        orbit_groups = re.findall(r'<ul class="ds-home-orbit [^"]+"[^>]*>(.*?)</ul>', html, re.S)
        self.assertEqual(len(orbit_groups), 3)
        self.assertEqual(
            [group.count('class="ds-hero-icon-item ds-home-capability"') for group in orbit_groups],
            [2, 2, 2],
        )
        self.assertEqual(html.count('class="ds-hero-icon-item ds-home-capability"'), 6)
        self.assertEqual(html.count('class="ds-home-capability-label"'), 6)
        self.assertNotIn('dsBinaryFaceCanvas', html)
        self.assertIn(f'href="{reverse("register")}"', html)
        self.assertIn(f'href="{reverse("login")}"', html)

    def test_register_nav_does_not_link_back_to_itself(self):
        html = self.client.get(reverse('register')).content.decode()
        nav = html[:html.index('</nav>')]
        self.assertNotIn('class="ds-btn-register"', nav)


class ConsentTests(ApiTestCase):
    def test_registration_requires_privacy_acceptance(self):
        data = {'username': 'carol', 'email': 'c@example.com', 'password1': PW + 'x1!', 'password2': PW + 'x1!'}
        r = self.client.post(reverse('register'), data)
        self.assertEqual(r.status_code, 200)
        self.assertFalse(User.objects.filter(username='carol').exists())
        r = self.client.post(reverse('register'), {**data, 'accept_privacy': 'on'})
        self.assertTrue(User.objects.filter(username='carol').exists(), r.content[:300])

    def test_privacy_page_is_public_and_states_retention(self):
        with override_settings(DELETE_ANALYSIS_AFTER_DAYS=14, AUDIT_LOG_RETENTION_DAYS=1,
                               MAX_UPLOAD_SIZE=512 * 1024):
            r = self.client.get(reverse('privacy'))
        self.assertContains(r, 'operator-run cleanup process is configured to delete analyses and their files after <strong>14 days</strong>')
        self.assertContains(r, 'up to 512.0')
        self.assertContains(r, 'kept for 1 day.')
        self.assertContains(r, 'raw, uncalibrated indicators—not probabilities')
        self.assertContains(r, 'contact the organization that operates this site')

        with override_settings(DELETE_ANALYSIS_AFTER_DAYS=1, AUDIT_LOG_RETENTION_DAYS=2,
                               MAX_UPLOAD_SIZE=10 * 1024 * 1024):
            singular = self.client.get(reverse('privacy'))
        self.assertContains(singular, 'operator-run cleanup process is configured to delete analyses and their files after <strong>1 day</strong>')
        self.assertContains(singular, 'up to 10.0')
        self.assertContains(singular, 'kept for 2 days.')

        with override_settings(DELETE_ANALYSIS_AFTER_DAYS=0):
            self.assertContains(self.client.get(reverse('privacy')), 'kept until you delete them')


class WebDeletionTests(ApiTestCase):
    def test_owner_can_delete_via_web_and_files_are_removed(self):
        rec = self.two_face_analysis(self.alice)
        from django.conf import settings
        files = storage.artifact_names(rec) | {rec.image.name}
        self.client.force_login(self.alice)
        r = self.client.post(reverse('delete_analysis', args=[rec.pk]))
        self.assertRedirects(r, reverse('history'))
        self.assertFalse(UploadedImage.objects.filter(pk=rec.pk).exists())
        for n in files:
            self.assertFalse(os.path.exists(os.path.join(str(settings.MEDIA_ROOT), n)), n)

    def test_other_user_cannot_delete_and_get_is_not_allowed(self):
        rec = self.two_face_analysis(self.alice)
        self.client.force_login(self.bob)
        self.assertEqual(self.client.post(reverse('delete_analysis', args=[rec.pk])).status_code, 404)
        self.client.force_login(self.alice)
        self.assertEqual(self.client.get(reverse('delete_analysis', args=[rec.pk])).status_code, 405)
        self.assertTrue(UploadedImage.objects.filter(pk=rec.pk).exists())

    def test_result_page_offers_a_csrf_protected_delete_form(self):
        rec = self.two_face_analysis(self.alice)
        self.client.force_login(self.alice)
        html = self.client.get(reverse('result', args=[rec.pk])).content.decode()
        self.assertIn(reverse('delete_analysis', args=[rec.pk]), html)
        self.assertIn('csrfmiddlewaretoken', html)

    def test_account_deletion_needs_password_and_typed_confirmation(self):
        self.quick_upload(self.alice)
        self.client.force_login(self.alice)
        url = reverse('delete_account')
        self.client.post(url, {'password': 'wrong', 'confirm': 'DELETE'})
        self.client.post(url, {'password': PW, 'confirm': 'nope'})
        self.assertTrue(User.objects.filter(username='alice').exists())
        r = self.client.post(url, {'password': PW, 'confirm': 'DELETE'})
        self.assertRedirects(r, reverse('home'), fetch_redirect_response=False)
        self.assertFalse(User.objects.filter(username='alice').exists())
        self.assertEqual(UploadedImage.objects.count(), 0)
        self.assertEqual(self.client.get(reverse('account')).status_code, 302)    # logged out

    def test_account_page_requires_login(self):
        self.assertEqual(self.client.get(reverse('account')).status_code, 302)


class CSPTests(ApiTestCase):
    def nonce_of(self, resp):
        return re.search(r"'nonce-([^']+)'", resp['Content-Security-Policy']).group(1)

    def test_landing_page_is_public_with_nonce_on_every_script(self):
        r = self.client.get(reverse('home'))
        self.assertEqual(r.status_code, 200)
        nonce = self.nonce_of(r)
        html = r.content.decode()
        self.assertTrue(all(f'nonce="{nonce}"' in t for t in re.findall(r'<script\b[^>]*>', html)))
        for anchor in ('id="models"', 'id="algorithms"', 'id="how-it-works"'):
            self.assertIn(anchor, html)
        self.assertNotIn('href="/how-it-works/"', html[:html.index('</nav>')])   # navbar no longer carries public info links

    def test_signed_in_users_skip_the_landing_page(self):
        self.client.force_login(self.alice)
        self.assertRedirects(self.client.get(reverse('home')), reverse('history'))
        nav = self.client.get(reverse('history')).content.decode()
        nav = nav[:nav.index('</nav>')]
        for gone in ('/public/#models', '/public/#algorithms', '/dashboard/'):
            self.assertNotIn(f'href="{gone}"', nav)
        for kept in ('/auth/upload/', '/auth/analysis/'):
            self.assertIn(f'href="{kept}"', nav)

    def test_analysis_routes_use_canonical_analysis_urls(self):
        rec = self.two_face_analysis(self.alice)

        self.assertEqual(reverse('history'), '/auth/analysis/')
        self.assertEqual(
            reverse('result', args=[rec.pk]),
            f'/auth/analysis/{rec.pk}/'
        )

        self.client.force_login(self.alice)

        response = self.client.get(reverse('history'))
        self.assertContains(response, f'/auth/detail/{rec.pk}/')

        self.assertEqual(
            self.client.get(reverse('result', args=[rec.pk])).status_code,
            200
        )

        self.assertRedirects(
            self.client.get(f'/auth/result/{rec.pk}/'),
            reverse('result', args=[rec.pk]),
            status_code=301
        )
    def test_pages_get_policy_with_matching_nonce_on_every_script(self):
        self.client.force_login(self.alice)
        rec = self.two_face_analysis(self.alice)
        for url in (reverse('upload'), reverse('result', args=[rec.pk]), reverse('account'),
                    reverse('privacy'), reverse('report', args=[rec.pk]), reverse('history')):
            r = self.client.get(url)
            self.assertEqual(r.status_code, 200, url)
            nonce = self.nonce_of(r)
            html = r.content.decode()
            scripts = re.findall(r'<script\b[^>]*>', html)
            self.assertTrue(all(f'nonce="{nonce}"' in s for s in scripts), (url, scripts))
            self.assertNotRegex(html, r'\son(click|load|error|submit|change)\s*=', url)
            self.assertIn("object-src 'none'", r['Content-Security-Policy'])
            self.assertNotIn("script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'", r['Content-Security-Policy'])

    def test_nonce_differs_per_request(self):
        a = self.nonce_of(self.client.get(reverse('home')))
        b = self.nonce_of(self.client.get(reverse('home')))
        self.assertNotEqual(a, b)

    def test_api_responses_get_deny_all_policy(self):
        r = self.client.get('/api/v1/system/health/')
        self.assertEqual(r['Content-Security-Policy'], "default-src 'none'; frame-ancestors 'none'; base-uri 'none'")

    @override_settings(CSP_REPORT_ONLY=True)
    def test_report_only_mode(self):
        r = self.client.get(reverse('home'))
        self.assertIn('Content-Security-Policy-Report-Only', r)
        self.assertNotIn('Content-Security-Policy', r)

    @override_settings(CSP_IMG_SOURCES=['https://s3.example.com'])
    def test_extra_image_hosts_for_object_storage(self):
        self.assertIn('img-src \'self\' data: https://s3.example.com', self.client.get(reverse('home'))['Content-Security-Policy'])


class HonestPresentationTests(ApiTestCase):
    """The UI must not present uncalibrated scores as probabilities or confidence percentages."""

    def page(self, name, rec):
        self.client.force_login(self.alice)
        from django.urls import reverse
        return self.client.get(reverse(name, args=[rec.pk])).content.decode()

    def test_result_detail_report_have_no_confidence_percentages(self):
        rec = self.two_face_analysis(self.alice)
        for name in ('result', 'detail', 'report'):
            html = self.page(name, rec)
            self.assertNotIn('AI Confidence', html, name)
            self.assertNotRegex(html, r'Confidence</small>', name)
            self.assertIn('uncalibrated', html.lower(), name)

    def test_result_states_score_is_not_a_probability_and_lists_each_face(self):
        rec = self.two_face_analysis(self.alice)
        html = self.page('result', rec)
        self.assertIn('Not a probability', html)
        self.assertIn('System assessment', html)
        self.assertIn('Model coverage:', html)
        self.assertIn(f'{rec.analysis.executed_models} available', html)
        self.assertIn(f'{rec.analysis.unavailable_models} unavailable', html)
        self.assertIn('Model votes:', html)
        self.assertIn('headline summarizes the available model outputs', html)
        self.assertIn('Forensic checks are separate supporting observations', html)
        self.assertIn('raw, uncalibrated indicators—not probabilities', html)
        self.assertIn('Neither the assessment nor these indicators prove', html)
        self.assertNotIn('Looks fake', html)
        self.assertNotIn('Looks real', html)
        self.assertIn('Face 1', html)
        self.assertIn('Face 2', html)
        self.assertIn('not proof', html.lower())

    def test_public_and_upload_copy_uses_current_limit_and_cautious_score_language(self):
        with override_settings(MAX_UPLOAD_SIZE=15 * 1024 * 1024):
            home = self.client.get(reverse('home')).content.decode()
            self.client.force_login(self.alice)
            upload = self.client.get(reverse('upload')).content.decode()

        self.assertIn('up to 15 MB', home)
        self.assertIn('Maximum: 15 MB', upload)
        self.assertIn('raw, uncalibrated score—not a probability', home)
        self.assertIn('not proof that an image is authentic or manipulated', home)
        self.assertNotIn('confidence score', home.lower())
        self.assertNotIn('probabilistic', home.lower())
        self.assertNotIn('Maximum: 10 MB', upload)

    def test_registration_does_not_promise_unlimited_analyses(self):
        html = self.client.get(reverse('register')).content.decode()
        self.assertIn('Personal analysis history &amp; verifiable records', html)
        self.assertNotIn('Unlimited image analysis', html)

    def test_detail_report_and_history_use_cautious_assessment_language(self):
        rec = self.two_face_analysis(self.alice)
        states = (
            ('LIKELY_MANIPULATED', 'Possible manipulation indicators', 'Possible manipulation'),
            ('LIKELY_AUTHENTIC', 'No strong manipulation indicators', 'No strong indicators'),
        )
        for prediction, assessment_label, history_label in states:
            with self.subTest(prediction=prediction):
                rec.analysis.final_prediction = prediction
                rec.analysis.save(update_fields=['final_prediction'])
                detail = self.page('detail', rec)
                report = self.page('report', rec)
                history = self.client.get(reverse('history')).content.decode()

                for html in (detail, report):
                    self.assertIn('System assessment', html)
                    self.assertIn(assessment_label, html)
                    self.assertIn('not proof that an image is authentic or manipulated', html)

                self.assertIn(f'>{history_label}</span>', history)
                self.assertNotIn('>Manipulated</span>', history)
                self.assertNotIn('>Authentic</span>', history)

    def test_result_omits_disabled_models_from_expected_count(self):
        rec = self.two_face_analysis(self.alice)
        analysis = rec.analysis
        pipeline_info = analysis.pipeline_info
        disabled_key = next(
            model['key'] for model in pipeline_info['models'] if model['key'] == 'sdxl_detector')
        next(model for model in pipeline_info['models'] if model['key'] == disabled_key)['status'] = 'DISABLED'
        analysis.pipeline_info = pipeline_info
        analysis.save(update_fields=['pipeline_info'])
        for face in analysis.faces.all():
            face.ensemble['models_expected'] = 5
            face.save(update_fields=['ensemble'])
        disabled_result = analysis.model_results.get(model_key=disabled_key, face_index=0)
        disabled_result.details['status'] = 'DISABLED'
        disabled_result.save(update_fields=['details'])

        html = self.page('result', rec)
        self.assertIn('4 expected, 3 ran', html)
        self.assertIn('3 of 4 models ran', html)
        self.assertIn('Disabled in configuration', html)
        self.assertNotIn('3 of 5 models contributed', html)

    def test_history_shows_no_percent_confidence(self):
        self.two_face_analysis(self.alice)
        self.client.force_login(self.alice)
        from django.urls import reverse
        html = self.client.get(reverse('history')).content.decode()
        self.assertNotRegex(html, r'\d+(\.\d+)?%</(strong|small)>')


class UnavailableIndicatorTests(ApiTestCase):
    def test_unavailable_forensic_module_shows_no_score(self):
        rec = self.two_face_analysis(self.alice)           # PNG upload -> ELA is not applicable
        from detector.models import ForensicAnalysis
        f = ForensicAnalysis.objects.get(analysis__uploaded_image=rec)
        self.assertEqual(f.ela_level, 'UNAVAILABLE')
        self.client.force_login(self.alice)
        from django.urls import reverse
        html = self.client.get(reverse('result', args=[rec.pk])).content.decode()
        i = html.index('id="fc-ela"')
        ela = html[i:html.index('id="fc-freq"', i)]
        self.assertNotIn('/100', ela)
        self.assertIn('not applicable', ela.lower())
        self.assertIn('not a probability', html)


class PublicPagesTests(ApiTestCase):
    """Public pages live under /public/ and are always dark."""

    def test_public_pages_return_200(self):
        for name in ('home', 'login', 'register', 'privacy', 'limitations'):
            r = self.client.get(reverse(name))
            self.assertEqual(r.status_code, 200, name)
            self.assertTrue(reverse(name).startswith('/public/'), name)
            self.assertIn('data-lock-theme="dark"', r.content.decode(), name)

    def test_root_redirects_to_public_and_app_pages_need_login(self):
        self.assertRedirects(self.client.get('/'), '/public/', fetch_redirect_response=False)
        r = self.client.get(reverse('upload'))
        self.assertEqual(r.status_code, 302)
        self.assertTrue(reverse('upload').startswith('/auth/'))

    def test_register_logs_the_user_in_and_opens_the_app(self):
        data = {'username': 'dave', 'email': 'd@example.com', 'password1': PW + 'x1!', 'password2': PW + 'x1!',
                'accept_privacy': 'on'}
        r = self.client.post(reverse('register'), data)
        self.assertRedirects(r, reverse('history'), fetch_redirect_response=False)
        self.assertEqual(self.client.get(reverse('history')).status_code, 200)

    def test_upload_page_has_no_consent_checkbox(self):
        self.client.force_login(self.alice)
        self.assertNotIn('name="consent"', self.client.get(reverse('upload')).content.decode())

    def test_top_level_health_and_readiness(self):
        r_health = self.client.get('/health/')
        self.assertEqual(r_health.status_code, 200)
        self.assertEqual(r_health.json()['data']['status'], 'ok')

        r_ready = self.client.get('/ready/')
        self.assertEqual(r_ready.status_code, 200)
        self.assertEqual(r_ready.json()['data']['status'], 'ready')
