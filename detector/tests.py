"""
detector/tests.py — Simple unit tests for DeepShield.

Run all tests with:
  python manage.py test detector

These tests check the core functionality without needing actual AI models.
They are intentionally simple and beginner-friendly.

What is tested:
  1.  Home page loads
  2.  Register form works
  3.  Login works
  4.  User cannot upload without login (redirect to login)
  5.  Valid registration creates user
  6.  Login with wrong password fails
  7.  Dashboard is accessible after login
  8.  SHA-256 hash generation works
  9.  Ensemble logic returns a result
  10. Model not_configured result is handled safely
  11. History page is accessible
  12. Upload page returns 200 for logged-in user
"""

import hashlib
import tempfile
import os

from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse

from ai_models.ensemble import calculate_ensemble_result, normalize_prediction


# ============================================================
# TEST 1-3: Page Access Tests
# ============================================================
class PageAccessTests(TestCase):
    """Test that public pages load correctly."""

    def setUp(self):
        """setUp() runs before every test method in this class."""
        self.client = Client()

    def test_home_page_loads(self):
        """Home page should return HTTP 200 (OK) without login."""
        response = self.client.get(reverse('home'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'DeepShield')

    def test_register_page_loads(self):
        """Register page should return HTTP 200."""
        response = self.client.get(reverse('register'))
        self.assertEqual(response.status_code, 200)

    def test_login_page_loads(self):
        """Login page should return HTTP 200."""
        response = self.client.get(reverse('login'))
        self.assertEqual(response.status_code, 200)


# ============================================================
# TEST 4: Upload Protection Test
# ============================================================
class AuthProtectionTests(TestCase):
    """Test that protected pages redirect unauthenticated users."""

    def setUp(self):
        self.client = Client()

    def test_upload_requires_login(self):
        """
        Visiting /upload/ without login should redirect to the login page.
        HTTP 302 = redirect.
        """
        response = self.client.get(reverse('upload'))
        # Should redirect (302) not show the page (200)
        self.assertEqual(response.status_code, 302)
        # The redirect should go to the login URL
        self.assertIn('/public/login/', response['Location'])

    def test_history_requires_login(self):
        """History page should redirect unauthenticated users."""
        response = self.client.get(reverse('history'))
        self.assertEqual(response.status_code, 302)


# ============================================================
# TEST 5-6: Registration and Login Tests
# ============================================================
class RegistrationLoginTests(TestCase):
    """Test user registration and login flows."""

    def setUp(self):
        self.client = Client()

    def test_valid_registration_creates_user(self):
        """
        Posting valid data to the register page should create a new user
        and redirect to the login page.
        """
        response = self.client.post(reverse('register'), {
            'username': 'testuser',
            'email': 'test@example.com',
            'password1': 'SecurePass123!',
            'password2': 'SecurePass123!',
            'accept_privacy': 'on',
        })
        # Should redirect after successful registration
        self.assertEqual(response.status_code, 302)
        # The user should now exist in the database
        self.assertTrue(User.objects.filter(username='testuser').exists())

    def test_invalid_registration_shows_errors(self):
        """Submitting passwords that don't match should fail."""
        response = self.client.post(reverse('register'), {
            'username': 'testuser2',
            'email': 'test2@example.com',
            'password1': 'SecurePass123!',
            'password2': 'WrongPassword!',  # Doesn't match
        })
        # Should stay on the register page (200, not redirect)
        self.assertEqual(response.status_code, 200)
        # User should NOT have been created
        self.assertFalse(User.objects.filter(username='testuser2').exists())

    def test_valid_login_works(self):
        """A user with correct credentials should be logged in successfully."""
        # First create a user
        User.objects.create_user(username='loginuser', password='TestPass999!')
        # Try to login
        response = self.client.post(reverse('login'), {
            'username': 'loginuser',
            'password': 'TestPass999!',
        })
        # Should redirect into the app after login
        self.assertEqual(response.status_code, 302)

    def test_wrong_password_fails(self):
        """Wrong password should stay on the login page and show an error."""
        User.objects.create_user(username='loginuser2', password='RightPass!')
        response = self.client.post(reverse('login'), {
            'username': 'loginuser2',
            'password': 'WrongPass!',
        })
        # Should stay on the login page
        self.assertEqual(response.status_code, 200)


# ============================================================
# TEST 7: Dashboard Access Test
# ============================================================
class DashboardTests(TestCase):
    """Test dashboard access after login."""

    def setUp(self):
        self.client = Client()
        # Create and log in a test user
        self.user = User.objects.create_user(username='dashuser', password='Pass123!')
        self.client.login(username='dashuser', password='Pass123!')

    def test_history_accessible_after_login(self):
        """Logged-in users should see the history page (HTTP 200)."""
        response = self.client.get(reverse('history'))
        self.assertEqual(response.status_code, 200)

    def test_upload_page_accessible_after_login(self):
        """Logged-in users should see the upload page (HTTP 200)."""
        response = self.client.get(reverse('upload'))
        self.assertEqual(response.status_code, 200)


# ============================================================
# TEST 8: SHA-256 Hash Test
# ============================================================
class SHA256Tests(TestCase):
    """Test that SHA-256 hash generation works correctly."""

    def test_sha256_of_file(self):
        """
        Write a test file to disk and verify SHA-256 hash is calculated correctly.
        We compare the result from our function to Python's built-in hashlib.
        """
        from ai_models.preprocessing import calculate_sha256

        # Create a temporary file with known content
        test_content = b"DeepShield test file content 12345"

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(test_content)
            tmp_path = tmp.name

        try:
            # Calculate hash using our function
            calculated_hash = calculate_sha256(tmp_path)

            # Calculate expected hash using Python's hashlib directly
            expected_hash = hashlib.sha256(test_content).hexdigest()

            self.assertEqual(calculated_hash, expected_hash)
            self.assertEqual(len(calculated_hash), 64)  # SHA-256 is always 64 hex chars

        finally:
            # Clean up the temporary file
            os.unlink(tmp_path)

    def test_sha256_different_content_different_hash(self):
        """Two files with different content must have different SHA-256 hashes."""
        from ai_models.preprocessing import calculate_sha256

        # Create two temp files with different content
        with tempfile.NamedTemporaryFile(delete=False) as f1:
            f1.write(b"Content A")
            path1 = f1.name

        with tempfile.NamedTemporaryFile(delete=False) as f2:
            f2.write(b"Content B")
            path2 = f2.name

        try:
            hash1 = calculate_sha256(path1)
            hash2 = calculate_sha256(path2)
            self.assertNotEqual(hash1, hash2)
        finally:
            os.unlink(path1)
            os.unlink(path2)


# ============================================================
# TEST 9-10: Ensemble Logic Tests
# ============================================================
class EnsembleTests(TestCase):
    """Test the ensemble decision logic."""

    def setUp(self):
        """Sample weights matching settings.MODEL_WEIGHTS."""
        self.weights = {
            "custom_model_1":    0.20,
            "custom_model_2":    0.20,
            "custom_model_3":    0.15,
            "pretrained_model_1": 0.15,
            "pretrained_model_2": 0.15,
            "pretrained_model_3": 0.15,
        }

    def test_all_not_configured_returns_uncertain(self):
        """
        If all models are not configured, the result should be UNCERTAIN
        with 0 executed models.
        """
        not_configured_results = [
            {"model_key": k, "prediction": "NOT_CONFIGURED",
             "confidence": 0.0, "fake_probability": 0.0, "status": "not_configured"}
            for k in self.weights.keys()
        ]
        result = calculate_ensemble_result(not_configured_results, self.weights)
        self.assertEqual(result['final_prediction'], 'UNCERTAIN')
        self.assertEqual(result['executed_models'], 0)

    def test_all_fake_returns_manipulated(self):
        """
        If all models say FAKE with high confidence, result should be LIKELY_MANIPULATED.
        """
        fake_results = [
            {"model_key": k, "prediction": "FAKE",
             "confidence": 0.95, "fake_probability": 0.95, "status": "success"}
            for k in self.weights.keys()
        ]
        result = calculate_ensemble_result(fake_results, self.weights)
        self.assertEqual(result['final_prediction'], 'LIKELY_MANIPULATED')
        self.assertEqual(result['executed_models'], 6)
        self.assertEqual(result['manipulation_votes'], 6)

    def test_all_real_returns_authentic(self):
        """
        If all models say REAL with high confidence, result should be LIKELY_AUTHENTIC.
        """
        real_results = [
            {"model_key": k, "prediction": "REAL",
             "confidence": 0.95, "fake_probability": 0.05, "status": "success"}
            for k in self.weights.keys()
        ]
        result = calculate_ensemble_result(real_results, self.weights)
        self.assertEqual(result['final_prediction'], 'LIKELY_AUTHENTIC')


# ============================================================
# TEST 11: normalize_prediction Tests
# ============================================================
class NormalizePredictionTests(TestCase):
    """Test the label normalisation function."""

    def test_fake_labels_normalised(self):
        """Various 'fake' labels should all normalise to FAKE."""
        for label in ('fake', 'FAKE', 'deepfake', 'DEEPFAKE', 'generated'):
            self.assertEqual(normalize_prediction(label), 'FAKE')

    def test_real_labels_normalised(self):
        """Various 'real' labels should normalise to REAL."""
        for label in ('real', 'REAL', 'genuine', 'authentic'):
            self.assertEqual(normalize_prediction(label), 'REAL')

    def test_tampered_labels_normalised(self):
        """Various 'tampered' labels should normalise to TAMPERED."""
        for label in ('tampered', 'TAMPERED', 'forged'):
            self.assertEqual(normalize_prediction(label), 'TAMPERED')

    def test_unknown_label_returns_uncertain(self):
        """An unknown label should return UNCERTAIN without crashing."""
        result = normalize_prediction('some_unknown_label_xyz')
        self.assertEqual(result, 'UNCERTAIN')
