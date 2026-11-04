"""
detector/forms.py — Django form definitions for DeepShield.

Forms handle:
  1. User input validation (checking data before saving it)
  2. Rendering HTML form fields in templates
  3. Showing error messages when input is wrong

Forms in this file:
  - UserRegistrationForm  → for creating a new account
  - ImageUploadForm       → for uploading an image for analysis
"""

import re
from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from detector.image_validation import validate_uploaded_image


# ============================================================
# FORM 1: User Registration
# ============================================================
class UserRegistrationForm(UserCreationForm):
    """
    Registration form for new users.

    Extends Django's built-in UserCreationForm which already handles:
      - Password confirmation (type it twice)
      - Password strength validation
      - Duplicate username checking

    We add:
      - Email field (not included by default)
      - Styling for Bootstrap 5
    """

    # Add email field — not included in UserCreationForm by default
    email = forms.EmailField(
        required=True,
        help_text='Required. Enter a valid email address.',
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your email address',
        })
    )

    accept_privacy = forms.BooleanField(
        required=True, label='I have read and accept the Privacy Notice.',
        error_messages={'required': 'You must accept the Privacy Notice to create an account.'},
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}))

    class Meta:
        # Use Django's built-in User model
        model = User
        # These fields will appear in the form
        fields = ('username', 'email', 'password1', 'password2')

    def __init__(self, *args, **kwargs):
        """Add Bootstrap CSS classes to all form fields."""
        super().__init__(*args, **kwargs)

        # Add Bootstrap 'form-control' class and placeholder to each field
        self.fields['username'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Choose a username',
        })
        self.fields['password1'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Create a password',
        })
        self.fields['password2'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Confirm your password',
        })

    def clean_email(self):
        """
        Extra validation: check that the email is not already registered.
        Django's clean_<fieldname> methods run during form validation.
        """
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("An account with this email address already exists.")
        return email


# ============================================================
# FORM 2: Image Upload
# ============================================================
class ImageUploadForm(forms.Form):
    """
    Upload form. Validation is based on file CONTENT (magic bytes, full decode,
    pixel limits) — see detector/image_validation.py. The client-supplied
    extension and Content-Type are never trusted.
    """

    image = forms.FileField(
        label='Select Image',
        help_text='JPEG or PNG. Maximum size: 10 MB.',
        widget=forms.ClearableFileInput(attrs={
            'class': 'form-control',
            'accept': '.jpg,.jpeg,.png,image/jpeg,image/png',  # UX hint only, not security
            'id': 'imageUpload',
        })
    )

    original_filename = ''   # sanitised display name, set by clean_image()

    @staticmethod
    def _display_name(name: str) -> str:
        """Sanitised name for DISPLAY only (never used as a filesystem path)."""
        base = re.split(r'[\\/]', name or '')[-1]
        base = re.sub(r'[\x00-\x1f\x7f]', '', base).strip()
        return (base or 'image')[:255]

    def clean_image(self):
        image = self.cleaned_data.get('image')
        if not image:
            raise forms.ValidationError('Please select an image file to upload.')
        info = validate_uploaded_image(image)          # raises ValidationError
        self.original_filename = self._display_name(image.name)
        # Server-side name: extension comes from the verified content type.
        image.name = f'upload.{info.extension}'
        return image
