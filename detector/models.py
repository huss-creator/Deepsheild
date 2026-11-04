"""
detector/models.py — Database table definitions for DeepShield.

Django ORM (Object Relational Mapper) lets us define database tables
as Python classes. Each class = one table. Each attribute = one column.

Tables in this file:
  1. UploadedImage  — stores details about each uploaded image
  2. Analysis       — stores the combined analysis result for an image
  3. ModelResult    — stores the individual result from each AI model

The built-in Django User model handles user accounts.
We do not need to create a custom User table.
"""

import os
import uuid
from django.conf import settings
from django.db import models
from django.contrib.auth.models import User  # Built-in Django user model


# ============================================================
# HELPER: Generate a unique upload path for each image
# ============================================================
def user_media_prefix(user_id, *parts):
    """Per-user private folder: users/user_<id>/<parts...> (relative to MEDIA_ROOT, forward slashes).

    The numeric user id is stable and non-guessable-by-name; usernames/emails never touch the disk.
    """
    return '/'.join(['users', f'user_{int(user_id)}', *parts])


def upload_to_path(instance, filename):
    """Original uploads -> users/user_<id>/uploads/<uuid><ext> (server-generated name)."""
    ext = os.path.splitext(filename)[1].lower()
    return user_media_prefix(instance.user_id, 'uploads', f"{uuid.uuid4()}{ext}")


def face_upload_path(instance, filename):
    """Cropped faces -> users/user_<id>/faces/face_<uuid><ext>."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ('.jpg', '.jpeg', '.png'):
        ext = '.bin'
    return user_media_prefix(instance.user_id, 'faces', f"face_{uuid.uuid4()}{ext}")


# ============================================================
# MODEL 1: UploadedImage
# ============================================================
class UploadedImage(models.Model):
    """
    Stores all information about a single uploaded image.

    When a user uploads an image, Django:
      1. Saves the file to media/uploads/
      2. Creates one UploadedImage row in the database
      3. Stores metadata like filename, size, and SHA-256 hash
    """

    # PROCESSING STATUS CHOICES
    # These are the possible states an image can be in during processing.
    STATUS_PENDING = 'pending'
    STATUS_PROCESSING = 'processing'
    STATUS_COMPLETED = 'completed'
    STATUS_FAILED = 'failed'
    STATUS_NO_FACE = 'no_face'
    STATUS_CANCELLED = 'cancelled'

    PROCESSING_STATUS_CHOICES = [
        (STATUS_PENDING,    'Pending'),
        (STATUS_PROCESSING, 'Processing'),
        (STATUS_COMPLETED,  'Completed'),
        (STATUS_FAILED,     'Failed'),
        (STATUS_NO_FACE,    'No Face Detected'),
        (STATUS_CANCELLED,  'Cancelled'),
    ]

    # Link to the user who uploaded this image.
    # on_delete=CASCADE means: if user is deleted, their uploads are also deleted.
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='uploaded_images')

    # The actual image file. Saved to media/uploads/ folder.
    image = models.ImageField(upload_to=upload_to_path)

    # We save the original filename for display purposes only.
    # We do not use it as the actual filename (that would be a security risk).
    original_filename = models.CharField(max_length=255)

    # SHA-256 hash of the image file.
    # Used to detect exact duplicate files and verify file integrity.
    # This does NOT detect if an image is fake — it's just a file fingerprint.
    sha256_hash = models.CharField(max_length=64, blank=True)

    # File size in bytes
    file_size = models.PositiveIntegerField(default=0)

    # When the image was uploaded
    # Public identifier used in API URLs (never expose sequential integer IDs externally)
    public_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)

    uploaded_at = models.DateTimeField(auto_now_add=True)

    # Was a face found in this image?
    face_detected = models.BooleanField(default=False)

    # How many faces were found
    face_count = models.IntegerField(default=0)

    # The cropped face image (saved separately after OpenCV processing)
    # blank=True, null=True means it's optional — may not exist if no face found
    cropped_face = models.ImageField(upload_to=face_upload_path, blank=True, null=True)

    # Current processing state
    processing_status = models.CharField(
        max_length=20,
        choices=PROCESSING_STATUS_CHOICES,
        default=STATUS_PENDING
    )

    # Store any error message if processing fails
    error_message = models.TextField(blank=True, default='')

    class Meta:
        # Sort by newest first when querying
        ordering = ['-uploaded_at']
        indexes = [
            models.Index(fields=['user', '-uploaded_at'], name='img_user_created_idx'),
            models.Index(fields=['processing_status'], name='img_status_idx'),
        ]
        verbose_name = 'Uploaded Image'
        verbose_name_plural = 'Uploaded Images'

    def __str__(self):
        # No user-supplied filename here: admin titles/breadcrumbs would expose it.
        return f"{self.user.username} — {str(self.public_id)[:8]} ({self.uploaded_at.strftime('%Y-%m-%d %H:%M')})"

    def get_file_size_display(self):
        """Returns file size in a human-readable format like '2.3 MB'."""
        size = self.file_size
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        else:
            return f"{size / (1024 * 1024):.2f} MB"


# ============================================================
# MODEL 2: Analysis
# ============================================================
class Analysis(models.Model):
    """
    Stores the final ensemble analysis result for an uploaded image.

    One UploadedImage has one Analysis.
    The Analysis record is created after all models have run.
    It stores the combined/weighted final prediction.
    """

    # FINAL PREDICTION CHOICES
    PREDICTION_AUTHENTIC = 'LIKELY_AUTHENTIC'
    PREDICTION_MANIPULATED = 'LIKELY_MANIPULATED'
    PREDICTION_UNCERTAIN = 'UNCERTAIN'
    PREDICTION_NO_FACE = 'NO_FACE'

    FINAL_PREDICTION_CHOICES = [
        (PREDICTION_AUTHENTIC,   'Likely Authentic'),
        (PREDICTION_MANIPULATED, 'Likely Manipulated'),
        (PREDICTION_UNCERTAIN,   'Uncertain'),
        (PREDICTION_NO_FACE,     'No Face Detected'),
    ]

    # One-to-one link: each image has exactly one analysis result
    uploaded_image = models.OneToOneField(
        UploadedImage,
        on_delete=models.CASCADE,
        related_name='analysis'
    )

    # The final combined prediction
    final_prediction = models.CharField(
        max_length=30,
        choices=FINAL_PREDICTION_CHOICES,
        default=PREDICTION_UNCERTAIN
    )

    # Combined confidence score (0 to 100, stored as integer percentage)
    overall_confidence = models.FloatField(default=0.0)

    # Count of models that voted for authentic (REAL/ORIGINAL)
    authentic_votes = models.IntegerField(default=0)

    # Count of models that voted for manipulation (FAKE/TAMPERED)
    manipulation_votes = models.IntegerField(default=0)

    # Total models that successfully ran (not counting not_configured or error)
    executed_models = models.IntegerField(default=0)

    # Models that were not available (not_configured + error)
    unavailable_models = models.IntegerField(default=0)

    # Short human-readable explanation of how the result was reached
    explanation = models.TextField(blank=True, default='')

    # Full pipeline record: detector config, per-face bbox/quality, ensemble method and
    # version, model health, timings. Normalised into tables in Phase 6.
    pipeline_info = models.JSONField(default=dict, blank=True)

    # When this analysis was created
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Analysis'
        verbose_name_plural = 'Analyses'

    def __str__(self):
        return f"Analysis for upload #{self.uploaded_image_id} — {self.final_prediction}"

    def get_confidence_display(self):
        """Returns confidence as a clean percentage string like '82%'."""
        return f"{self.overall_confidence:.1f}%"

    def get_prediction_label(self):
        """Returns a user-friendly label for the final prediction."""
        labels = {
            self.PREDICTION_AUTHENTIC:   'Likely Authentic',
            self.PREDICTION_MANIPULATED: 'Likely Manipulated',
            self.PREDICTION_UNCERTAIN:   'Uncertain',
            self.PREDICTION_NO_FACE:     'No Face Detected',
        }
        return labels.get(self.final_prediction, 'Unknown')

    def primary_model_results(self):
        """Model results for the representative face (the one that decided the image-level
        result). The other faces' results stay available via model_results."""
        rep = (self.pipeline_info or {}).get('image_level', {}).get('representative_face') or 0
        return self.model_results.filter(face_index=rep).order_by('model_key')

    def get_prediction_badge_class(self):
        """Returns a Bootstrap badge CSS class based on the prediction."""
        classes = {
            self.PREDICTION_AUTHENTIC:   'badge-authentic',
            self.PREDICTION_MANIPULATED: 'badge-manipulated',
            self.PREDICTION_UNCERTAIN:   'badge-uncertain',
            self.PREDICTION_NO_FACE:     'badge-uncertain',
        }
        return classes.get(self.final_prediction, 'badge-secondary')


# ============================================================
# MODEL 3: ModelResult
# ============================================================
class ModelResult(models.Model):
    """
    Stores the result from a single AI model for one analysis.

    One Analysis has many ModelResults (one per AI model).
    There will be up to 6 ModelResult rows per analysis.

    STATUS:
      success        — model ran and returned a prediction
      not_configured — model file not yet added
      error          — model ran but threw an exception
    """

    # MODEL TYPE CHOICES
    TYPE_CUSTOM = 'custom'
    TYPE_PRETRAINED = 'pretrained'
    TYPE_TAMPERING = 'tampering'

    MODEL_TYPE_CHOICES = [
        (TYPE_CUSTOM,     'Custom Trained'),
        (TYPE_PRETRAINED, 'Pretrained / Transfer Learning'),
        (TYPE_TAMPERING,  'Tampering Detector'),
    ]

    # STATUS CHOICES
    STATUS_SUCCESS        = 'success'
    STATUS_NOT_CONFIGURED = 'not_configured'
    STATUS_ERROR          = 'error'

    STATUS_CHOICES = [
        (STATUS_SUCCESS,        'Success'),
        (STATUS_NOT_CONFIGURED, 'Not Configured'),
        (STATUS_ERROR,          'Error'),
    ]

    # Link to the parent analysis
    analysis = models.ForeignKey(
        Analysis,
        on_delete=models.CASCADE,
        related_name='model_results'  # lets us do: analysis.model_results.all()
    )

    # Human-readable name of the model
    model_name = models.CharField(max_length=100)

    # Internal key used to match this result to settings (e.g. 'custom_model_1')
    model_key = models.CharField(max_length=50, default='')

    # Type of model
    model_type = models.CharField(max_length=20, blank=True, default='')   # architecture family

    # Normalised prediction label: REAL, FAKE, ORIGINAL, TAMPERED, UNCERTAIN, etc.
    prediction = models.CharField(max_length=30, default='NOT_CONFIGURED')

    # Confidence score from 0.0 to 1.0 (e.g. 0.87 means 87%)
    confidence = models.FloatField(default=0.0)

    # Probability of fake/manipulation (main metric used in ensemble)
    # Range: 0.0 (definitely real) to 1.0 (definitely fake)
    fake_probability = models.FloatField(default=0.0)

    # Raw model status
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_NOT_CONFIGURED
    )

    # Optional message (error details, or info message for not_configured)
    message = models.TextField(blank=True, default='')

    # Path to Grad-CAM heatmap image (if generated), relative to media root
    gradcam_image = models.ImageField(upload_to='heatmaps/', blank=True, null=True)

    # Whether Grad-CAM was generated for this model
    has_gradcam = models.BooleanField(default=False)

    model_version = models.ForeignKey('ModelVersion', null=True, blank=True, on_delete=models.SET_NULL,
                                      related_name='results')

    # 0-based index of the face this result belongs to
    face_index = models.PositiveSmallIntegerField(default=0)

    # Model version, weights checksum, preprocessing/threshold version, score type,
    # inference time, explanation method/target/layer (reproducibility)
    details = models.JSONField(default=dict, blank=True)

    # When this individual model result was saved
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['model_key']
        verbose_name = 'Model Result'
        verbose_name_plural = 'Model Results'
        constraints = [
            models.UniqueConstraint(fields=['analysis', 'face_index', 'model_key'],
                                    name='uniq_result_per_face_model'),
            models.CheckConstraint(check=models.Q(fake_probability__gte=0, fake_probability__lte=1),
                                   name='result_score_between_0_and_1'),
        ]
        indexes = [models.Index(fields=['analysis', 'face_index'], name='result_face_idx')]

    def __str__(self):
        return f"{self.model_name}: {self.prediction} ({self.confidence:.0%})"

    def get_confidence_percent(self):
        """Returns confidence as an integer percentage, e.g. 87."""
        return int(self.confidence * 100)

    def get_fake_probability_percent(self):
        """Returns fake probability as an integer percentage."""
        return int(self.fake_probability * 100)

    def get_badge_class(self):
        """Returns a CSS class name based on the prediction for badge colouring."""
        if self.prediction in ('FAKE', 'TAMPERED'):
            return 'badge-manipulated'
        elif self.prediction in ('REAL', 'ORIGINAL'):
            return 'badge-authentic'
        elif self.status == self.STATUS_NOT_CONFIGURED:
            return 'badge-not-configured'
        elif self.status == self.STATUS_ERROR:
            return 'badge-error'
        else:
            return 'badge-uncertain'


# ============================================================
# MODEL 4: ForensicAnalysis
# ============================================================
class ForensicAnalysis(models.Model):
    """
    Stores the results of all forensic analysis modules for one analysis.

    One Analysis has one ForensicAnalysis record.
    Created during the upload pipeline AFTER the AI models have run.

    Forensic Modules:
        1. ELA (Error Level Analysis)      — JPEG recompression artifacts
        2. Frequency Analysis              — FFT frequency-domain patterns
        3. Noise Analysis                  — Noise residual and consistency
        4. Metadata Extraction             — EXIF metadata and software detection

    Evidence scores (0-100) represent how strongly each module's
    findings suggest possible manipulation. These are NOT accuracy scores.
    They are evidence weight scores that feed into the overall risk level.
    """

    # Level choices used for ELA, frequency, and noise levels
    LEVEL_UNAVAILABLE = 'UNAVAILABLE'
    LEVEL_LOW         = 'LOW'
    LEVEL_MEDIUM      = 'MEDIUM'
    LEVEL_HIGH        = 'HIGH'

    LEVEL_CHOICES = [
        (LEVEL_UNAVAILABLE, 'Unavailable'),
        (LEVEL_LOW,         'Low'),
        (LEVEL_MEDIUM,      'Medium'),
        (LEVEL_HIGH,        'High'),
    ]

    # Risk level choices for the overall evidence-based risk
    RISK_LOW       = 'LOW'
    RISK_MEDIUM    = 'MEDIUM'
    RISK_HIGH      = 'HIGH'
    RISK_VERY_HIGH = 'VERY_HIGH'

    RISK_CHOICES = [
        (RISK_LOW,       'Low Risk'),
        (RISK_MEDIUM,    'Medium Risk'),
        (RISK_HIGH,      'High Risk'),
        (RISK_VERY_HIGH, 'Very High Risk'),
    ]

    # Link to parent analysis (OneToOne: one analysis → one forensic result)
    analysis = models.OneToOneField(
        Analysis,
        on_delete=models.CASCADE,
        related_name='forensic_analysis'
    )

    # ---- ELA Results ----
    ela_score          = models.FloatField(default=0.0)
    ela_level          = models.CharField(max_length=15, choices=LEVEL_CHOICES, default=LEVEL_UNAVAILABLE)
    ela_image          = models.ImageField(upload_to='forensics/ela/', blank=True, null=True)
    ela_interpretation = models.TextField(blank=True, default='')

    # ---- Frequency Analysis Results ----
    freq_score           = models.FloatField(default=0.0)
    freq_level           = models.CharField(max_length=15, choices=LEVEL_CHOICES, default=LEVEL_UNAVAILABLE)
    freq_image           = models.ImageField(upload_to='forensics/freq/', blank=True, null=True)
    freq_low_pattern     = models.CharField(max_length=50, blank=True, default='N/A')
    freq_high_pattern    = models.CharField(max_length=50, blank=True, default='N/A')
    freq_interpretation  = models.TextField(blank=True, default='')

    # ---- Noise Analysis Results ----
    noise_score          = models.FloatField(default=0.0)
    noise_level          = models.CharField(max_length=15, choices=LEVEL_CHOICES, default=LEVEL_UNAVAILABLE)
    noise_image          = models.ImageField(upload_to='forensics/noise/', blank=True, null=True)
    noise_consistency    = models.CharField(max_length=20, blank=True, default='N/A')
    noise_mean           = models.FloatField(default=0.0)
    noise_std            = models.FloatField(default=0.0)
    noise_interpretation = models.TextField(blank=True, default='')

    # ---- Metadata Results ----
    meta_format                 = models.CharField(max_length=20, blank=True, default='Unknown')
    meta_width                  = models.IntegerField(default=0)
    meta_height                 = models.IntegerField(default=0)
    meta_mode                   = models.CharField(max_length=10, blank=True, default='')
    meta_file_size_display      = models.CharField(max_length=30, blank=True, default='')
    meta_has_exif               = models.BooleanField(default=False)
    meta_camera_make            = models.CharField(max_length=100, blank=True, default='')
    meta_camera_model           = models.CharField(max_length=100, blank=True, default='')
    meta_software               = models.CharField(max_length=200, blank=True, default='')
    meta_datetime               = models.CharField(max_length=50, blank=True, default='')
    meta_is_editing_software    = models.BooleanField(default=False)
    meta_editing_software_name  = models.CharField(max_length=200, blank=True, default='')
    meta_raw_json               = models.TextField(blank=True, default='{}')
    meta_exif_count             = models.IntegerField(default=0)
    meta_note                   = models.TextField(blank=True, default='')

    # ---- Copy-Move (Clone) Detection Results ----
    copy_move_score          = models.FloatField(default=0.0)
    copy_move_level          = models.CharField(max_length=15, choices=LEVEL_CHOICES, default=LEVEL_UNAVAILABLE)
    copy_move_image          = models.ImageField(upload_to='forensics/copymove/', blank=True, null=True)
    copy_move_cluster_size   = models.IntegerField(default=0)
    copy_move_region_count   = models.IntegerField(default=0)
    copy_move_interpretation = models.TextField(blank=True, default='')

    # ---- Face Boundary / Blending Analysis Results ----
    face_boundary_score           = models.FloatField(default=0.0)
    face_boundary_level           = models.CharField(max_length=15, choices=LEVEL_CHOICES, default=LEVEL_UNAVAILABLE)
    face_boundary_image           = models.ImageField(upload_to='forensics/faceboundary/', blank=True, null=True)
    face_boundary_detected        = models.BooleanField(default=False)
    face_boundary_sharpness_ratio = models.FloatField(default=0.0)
    face_boundary_color_jump      = models.FloatField(default=0.0)
    face_boundary_interpretation  = models.TextField(blank=True, default='')

    # ---- Resampling / Interpolation Artifact Results ----
    resampling_score          = models.FloatField(default=0.0)
    resampling_level          = models.CharField(max_length=15, choices=LEVEL_CHOICES, default=LEVEL_UNAVAILABLE)
    resampling_image          = models.ImageField(upload_to='forensics/resampling/', blank=True, null=True)
    resampling_peak_ratio     = models.FloatField(default=0.0)
    resampling_interpretation = models.TextField(blank=True, default='')

    # ---- C2PA Content Credentials Results ----
    # NOT_PRESENT is the normal/expected status for most images — it is
    # kept out of LEVEL_CHOICES because it's a provenance-check outcome,
    # not an anomaly level.
    C2PA_VALID           = 'VALID'
    C2PA_INVALID         = 'INVALID'
    C2PA_NOT_PRESENT     = 'NOT_PRESENT'
    C2PA_LIBRARY_MISSING = 'LIBRARY_MISSING'
    C2PA_ERROR           = 'ERROR'

    C2PA_STATUS_CHOICES = [
        (C2PA_VALID,           'Valid Manifest'),
        (C2PA_INVALID,         'Invalid Manifest'),
        (C2PA_NOT_PRESENT,     'No Manifest Present'),
        (C2PA_LIBRARY_MISSING, 'Library Not Installed'),
        (C2PA_ERROR,           'Check Error'),
    ]

    c2pa_status            = models.CharField(max_length=20, choices=C2PA_STATUS_CHOICES, default=C2PA_NOT_PRESENT)
    c2pa_has_manifest       = models.BooleanField(default=False)
    c2pa_claim_generator    = models.CharField(max_length=255, blank=True, default='')
    c2pa_validation_state   = models.CharField(max_length=50, blank=True, default='')
    c2pa_manifest_summary   = models.TextField(blank=True, default='')
    c2pa_interpretation     = models.TextField(blank=True, default='')

    # ---- Evidence Scores (0-100) ----
    # These are aggregated by forensics/evidence.py
    evidence_ai         = models.FloatField(default=0.0)   # AI model evidence strength
    evidence_ela        = models.FloatField(default=0.0)   # ELA evidence strength
    evidence_freq       = models.FloatField(default=0.0)   # Frequency evidence strength
    evidence_noise      = models.FloatField(default=0.0)   # Noise evidence strength
    evidence_meta       = models.FloatField(default=0.0)   # Metadata evidence strength
    evidence_copy_move  = models.FloatField(default=0.0)   # Copy-move evidence strength
    evidence_boundary   = models.FloatField(default=0.0)   # Face boundary evidence strength
    evidence_resampling = models.FloatField(default=0.0)   # Resampling evidence strength

    # ---- Overall Risk ----
    # Structured evidence (AI / forensic indicators / provenance / integrity / explainability +
    # assessment reasoning). The authoritative record; the columns around it are legacy display fields.
    evidence_report = models.JSONField(default=dict, blank=True)

    overall_risk_score  = models.FloatField(default=0.0)
    overall_risk        = models.CharField(max_length=15, choices=RISK_CHOICES, default=RISK_LOW)
    risk_interpretation = models.TextField(blank=True, default='')

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Forensic Analysis'
        verbose_name_plural = 'Forensic Analyses'

    def __str__(self):
        return (
            f"Forensic Analysis for Analysis #{self.analysis_id} "
            f"— Risk: {self.overall_risk}"
        )

    def get_ela_level_display_class(self):
        """Returns a CSS class for colour-coding the ELA level badge."""
        return {
            'LOW':  'badge-authentic',
            'MEDIUM': 'badge-uncertain',
            'HIGH': 'badge-manipulated',
        }.get(self.ela_level, 'badge-not-configured')

    def get_freq_level_display_class(self):
        """Returns a CSS class for colour-coding the frequency level badge."""
        return {
            'LOW':  'badge-authentic',
            'MEDIUM': 'badge-uncertain',
            'HIGH': 'badge-manipulated',
        }.get(self.freq_level, 'badge-not-configured')

    def get_noise_level_display_class(self):
        """Returns a CSS class for colour-coding the noise level badge."""
        return {
            'LOW':  'badge-authentic',
            'MEDIUM': 'badge-uncertain',
            'HIGH': 'badge-manipulated',
        }.get(self.noise_level, 'badge-not-configured')

    def get_copy_move_level_display_class(self):
        """Returns a CSS class for colour-coding the copy-move level badge."""
        return {
            'LOW':  'badge-authentic',
            'MEDIUM': 'badge-uncertain',
            'HIGH': 'badge-manipulated',
        }.get(self.copy_move_level, 'badge-not-configured')

    def get_face_boundary_level_display_class(self):
        """Returns a CSS class for colour-coding the face boundary level badge."""
        return {
            'LOW':  'badge-authentic',
            'MEDIUM': 'badge-uncertain',
            'HIGH': 'badge-manipulated',
        }.get(self.face_boundary_level, 'badge-not-configured')

    def get_resampling_level_display_class(self):
        """Returns a CSS class for colour-coding the resampling level badge."""
        return {
            'LOW':  'badge-authentic',
            'MEDIUM': 'badge-uncertain',
            'HIGH': 'badge-manipulated',
        }.get(self.resampling_level, 'badge-not-configured')

    def get_c2pa_status_display_class(self):
        """Returns a CSS class for colour-coding the C2PA status badge."""
        return {
            'VALID':           'badge-authentic',
            'INVALID':         'badge-manipulated',
            'NOT_PRESENT':     'badge-not-configured',
            'LIBRARY_MISSING': 'badge-not-configured',
            'ERROR':           'badge-not-configured',
        }.get(self.c2pa_status, 'badge-not-configured')

    def get_risk_display_class(self):
        """Returns a CSS class for colour-coding the overall risk badge."""
        return {
            'LOW':       'badge-authentic',
            'MEDIUM':    'badge-uncertain',
            'HIGH':      'badge-manipulated',
            'VERY_HIGH': 'badge-manipulated',
        }.get(self.overall_risk, 'badge-not-configured')

    def get_meta_resolution(self):
        """Returns resolution as 'Width × Height' string."""
        if self.meta_width and self.meta_height:
            return f"{self.meta_width} × {self.meta_height}"
        return 'Unknown'




# ============================================================
# MODEL 5: Face — one row per detected face (normalised from the pipeline output)
# ============================================================
class Face(models.Model):
    """A detected face: geometry, detector confidence, alignment, quality flags, the saved crop and
    that face's own ensemble result. Per-model results link via ModelResult.face_index."""

    analysis = models.ForeignKey(Analysis, on_delete=models.CASCADE, related_name='faces')
    face_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    face_index = models.PositiveSmallIntegerField()
    x = models.PositiveIntegerField()
    y = models.PositiveIntegerField()
    w = models.PositiveIntegerField()
    h = models.PositiveIntegerField()
    detector_confidence = models.FloatField(null=True, blank=True)   # None: detector gives no score
    aligned = models.BooleanField(default=False)
    quality_flags = models.JSONField(default=list, blank=True)
    crop_image = models.ImageField(upload_to='faces/', blank=True, default='')   # lossless PNG crop
    ensemble_label = models.CharField(max_length=30, blank=True, default='')
    ensemble_score = models.FloatField(null=True, blank=True)
    ensemble = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['face_index']
        constraints = [models.UniqueConstraint(fields=['analysis', 'face_index'], name='uniq_face_per_analysis')]

    def __str__(self):
        return f'Face {self.face_index} of analysis {self.analysis_id}'


# ============================================================
# MODEL 6: ModelVersion — which exact model produced a result (reproducibility)
# ============================================================
class ModelVersion(models.Model):
    key = models.CharField(max_length=40)
    name = models.CharField(max_length=80)
    version = models.CharField(max_length=40)
    weights_sha256 = models.CharField(max_length=64, blank=True, default='')
    framework = models.CharField(max_length=20, blank=True, default='')
    input_size = models.PositiveIntegerField(null=True, blank=True)
    preprocessing_version = models.CharField(max_length=40, blank=True, default='')
    # training dataset/version/date/parameters/evaluation metrics, thresholds (see ModelSpec.metadata)
    metadata = models.JSONField(default=dict, blank=True)
    first_seen = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['key', '-first_seen']
        constraints = [models.UniqueConstraint(fields=['key', 'version', 'weights_sha256'],
                                               name='uniq_model_version')]

    def __str__(self):
        return f'{self.name} v{self.version}'


# ============================================================
# MODEL 7: Report — immutable, hashed (and HMAC-signed) evidence report
# ============================================================
class Report(models.Model):
    analysis = models.OneToOneField(Analysis, on_delete=models.CASCADE, related_name='report')
    content = models.JSONField()
    content_sha256 = models.CharField(max_length=64)
    signature = models.CharField(max_length=64, blank=True, default='')   # HMAC-SHA256 (hex)
    software_version = models.CharField(max_length=40)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'Report for analysis {self.analysis_id} ({self.content_sha256[:12]})'


# ============================================================
# MODEL 8: AuditLog — security-relevant events (no sensitive content)
# ============================================================
class AuditLog(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL,
                             related_name='audit_events')
    username = models.CharField(max_length=150, blank=True, default='')   # survives user deletion
    action = models.CharField(max_length=40)
    object_type = models.CharField(max_length=40, blank=True, default='')
    object_id = models.CharField(max_length=64, blank=True, default='')
    success = models.BooleanField(default=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    request_id = models.CharField(max_length=64, blank=True, default='')
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [models.Index(fields=['user', '-created_at'], name='audit_user_idx'),
                   models.Index(fields=['action', '-created_at'], name='audit_action_idx')]

    def __str__(self):
        return f'{self.created_at:%Y-%m-%d %H:%M} {self.action} {self.username or "-"}'


# ============================================================
# MODEL 9: AnalysisJob — background processing state (real stages, never faked progress)
# ============================================================
class AnalysisJob(models.Model):
    """One background job per upload. `status` is the actual stage the worker is in; the API
    exposes it as-is (plus stage_index/stage_total) instead of an invented percentage."""

    QUEUED, VALIDATING, FACE_DETECTION = 'QUEUED', 'VALIDATING', 'FACE_DETECTION'
    AI_ANALYSIS, FORENSIC_ANALYSIS, AGGREGATING = 'AI_ANALYSIS', 'FORENSIC_ANALYSIS', 'AGGREGATING'
    REPORT_GENERATION, COMPLETED = 'REPORT_GENERATION', 'COMPLETED'
    FAILED, CANCELLED = 'FAILED', 'CANCELLED'

    STAGES = [QUEUED, VALIDATING, FACE_DETECTION, AI_ANALYSIS, FORENSIC_ANALYSIS, AGGREGATING,
              REPORT_GENERATION, COMPLETED]
    TERMINAL = (COMPLETED, FAILED, CANCELLED)
    ACTIVE = tuple(STAGES[:-1])
    STATUS_CHOICES = [(x, x) for x in STAGES + [FAILED, CANCELLED]]

    upload = models.OneToOneField(UploadedImage, on_delete=models.CASCADE, related_name='job')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=QUEUED)
    last_stage = models.CharField(max_length=20, blank=True, default='')   # stage reached before FAILED/CANCELLED
    attempts = models.PositiveSmallIntegerField(default=0)
    celery_task_id = models.CharField(max_length=64, blank=True, default='')
    cancel_requested = models.BooleanField(default=False)
    partial = models.BooleanField(default=False)               # finished, but some component was unavailable
    warnings = models.JSONField(default=list, blank=True)
    error_code = models.CharField(max_length=30, blank=True, default='')
    error_message = models.CharField(max_length=200, blank=True, default='')   # always generic/safe
    timings_ms = models.JSONField(default=dict, blank=True)
    # Live progress for the processing page: detected face boxes (+ landmarks) as soon as detection
    # finishes, and a short list of human-readable events. Contains no scores.
    live = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [models.Index(fields=['status', 'updated_at'], name='job_status_updated_idx'),
                   models.Index(fields=['status', 'created_at'], name='job_status_created_idx')]

    @property
    def is_terminal(self):
        return self.status in self.TERMINAL

    @property
    def stage_index(self):
        stage = self.status if self.status in self.STAGES else self.last_stage
        return self.STAGES.index(stage) + 1 if stage in self.STAGES else 0

    def __str__(self):
        return f'Job {self.pk} for upload {self.upload_id}: {self.status}'
