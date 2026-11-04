"""
detector/views.py — All page logic for DeepShield.

Each function in this file corresponds to one page of the website.
Django calls the right function based on the URL the user visits.

Views handle:
  1. Receiving the browser request
  2. Processing data (forms, uploads, AI analysis)
  3. Sending back an HTML response

Public pages (URLs start with /public/):  landing, register, login, privacy, limitations
Signed-in pages (URLs start with /auth/): upload, processing, result, history, detail, report, account
"""

import json
import logging

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Avg
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_POST

from detector import audit, ratelimit
from detector.jobs import JobLimitError
from detector.services import process_upload

from detector.models import UploadedImage, Analysis, ForensicAnalysis
from detector.forms import UserRegistrationForm, ImageUploadForm

logger = logging.getLogger(__name__)


def _too_many(request, template, context, scope):
    """429 page for rate-limited actions."""
    messages.error(request, "Too many attempts. Please wait a while before trying again.")
    response = render(request, template, context, status=429)
    response['Retry-After'] = str(ratelimit.retry_after(scope))
    return response


# ============================================================
# HOME PAGE
# ============================================================
def home_view(request):
    """
    Show the home/landing page.
    This page is visible to everyone (no login required).
    """
    if request.user.is_authenticated:
        return redirect('upload')          # signed-in users go straight to the Analyse page
    return render(request, 'home.html')


# ============================================================
# REGISTER PAGE
# ============================================================
def register_view(request):
    """
    Allow a new user to create an account.

    GET request:  Show empty registration form.
    POST request: Validate form data and create user if valid.
    """

    # If user is already logged in, send them to the Analyse page
    if request.user.is_authenticated:
        return redirect('upload')

    if request.method == 'POST':
        if not ratelimit.allow('register', ratelimit.client_ip(request)):
            return _too_many(request, 'registration/register.html',
                             {'form': UserRegistrationForm()}, 'register')
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            # Save the new user to the database
            # Django automatically hashes the password (never stored as plain text)
            user = form.save()
            login(request, user, backend='django.contrib.auth.backends.ModelBackend')
            audit.audit(audit.LOGIN, request, user=user, channel='web')
            messages.success(request, f"Welcome, {user.username}! Your account is ready.")
            return redirect('upload')
        # If form has errors, fall through to re-render with error messages
    else:
        # GET request — show an empty form
        form = UserRegistrationForm()

    return render(request, 'registration/register.html', {'form': form})


# ============================================================
# LOGIN PAGE
# ============================================================
def login_view(request):
    """
    Allow an existing user to log in.

    GET request:  Show the login form.
    POST request: Validate credentials and log the user in.
    """

    if request.user.is_authenticated:
        return redirect('upload')

    next_url = request.POST.get('next') or request.GET.get('next') or ''

    if request.method == 'POST':
        username = request.POST.get('username', '').strip()[:150]
        password = request.POST.get('password', '')
        ip = ratelimit.client_ip(request)
        pair_id, ip_id = f'{ip}|{username.lower()}', ip

        if ratelimit.is_limited('login_fail', pair_id) or ratelimit.is_limited('login_fail_ip', ip_id):
            logger.warning('login throttled', extra={'ip': ip})
            return _too_many(request, 'registration/login.html', {'next': next_url}, 'login_fail')

        user = authenticate(request, username=username, password=password)
        if user is not None:
            ratelimit.reset('login_fail', pair_id)
            login(request, user)
            audit.audit(audit.LOGIN, request, user=user, channel='web')
            # Only same-host redirects are allowed (prevents open redirect).
            if not url_has_allowed_host_and_scheme(
                    next_url, allowed_hosts={request.get_host()},
                    require_https=request.is_secure()):
                next_url = ''
            return redirect(next_url or 'upload')
        audit.audit(audit.LOGIN_FAILED, request, success=False, channel='web', username_hash=audit.hash_identifier(username))
        ratelimit.hit('login_fail', pair_id)
        ratelimit.hit('login_fail_ip', ip_id)
        messages.error(request, "Invalid username or password. Please try again.")

    return render(request, 'registration/login.html', {'next': next_url})


# ============================================================
# LOGOUT
# ============================================================
@require_POST
def logout_view(request):
    audit.audit(audit.LOGOUT, request, channel='web')
    """
    Log out the current user and redirect to home page.
    POST only (a GET link could be triggered cross-site to log a user out).
    """
    logout(request)
    messages.info(request, "You have been logged out successfully.")
    return redirect('home')


# ============================================================
# UPLOAD PAGE
# ============================================================
@login_required
def upload_view(request):
    """
    Allow a logged-in user to upload an image for analysis.

    GET request:  Show the empty upload form.
    POST request: Validate and process the uploaded image.

    When an image is successfully uploaded, this view:
      1. Saves the file to media/uploads/
      2. Creates an UploadedImage record in the database
      3. Runs the full preprocessing pipeline (face detection, SHA-256)
      4. Runs all 7 AI models
      5. Saves Analysis and ModelResult records
      6. Runs forensic modules (ELA, Frequency, Noise, Metadata)
      7. Saves ForensicAnalysis record
      8. Redirects to the result page
    """

    if request.method == 'POST':
        if not ratelimit.allow('upload', f'user:{request.user.pk}'):
            return _too_many(request, 'detector/upload.html',
                             {'form': ImageUploadForm()}, 'upload')
        form = ImageUploadForm(request.POST, request.FILES)

        if form.is_valid():
            # Get the validated image file
            uploaded_file = form.cleaned_data['image']

            try:
                img_record = process_upload(request.user, uploaded_file, form.original_filename)
            except JobLimitError as exc:
                audit.audit(audit.THROTTLED, request, success=False, scope=exc.code.lower())
                messages.error(request, exc.message)
                return redirect('upload')
            audit.audit(audit.ANALYSIS_CREATED, request, obj=img_record,
                        success=img_record.processing_status != UploadedImage.STATUS_FAILED)
            status = img_record.processing_status
            job = getattr(img_record, 'job', None)
            if job is not None and job.error_code == 'QUEUE_UNAVAILABLE':
                messages.error(request, "The analysis queue is unavailable. Please try again later.")
                return redirect('upload')
            if status == UploadedImage.STATUS_FAILED:
                messages.error(request, "An error occurred while processing the image. Please try again.")
                return redirect('upload')
            if status in (UploadedImage.STATUS_PENDING, UploadedImage.STATUS_PROCESSING):
                messages.info(request, "Your image is being analysed.")
                return redirect('processing', pk=img_record.pk)       # real stages, polled from the job
            messages.success(request, "Image analysis completed successfully.")
            return redirect('result', pk=img_record.pk)

        # If form is invalid, fall through to re-render with errors
    else:
        form = ImageUploadForm()

    return render(request, 'detector/upload.html', {'form': form, 'retention_days': dj_settings.DELETE_ANALYSIS_AFTER_DAYS})


# ============================================================
# RESULT PAGE
# ============================================================
_SCOPE_LABEL = {'faceswap': 'Trained for: face-swap / deepfake',
                'ai_generated': 'Trained for: AI-generated images',
                'both': 'Trained for: face-swap + AI-generated'}


def _face_summaries(analysis):
    """Per-face ensemble results for display: label, score (+ what kind), disagreement, plain-language reasons."""
    faces_view, rep_face = [], None
    if analysis is None:
        return faces_view, rep_face
    from ai_models.ensemble import _REASON_TEXT
    pipeline_info = analysis.pipeline_info or {}
    model_health = pipeline_info.get('models') or []
    expected_models = (sum(1 for model in model_health if model.get('status') != 'DISABLED')
                       if model_health else None)
    rep_idx = (pipeline_info.get('image_level') or {}).get('representative_face')
    scopes = {k: (m or {}).get('scope', 'both') for k, m in (pipeline_info.get('model_metadata') or {}).items()}
    for f in analysis.faces.order_by('face_index'):
        e = f.ensemble or {}
        available = e.get('models_available', 0)
        item = {'index': f.face_index + 1,
                'face_index': f.face_index,
                'crop': f.crop_image.url if f.crop_image else None,
                'label': e.get('label') or f.ensemble_label,
                'score': e.get('score'), 'score_kind': e.get('score_kind', ''), 'validated': e.get('validated'),
                'disagreement': bool(e.get('disagreement')), 'std': e.get('score_std'),
                'reasons': [_REASON_TEXT.get(r, r) for r in e.get('reasons', [])],
                'quality_flags': e.get('quality_flags') or f.quality_flags or [],
                'votes_m': e.get('manipulated_votes', 0), 'votes_a': e.get('authentic_votes', 0),
                'available': available,
                'expected': max(available, expected_models if expected_models is not None
                                else e.get('models_expected', 0)),
                'explanation': e.get('explanation', ''), 'representative': f.face_index == rep_idx,
                'na_votes': e.get('not_applicable_votes', 0),
                'domains': list((e.get('domains') or {}).values()),
                'specialists': [s_['name'] for s_ in e.get('specialist_flags') or []],
                'model_results': list(analysis.model_results.filter(face_index=f.face_index).order_by('model_key'))}
        for mr in item['model_results']:
            mr.domain_label = _SCOPE_LABEL.get(scopes.get(mr.model_key, 'both'), '')
        faces_view.append(item)
        if item['representative']:
            rep_face = item
    return faces_view, rep_face



# ------------------------------------------------------------------ result page: image viewer + forensic cards
_LEVEL_TEXT = {'LOW': 'Low', 'MEDIUM': 'Medium', 'HIGH': 'High'}


def _media_url(rel):
    from django.core.files.storage import default_storage
    return default_storage.url(rel) if rel else ''


def _viz_payload(upload, analysis, faces_view):
    """Everything the image viewer needs (face boxes, landmarks, crop geometry, attention grids).
    Rendered into a data attribute (not an inline script) so the CSP nonce rules stay simple."""
    info = analysis.pipeline_info or {}
    img = info.get('image') or {}
    stored = {f.get('index'): f for f in (info.get('faces') or [])}
    src = _media_url(img.get('processing_copy')) or (upload.image.url if upload.image else '')
    faces = []
    for f in analysis.faces.order_by('face_index'):
        meta = stored.get(f.face_index, {})
        view = next((v for v in faces_view if v['face_index'] == f.face_index), {})
        grids = []
        for mr in view.get('model_results', []):
            cam = ((mr.details or {}).get('explanation') or {}).get('cam')
            if cam and mr.status == 'success' and mr.prediction == 'FAKE':
                grids.append({'key': mr.model_key, 'name': mr.model_name, 'grid': cam})
        faces.append({
            'index': f.face_index, 'bbox': [f.x, f.y, f.w, f.h], 'confidence': f.detector_confidence,
            'landmarks': meta.get('landmarks'), 'crop_geo': meta.get('crop_geo'),
            'crop': f.crop_image.url if f.crop_image else '', 'label': f.ensemble_label,
            'grids': grids})
    return {'src': src, 'width': img.get('width'), 'height': img.get('height'), 'faces': faces}


def _forensic_cards(forensic, upload):
    """One dict per forensic test for the result page cards (name, risk level, score, image, details)."""
    if forensic is None:
        return []

    def img(field):
        return field.url if field else ''

    def card(key, name, icon, level, score, image, desc, interp, metrics=None, image_label=''):
        avail = level not in ('UNAVAILABLE', '', None)
        return {'key': key, 'name': name, 'icon': icon, 'level': level, 'available': avail,
                'level_text': _LEVEL_TEXT.get(level, 'Not applicable'),
                'score': round(score) if avail and score is not None else None,
                'score_exact': f'{score:.1f}' if avail and score is not None else '',
                'image': image, 'image_label': image_label, 'desc': desc, 'interpretation': interp,
                'metrics': metrics or []}

    f = forensic
    cards = [
        card('ela', 'Error Level Analysis (ELA)', 'bi-layers', f.ela_level, f.ela_score, img(f.ela_image),
             'Re-saves the image and compares compression levels. Edited areas often compress differently and glow brighter.',
             f.ela_interpretation, [], 'ELA visualisation'),
        card('freq', 'Frequency Analysis (FFT)', 'bi-activity', f.freq_level, f.freq_score, img(f.freq_image),
             'Looks for regular peaks in the frequency spectrum that AI generators and resizing leave behind.',
             f.freq_interpretation, [('Low frequency', f.freq_low_pattern), ('High frequency', f.freq_high_pattern)],
             'Frequency spectrum'),
        card('noise', 'Noise &amp; Artifact Analysis'.replace('&amp;', '&'), 'bi-soundwave', f.noise_level, f.noise_score,
             img(f.noise_image),
             'Real cameras leave a random grain. Smooth or uneven noise can mean an area was edited.',
             f.noise_interpretation, [('Consistency', f.noise_consistency), ('Noise mean', f'{f.noise_mean:.2f}'),
                                      ('Noise std', f'{f.noise_std:.2f}')], 'Noise residual'),
        card('copymove', 'Copy-Move (Clone) Detection', 'bi-front', f.copy_move_level, f.copy_move_score,
             img(f.copy_move_image), 'Searches for regions that were cloned inside the same image.',
             f.copy_move_interpretation, [('Regions', str(f.copy_move_region_count)),
                                          ('Largest cluster', str(f.copy_move_cluster_size))], 'Matched regions'),
        card('boundary', 'Face Boundary Analysis', 'bi-vector-pen', f.face_boundary_level, f.face_boundary_score,
             img(f.face_boundary_image), 'Checks the edge where a pasted face meets the neck and hair for seams.',
             f.face_boundary_interpretation, [('Sharpness ratio', f'{f.face_boundary_sharpness_ratio:.2f}'),
                                              ('Colour jump', f'{f.face_boundary_color_jump:.2f}')],
             'Boundary map'),
        card('resampling', 'Resampling Analysis', 'bi-aspect-ratio', f.resampling_level, f.resampling_score,
             img(f.resampling_image), 'Detects traces of resizing or rotating part of the image.',
             f.resampling_interpretation, [('Peak ratio', f'{f.resampling_peak_ratio:.2f}')], 'Resampling spectrum'),
    ]
    c2 = card('c2pa', 'C2PA Content Credentials', 'bi-patch-check', 'NONE', None, '',
              'Looks for signed content credentials that record how a file was made.',
              f.c2pa_interpretation,
              [('Status', f.get_c2pa_status_display())] + ([('Claim generator', f.c2pa_claim_generator),
                                                           ('Validation', f.c2pa_validation_state)] if f.c2pa_has_manifest else []))
    c2['level_text'] = f.get_c2pa_status_display()
    c2['level'] = 'VALID' if f.c2pa_status == 'VALID' else ('INVALID' if f.c2pa_status == 'INVALID' else 'NONE')
    c2['available'] = False
    cards.append(c2)
    meta_metrics = [('Format', f.meta_format), ('Resolution', f.get_meta_resolution()),
                    ('Colour mode', f.meta_mode or 'Unknown'), ('EXIF tags', str(f.meta_exif_count))]
    if f.meta_camera_make:
        meta_metrics.append(('Camera', f'{f.meta_camera_make} {f.meta_camera_model}'.strip()))
    if f.meta_software:
        meta_metrics.append(('Software', f.meta_software))
    if f.meta_datetime:
        meta_metrics.append(('Date / time', f.meta_datetime))
    exif = card('exif', 'Image Metadata (EXIF)', 'bi-info-square', 'NONE', None, '',
                'Reads the file\'s hidden camera and software tags.', f.meta_note, meta_metrics)
    exif['level'] = 'MEDIUM' if f.meta_is_editing_software else ('LOW' if f.meta_has_exif else 'NONE')
    exif['level_text'] = ('Editing software' if f.meta_is_editing_software
                          else ('EXIF found' if f.meta_has_exif else 'No EXIF'))
    exif['available'] = False
    cards.append(exif)
    return cards


@login_required
def result_view(request, pk):
    """
    Show the full analysis result for one uploaded image.
    pk = primary key of the UploadedImage record.
    """

    upload = get_object_or_404(UploadedImage, pk=pk, user=request.user)
    job = getattr(upload, 'job', None) if hasattr(upload, 'job') else None
    if job is not None and not job.is_terminal:
        return redirect('processing', pk=upload.pk)

    analysis = None
    model_results = []
    forensic = None

    try:
        analysis = upload.analysis
        model_results = analysis.primary_model_results()
        try:
            forensic = analysis.forensic_analysis
        except ForensicAnalysis.DoesNotExist:
            forensic = None
    except Analysis.DoesNotExist:
        pass

    faces_view, rep_face = _face_summaries(analysis)
    model_health = (analysis.pipeline_info or {}).get('models') or [] if analysis else []
    expected_models = (sum(1 for model in model_health if model.get('status') != 'DISABLED')
                       if model_health else
                       analysis.executed_models + analysis.unavailable_models if analysis else 0)
    analysis_explanation = analysis.explanation if analysis else ''
    if analysis and expected_models != analysis.executed_models + analysis.unavailable_models:
        stored_expected = analysis.executed_models + analysis.unavailable_models
        analysis_explanation = analysis_explanation.replace(
            f'{analysis.executed_models} of {stored_expected} models contributed',
            f'{analysis.executed_models} of {expected_models} models contributed')
        analysis_explanation = analysis_explanation.replace(
            f'({stored_expected} configured/expected)', f'({expected_models} configured/expected)')
    context = {
        'faces_view':   faces_view,
        'rep_face':     rep_face,
        'upload':       upload,
        'analysis':     analysis,
        'model_results': model_results,
        'expected_models': expected_models,
        'unavailable_models': max(0, expected_models - analysis.executed_models) if analysis else 0,
        'analysis_explanation': analysis_explanation,
        'forensic':     forensic,
        'forensic_cards': _forensic_cards(forensic, upload),
        'viz_json':     json.dumps(_viz_payload(upload, analysis, faces_view)) if analysis else '',
        'disclaimer': (
            "IMPORTANT DISCLAIMER: This result is an AI-based forensic estimate only. "
            "It is not a legal proof of authenticity or manipulation. "
            "Detection accuracy depends on the quality of training data and may fail on "
            "unseen manipulation techniques. Results should be interpreted by a qualified professional."
        ),
    }
    return render(request, 'detector/result.html', context)


# ============================================================
# HISTORY PAGE
# ============================================================
@login_required
def history_view(request):
    """
    Show all previous image uploads for the current user.
    Supports search (by filename) and filter (by verdict).
    """

    uploads = UploadedImage.objects.filter(
        user=request.user
    ).select_related('analysis').order_by('-uploaded_at')

    # --- Search filter (by filename) ---
    search_query = request.GET.get('q', '').strip()
    if search_query:
        uploads = uploads.filter(original_filename__icontains=search_query)

    # --- Verdict filter ---
    verdict_filter = request.GET.get('verdict', '').strip()
    if verdict_filter in ('LIKELY_AUTHENTIC', 'LIKELY_MANIPULATED', 'UNCERTAIN'):
        uploads = uploads.filter(analysis__final_prediction=verdict_filter)
    elif verdict_filter == 'no_face':
        uploads = uploads.filter(processing_status='no_face')
    elif verdict_filter == 'failed':
        uploads = uploads.filter(processing_status='failed')

    context = {
        'uploads':        uploads,
        'search_query':   search_query,
        'verdict_filter': verdict_filter,
    }
    return render(request, 'detector/history.html', context)


# ============================================================
# ANALYSIS DETAIL PAGE
# ============================================================
@login_required
def detail_view(request, pk):
    """
    Show the complete detailed result for one analysis.
    Accessed from the history page.
    """

    upload = get_object_or_404(UploadedImage, pk=pk, user=request.user)
    job = getattr(upload, 'job', None) if hasattr(upload, 'job') else None
    if job is not None and not job.is_terminal:
        return redirect('processing', pk=upload.pk)

    analysis = None
    model_results = []
    forensic = None

    try:
        analysis = upload.analysis
        model_results = analysis.primary_model_results()
        try:
            forensic = analysis.forensic_analysis
        except ForensicAnalysis.DoesNotExist:
            forensic = None
    except Analysis.DoesNotExist:
        pass

    context = {
        'upload':        upload,
        'analysis':      analysis,
        'model_results': model_results,
        'forensic':      forensic,
        'disclaimer': (
            "IMPORTANT DISCLAIMER: This result is an AI-based forensic estimate only. "
            "It is not a legal proof of authenticity or manipulation."
        ),
    }
    return render(request, 'detector/detail.html', context)


# ============================================================
# FORENSIC REPORT PAGE  [NEW]
# ============================================================
@login_required
def report_view(request, pk):
    """
    Show a clean, printable forensic report for one analysis.

    This page is optimised for print (browser Print-to-PDF).
    It includes all forensic evidence in a structured report format.

    URL: /report/<pk>/
    """

    upload = get_object_or_404(UploadedImage, pk=pk, user=request.user)
    job = getattr(upload, 'job', None) if hasattr(upload, 'job') else None
    if job is not None and not job.is_terminal:
        return redirect('processing', pk=upload.pk)

    analysis = None
    model_results = []
    forensic = None

    try:
        analysis = upload.analysis
        model_results = analysis.primary_model_results()
        try:
            forensic = analysis.forensic_analysis
        except ForensicAnalysis.DoesNotExist:
            forensic = None
    except Analysis.DoesNotExist:
        pass

    from django.utils import timezone
    faces_view, rep_face = _face_summaries(analysis)
    context = {
        'faces_view':    faces_view,
        'rep_face':      rep_face,
        'upload':        upload,
        'analysis':      analysis,
        'model_results': model_results,
        'forensic':      forensic,
        'report_time':   timezone.now(),
        'disclaimer': (
            "This forensic report is an AI-based analysis estimate only. "
            "It is NOT legal proof of image authenticity or manipulation. "
            "The results of this system must not be used as sole evidence in "
            "any legal, professional, or official proceeding without validation "
            "by qualified forensic professionals."
        ),
    }
    return render(request, 'detector/report.html', context)


# ============================================================
# PASSWORD RESET (Django's secure token flow; rate-limited, no account enumeration)
# ============================================================
from django.contrib.auth import views as auth_views  # noqa: E402


class ThrottledPasswordResetView(auth_views.PasswordResetView):
    """Same response whether or not the e-mail exists; abuse limited per IP."""
    template_name = 'registration/password_reset_form.html'
    email_template_name = 'registration/password_reset_email.txt'
    subject_template_name = 'registration/password_reset_subject.txt'
    success_url = '/password-reset/done/'

    def post(self, request, *args, **kwargs):
        if not ratelimit.allow('password_reset', ratelimit.client_ip(request)):
            return _too_many(request, self.template_name, {'form': self.get_form()}, 'password_reset')
        audit.audit(audit.PASSWORD_RESET_REQUESTED, request)
        return super().post(request, *args, **kwargs)


# ============================================================
# BACKGROUND JOB STATUS (real stages from the AnalysisJob, no invented percentages)
# ============================================================
from django.http import JsonResponse  # noqa: E402

from detector.models import AnalysisJob  # noqa: E402


def _owned_upload_or_404(request, pk):
    return get_object_or_404(UploadedImage.objects.select_related('job'), pk=pk, user=request.user)


def _job_payload(upload):
    job = getattr(upload, 'job', None)
    if job is None:
        return {'status': upload.processing_status.upper(), 'finished': True}
    live = job.live or {}
    return {'status': job.status, 'stage_index': job.stage_index, 'stage_total': len(AnalysisJob.STAGES),
            'stages': AnalysisJob.STAGES, 'finished': job.is_terminal, 'partial': job.partial,
            'warnings': job.warnings, 'error': job.error_message or None, 'attempts': job.attempts,
            'live': {'image': live.get('image'), 'faces': live.get('faces') or [],
                     'events': live.get('events') or [], 'detector': live.get('detector')}}


@login_required
def processing_view(request, pk):
    upload = _owned_upload_or_404(request, pk)
    job = getattr(upload, 'job', None)
    if job is None or job.is_terminal:
        return redirect('result', pk=upload.pk)
    return render(request, 'detector/processing.html', {'upload': upload, 'job': _job_payload(upload)})


@login_required
def job_status_view(request, pk):
    upload = _owned_upload_or_404(request, pk)
    return JsonResponse(_job_payload(upload))


@login_required
@require_POST
def cancel_job_view(request, pk):
    from detector import jobs
    upload = _owned_upload_or_404(request, pk)
    try:
        outcome = jobs.request_cancel(upload.job)
    except (ValueError, AnalysisJob.DoesNotExist):
        messages.info(request, "This analysis has already finished.")
    else:
        audit.audit(audit.ANALYSIS_CANCELLED, request, obj=upload, channel='web')
        messages.success(request, "Analysis cancelled." if outcome == 'cancelled' else "Cancelling the analysis…")
    return redirect('history')


# ============================================================
# PRIVACY, DATA DELETION, ACCOUNT
# ============================================================
from django.conf import settings as dj_settings  # noqa: E402
from django.contrib.auth.forms import PasswordChangeForm  # noqa: F401,E402


def privacy_view(request):
    return render(request, 'privacy.html', {
        'retention_days': dj_settings.DELETE_ANALYSIS_AFTER_DAYS,
        'audit_days': dj_settings.AUDIT_LOG_RETENTION_DAYS,
        'max_mb': dj_settings.MAX_UPLOAD_SIZE // (1024 * 1024)})


@login_required
@require_POST
def delete_analysis_view(request, pk):
    """Permanently delete one analysis and every stored file derived from it."""
    upload = get_object_or_404(UploadedImage, pk=pk, user=request.user)
    job = getattr(upload, 'job', None)
    if job is not None and not job.is_terminal:
        messages.error(request, 'This analysis is still running. Cancel it first, then delete it.')
        return redirect('processing', pk=upload.pk)
    audit.audit(audit.ANALYSIS_DELETED, request, obj=upload, channel='web')
    upload.delete()
    messages.success(request, 'The analysis and all its stored images were permanently deleted.')
    return redirect('history')


@login_required
def account_view(request):
    return render(request, 'account.html', {
        'analysis_count': UploadedImage.objects.filter(user=request.user).count(),
        'retention_days': dj_settings.DELETE_ANALYSIS_AFTER_DAYS})


@login_required
@require_POST
def delete_account_view(request):
    from detector import retention
    ident = f'acct|{request.user.pk}'
    if ratelimit.is_limited('login_fail', ident):
        messages.error(request, 'Too many incorrect attempts. Please try again later.')
        return redirect('account')
    if not request.user.check_password(request.POST.get('password', '')):
        ratelimit.hit('login_fail', ident)
        audit.audit(audit.ACCESS_DENIED, request, success=False, resource='account_delete')
        messages.error(request, 'Incorrect password. Nothing was deleted.')
        return redirect('account')
    if request.POST.get('confirm', '').strip() != 'DELETE':
        messages.error(request, 'Type DELETE to confirm. Nothing was deleted.')
        return redirect('account')
    retention.delete_account(request.user, request)
    logout(request)
    messages.success(request, 'Your account and all your data were permanently deleted.')
    return redirect('home')


# ============================================================
# PUBLIC INFORMATION PAGE
# ============================================================
def limitations_view(request):
    return render(request, 'pages/limitations.html')


def root_view(request):
    """/ sends signed-in users to the Analyse page and visitors to the public landing page."""
    return redirect('upload' if request.user.is_authenticated else 'home')
