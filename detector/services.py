"""Persistence of AI pipeline results (shared by the web view and, later, Celery tasks)."""

import logging
import os

from django.conf import settings

from ai_models.ensemble import legacy_final_prediction
from ai_models.types import AVAILABLE, LABEL_MANIPULATED, NOT_CONFIGURED, DISABLED
from detector.models import (Analysis, Face, ForensicAnalysis, ModelResult, ModelVersion,
                             UploadedImage)
from ai_models.registry import ARCHITECTURES

_FAMILY = {k: v[1] for k, v in ARCHITECTURES.items()}
logger = logging.getLogger(__name__)


def _legacy_status(status):
    return {AVAILABLE: ModelResult.STATUS_SUCCESS, NOT_CONFIGURED: ModelResult.STATUS_NOT_CONFIGURED,
            DISABLED: ModelResult.STATUS_NOT_CONFIGURED}.get(status, ModelResult.STATUS_ERROR)


def _legacy_prediction(out):
    if out['status'] == AVAILABLE:
        return 'FAKE' if out['label'] == LABEL_MANIPULATED else 'REAL'
    return 'NOT_CONFIGURED' if out['status'] in (NOT_CONFIGURED, DISABLED) else 'ERROR'


def _to_rel_media(path):
    if not path:
        return None
    return os.path.relpath(path, settings.MEDIA_ROOT).replace('\\', '/')


def persist_ai_results(upload, ai: dict) -> Analysis:
    """Create Analysis + ModelResult rows from run_ai_analysis() output."""
    level = ai['image_level']
    faces = ai['faces']
    rep = level.get('representative_face')
    rep_face = faces[rep] if rep is not None else None
    fe = rep_face['ensemble'] if rep_face else {}
    score = level.get('score') or 0.0
    label = level['label']
    confidence = (1 - score) * 100 if label == 'AUTHENTIC' else score * 100   # legacy semantics

    analysis = Analysis.objects.create(
        uploaded_image=upload,
        final_prediction=legacy_final_prediction(label),
        overall_confidence=round(confidence, 2),
        authentic_votes=fe.get('authentic_votes', 0),
        manipulation_votes=fe.get('manipulated_votes', 0),
        executed_models=fe.get('models_available', 0),
        unavailable_models=fe.get('models_expected', 0) - fe.get('models_available', 0),
        explanation=level.get('explanation', ''),
        pipeline_info={
            'pipeline_version': ai['pipeline_version'], 'image': ai.get('image'),
            'detector': ai.get('detector'), 'ensemble_config': ai.get('ensemble_config'),
            'image_level': level, 'timings_ms': ai['timings_ms'], 'models': ai.get('models'),
            'model_metadata': {k: {'scope': (m or {}).get('scope', 'both')}
                               for k, m in (ai.get('model_metadata') or {}).items()},
            'faces': [{k: v for k, v in f.items() if k not in ('model_outputs', 'explanations')}
                      for f in faces],
        })

    versions = {}
    meta_all = ai.get('model_metadata') or {}
    for f in faces:
        Face.objects.create(
            analysis=analysis, face_id=f['face_id'], face_index=f['index'],
            x=f['bbox'][0], y=f['bbox'][1], w=f['bbox'][2], h=f['bbox'][3],
            detector_confidence=f.get('detector_confidence'), aligned=f['aligned'],
            quality_flags=f['quality_flags'], crop_image=f['crop_path'] or '',
            ensemble_label=f['ensemble']['label'], ensemble_score=f['ensemble']['score'],
            ensemble=f['ensemble'])
        for out in f['model_outputs']:
            if out['version'] and (out['key'], out['version'], out['weights_sha256']) not in versions:
                m = meta_all.get(out['key'], {})
                versions[(out['key'], out['version'], out['weights_sha256'])] = ModelVersion.objects.get_or_create(
                    key=out['key'], version=out['version'], weights_sha256=out['weights_sha256'],
                    defaults={'name': out['name'], 'framework': m.get('framework', ''),
                              'input_size': m.get('input_size'),
                              'preprocessing_version': out['preprocessing_version'], 'metadata': m})[0]

    rows = []
    for f in faces:
        for out in f['model_outputs']:
            ex = f['explanations'].get(out['key'], {})
            heat = ex.get('image_path')
            score_m = out['score'] if out['score'] is not None else 0.0
            rows.append(ModelResult(
                analysis=analysis, face_index=f['index'],
                model_version=versions.get((out['key'], out['version'], out['weights_sha256'])),
                model_name=out['name'], model_key=out['key'], model_type=_FAMILY.get(out['key'], '')[:20],
                prediction=_legacy_prediction(out),
                confidence=(score_m if out['label'] == LABEL_MANIPULATED else 1 - score_m)
                if out['status'] == AVAILABLE else 0.0,
                fake_probability=score_m, status=_legacy_status(out['status']),
                message=out['message'] or '',
                has_gradcam=bool(heat), gradcam_image=_to_rel_media(heat),
                details={k: out[k] for k in ('status', 'version', 'weights_sha256',
                                              'preprocessing_version', 'threshold', 'inference_ms',
                                              'score_type', 'label')}
                        | {'face_id': f['face_id'],
                           'explanation': {k: v for k, v in ex.items() if k != 'image_path'}}))
    ModelResult.objects.bulk_create(rows)
    return analysis


def process_upload(user, uploaded_file, original_filename) -> UploadedImage:
    """Store an already-validated upload, create its background job and enqueue it.
    Raises detector.jobs.JobLimitError when the user/queue limits are reached. In eager mode
    (development/tests) the job has already finished when this returns."""
    from detector import jobs
    return jobs.submit(user, uploaded_file, original_filename)


def _run_forensic_analysis(analysis, image_path, on_stage=None, on_event=None):
    """
    Run all forensic modules and save results to the ForensicAnalysis record.

    This function is called internally by upload_view after AI analysis.
    It creates (or updates) one ForensicAnalysis record linked to the analysis.

    If any individual forensic module fails, it is logged and the rest
    continue. The system never crashes due to a forensic module failure.

    Args:
        analysis (Analysis): The Analysis DB record.
        image_path (str):    Path to the original uploaded image.
    """
    logger.info(f"Starting forensic analysis for Analysis #{analysis.pk}")

    def _ev(name):
        if on_event:
            try:
                on_event('forensic', {'name': name})
            except Exception:                       # live progress must never break the analysis
                logger.debug('live event failed', exc_info=True)

    # Folder for forensic visualization images
    from detector.models import user_media_prefix
    forensic_folder = os.path.join(settings.MEDIA_ROOT, *user_media_prefix(
        analysis.uploaded_image.user_id, 'forensics').split('/'))
    ela_folder         = os.path.join(forensic_folder, 'ela')
    freq_folder        = os.path.join(forensic_folder, 'freq')
    noise_folder       = os.path.join(forensic_folder, 'noise')
    copy_move_folder   = os.path.join(forensic_folder, 'copymove')
    boundary_folder    = os.path.join(forensic_folder, 'faceboundary')
    resampling_folder  = os.path.join(forensic_folder, 'resampling')

    # ---- ELA ----
    ela_result = {'ela_level': 'UNAVAILABLE', 'ela_score': 0.0, 'ela_image_path': None, 'ela_interpretation': ''}
    try:
        from forensics.ela import run_ela_analysis
        ela_result = run_ela_analysis(image_path, ela_folder)
    except Exception as e:
        logger.error(f"ELA module failed: {e}")
    _ev("Error Level Analysis")

    # ---- Frequency ----
    freq_result = {'freq_level': 'UNAVAILABLE', 'freq_score': 0.0, 'freq_image_path': None,
                   'freq_low_pattern': 'N/A', 'freq_high_pattern': 'N/A', 'freq_interpretation': ''}
    try:
        from forensics.frequency import run_frequency_analysis
        freq_result = run_frequency_analysis(image_path, freq_folder)
    except Exception as e:
        logger.error(f"Frequency analysis module failed: {e}")
    _ev("Frequency analysis")

    # ---- Noise ----
    noise_result = {'noise_level': 'UNAVAILABLE', 'noise_score': 0.0, 'noise_image_path': None,
                    'noise_consistency': 'N/A', 'noise_mean': 0.0, 'noise_std': 0.0, 'noise_interpretation': ''}
    try:
        from forensics.noise_analysis import run_noise_analysis
        noise_result = run_noise_analysis(image_path, noise_folder)
    except Exception as e:
        logger.error(f"Noise analysis module failed: {e}")
    _ev("Noise analysis")

    # ---- Metadata ----
    meta_result = {
        'meta_format': 'Unknown', 'meta_width': 0, 'meta_height': 0,
        'meta_mode': '', 'meta_file_size_display': '', 'meta_has_exif': False,
        'meta_camera_make': '', 'meta_camera_model': '', 'meta_software': '',
        'meta_datetime': '', 'meta_is_editing_software': False,
        'meta_editing_software_name': '', 'meta_raw_json': '{}',
        'meta_exif_count': 0, 'meta_note': '',
    }
    try:
        from forensics.metadata import extract_metadata
        meta_result = extract_metadata(image_path)
    except Exception as e:
        logger.error(f"Metadata module failed: {e}")
    _ev("Metadata")

    # ---- Copy-Move (Clone) Detection ----
    copy_move_result = {'copy_move_level': 'UNAVAILABLE', 'copy_move_score': 0.0, 'copy_move_image_path': None,
                        'copy_move_cluster_size': 0, 'copy_move_region_count': 0, 'copy_move_interpretation': ''}
    try:
        from forensics.copy_move import run_copy_move_analysis
        copy_move_result = run_copy_move_analysis(image_path, copy_move_folder)
    except Exception as e:
        logger.error(f"Copy-move module failed: {e}")
    _ev("Copy-move detection")

    # ---- Face Boundary / Blending Analysis ----
    boundary_result = {'face_boundary_level': 'UNAVAILABLE', 'face_boundary_score': 0.0, 'face_boundary_image_path': None,
                       'face_detected': False, 'face_boundary_sharpness_ratio': 0.0, 'face_boundary_color_jump': 0.0,
                       'face_boundary_interpretation': ''}
    try:
        from forensics.face_boundary import run_face_boundary_analysis
        boundary_result = run_face_boundary_analysis(image_path, boundary_folder)
    except Exception as e:
        logger.error(f"Face boundary module failed: {e}")
    _ev("Face boundary")

    # ---- Resampling / Interpolation Artifact Detection ----
    resampling_result = {'resampling_level': 'UNAVAILABLE', 'resampling_score': 0.0, 'resampling_image_path': None,
                         'resampling_peak_ratio': 0.0, 'resampling_interpretation': ''}
    try:
        from forensics.resampling import run_resampling_analysis
        resampling_result = run_resampling_analysis(image_path, resampling_folder)
    except Exception as e:
        logger.error(f"Resampling module failed: {e}")
    _ev("Resampling")

    # ---- C2PA Content Credentials Check ----
    c2pa_result = {'c2pa_status': 'ERROR', 'c2pa_has_manifest': False, 'c2pa_claim_generator': '',
                   'c2pa_validation_state': '', 'c2pa_manifest_summary': '', 'c2pa_interpretation': ''}
    try:
        from forensics.c2pa_check import run_c2pa_check
        c2pa_result = run_c2pa_check(image_path)
    except Exception as e:
        logger.error(f"C2PA check module failed: {e}")

    # ---- Evidence Aggregation ----
    evidence_result = {
        'evidence_ai': 0.0, 'evidence_ela': 0.0, 'evidence_freq': 0.0,
        'evidence_noise': 0.0, 'evidence_meta': 0.0, 'evidence_copy_move': 0.0,
        'evidence_boundary': 0.0, 'evidence_resampling': 0.0,
        'overall_risk_score': 0.0, 'overall_risk': 'LOW',
        'risk_interpretation': '',
    }
    try:
        from forensics.evidence import calculate_evidence_scores
        evidence_result = calculate_evidence_scores(
            ai_confidence       = analysis.overall_confidence,
            ai_prediction       = analysis.final_prediction,
            ela_level           = ela_result.get('ela_level', 'UNAVAILABLE'),
            freq_level          = freq_result.get('freq_level', 'UNAVAILABLE'),
            noise_level         = noise_result.get('noise_level', 'UNAVAILABLE'),
            meta_has_editing_sw = meta_result.get('meta_is_editing_software', False),
            meta_has_exif       = meta_result.get('meta_has_exif', False),
            meta_format         = meta_result.get('meta_format', 'Unknown'),
            copy_move_level     = copy_move_result.get('copy_move_level', 'UNAVAILABLE'),
            face_boundary_level = boundary_result.get('face_boundary_level', 'UNAVAILABLE'),
            resampling_level    = resampling_result.get('resampling_level', 'UNAVAILABLE'),
        )
    except Exception as e:
        logger.error(f"Evidence aggregation failed: {e}")

    if on_stage:
        on_stage('AGGREGATING')
    # A module that ran reports success True/False; one that raised leaves its default dict (no
    # 'success' key). Both count as "did not produce a result" for the job's partial/warnings.
    failed_modules = [name for name, res in (
        ('ELA', ela_result), ('frequency', freq_result), ('noise', noise_result), ('copy-move', copy_move_result),
        ('resampling', resampling_result), ('metadata', meta_result)) if res.get('success') is not True]
    if analysis.uploaded_image.face_detected and boundary_result.get('success') is not True:
        failed_modules.append('face boundary')
    if c2pa_result.get('c2pa_status') == 'LIBRARY_MISSING':
        failed_modules.append('C2PA (verifier not installed)')
    elif c2pa_result.get('success') is not True:
        failed_modules.append('C2PA')

    # ---- Structured evidence report (authoritative record) ----
    evidence_report = {}
    try:
        from forensics.evidence import build_evidence_report
        up = analysis.uploaded_image
        info = (analysis.pipeline_info or {})
        level = dict(info.get('image_level') or {})
        rep = level.get('representative_face')
        faces = info.get('faces') or []
        if rep is not None and rep < len(faces):
            level.update({k: v for k, v in faces[rep].get('ensemble', {}).items() if k not in level})
        expl = [r.details.get('explanation') for r in analysis.primary_model_results()
                if r.has_gradcam and r.details.get('explanation')]
        evidence_report = build_evidence_report(
            level,
            {'ela': ela_result, 'frequency': freq_result, 'noise': noise_result, 'copy_move': copy_move_result,
             'resampling': resampling_result, 'face_boundary': boundary_result,
             'metadata': meta_result, 'c2pa': c2pa_result},
            sha256=up.sha256_hash, file_size=up.file_size,
            explainability={'available': bool(expl), 'methods': expl})
    except Exception:
        logger.exception("Evidence report failed")

    # ---- Create ForensicAnalysis DB record ----
    try:
        forensic = ForensicAnalysis(
            analysis=analysis,
            evidence_report=evidence_report,

            # ELA fields
            ela_score          = ela_result.get('ela_score', 0.0),
            ela_level          = ela_result.get('ela_level', 'UNAVAILABLE'),
            ela_interpretation = ela_result.get('ela_interpretation', ''),

            # Frequency fields
            freq_score          = freq_result.get('freq_score', 0.0),
            freq_level          = freq_result.get('freq_level', 'UNAVAILABLE'),
            freq_low_pattern    = freq_result.get('freq_low_pattern', 'N/A'),
            freq_high_pattern   = freq_result.get('freq_high_pattern', 'N/A'),
            freq_interpretation = freq_result.get('freq_interpretation', ''),

            # Noise fields
            noise_score          = noise_result.get('noise_score', 0.0),
            noise_level          = noise_result.get('noise_level', 'UNAVAILABLE'),
            noise_consistency    = noise_result.get('noise_consistency', 'N/A'),
            noise_mean           = noise_result.get('noise_mean', 0.0),
            noise_std            = noise_result.get('noise_std', 0.0),
            noise_interpretation = noise_result.get('noise_interpretation', ''),

            # Metadata fields
            meta_format               = meta_result.get('meta_format', 'Unknown'),
            meta_width                = meta_result.get('meta_width', 0),
            meta_height               = meta_result.get('meta_height', 0),
            meta_mode                 = meta_result.get('meta_mode', ''),
            meta_file_size_display    = meta_result.get('meta_file_size_display', ''),
            meta_has_exif             = meta_result.get('meta_has_exif', False),
            meta_camera_make          = meta_result.get('meta_camera_make', ''),
            meta_camera_model         = meta_result.get('meta_camera_model', ''),
            meta_software             = meta_result.get('meta_software', ''),
            meta_datetime             = meta_result.get('meta_datetime', ''),
            meta_is_editing_software  = meta_result.get('meta_is_editing_software', False),
            meta_editing_software_name= meta_result.get('meta_editing_software_name', ''),
            meta_raw_json             = meta_result.get('meta_raw_json', '{}'),
            meta_exif_count           = meta_result.get('meta_exif_count', 0),
            meta_note                 = meta_result.get('meta_note', ''),

            # Copy-move fields
            copy_move_score          = copy_move_result.get('copy_move_score', 0.0),
            copy_move_level          = copy_move_result.get('copy_move_level', 'UNAVAILABLE'),
            copy_move_cluster_size   = copy_move_result.get('copy_move_cluster_size', 0),
            copy_move_region_count   = copy_move_result.get('copy_move_region_count', 0),
            copy_move_interpretation = copy_move_result.get('copy_move_interpretation', ''),

            # Face boundary fields
            face_boundary_score           = boundary_result.get('face_boundary_score', 0.0),
            face_boundary_level           = boundary_result.get('face_boundary_level', 'UNAVAILABLE'),
            face_boundary_detected        = boundary_result.get('face_detected', False),
            face_boundary_sharpness_ratio = boundary_result.get('face_boundary_sharpness_ratio', 0.0),
            face_boundary_color_jump      = boundary_result.get('face_boundary_color_jump', 0.0),
            face_boundary_interpretation  = boundary_result.get('face_boundary_interpretation', ''),

            # Resampling fields
            resampling_score          = resampling_result.get('resampling_score', 0.0),
            resampling_level          = resampling_result.get('resampling_level', 'UNAVAILABLE'),
            resampling_peak_ratio     = resampling_result.get('resampling_peak_ratio', 0.0),
            resampling_interpretation = resampling_result.get('resampling_interpretation', ''),

            # C2PA fields
            c2pa_status          = c2pa_result.get('c2pa_status', 'ERROR'),
            c2pa_has_manifest    = c2pa_result.get('c2pa_has_manifest', False),
            c2pa_claim_generator = c2pa_result.get('c2pa_claim_generator', ''),
            c2pa_validation_state= c2pa_result.get('c2pa_validation_state', ''),
            c2pa_manifest_summary= c2pa_result.get('c2pa_manifest_summary', ''),
            c2pa_interpretation  = c2pa_result.get('c2pa_interpretation', ''),

            # Evidence scores
            evidence_ai         = evidence_result.get('evidence_ai', 0.0),
            evidence_ela        = evidence_result.get('evidence_ela', 0.0),
            evidence_freq       = evidence_result.get('evidence_freq', 0.0),
            evidence_noise      = evidence_result.get('evidence_noise', 0.0),
            evidence_meta       = evidence_result.get('evidence_meta', 0.0),
            evidence_copy_move  = evidence_result.get('evidence_copy_move', 0.0),
            evidence_boundary   = evidence_result.get('evidence_boundary', 0.0),
            evidence_resampling = evidence_result.get('evidence_resampling', 0.0),

            # Overall risk
            overall_risk_score  = evidence_result.get('overall_risk_score', 0.0),
            overall_risk        = evidence_result.get('overall_risk', 'LOW'),
            risk_interpretation = evidence_result.get('risk_interpretation', ''),
        )
        forensic.save()

        # Save image files for ELA, frequency, noise, copy-move, face
        # boundary, and resampling visualizations. Need to store them
        # relative to MEDIA_ROOT like Django's ImageField expects.
        ela_path = ela_result.get('ela_image_path')
        if ela_path and os.path.exists(ela_path):
            forensic.ela_image = _to_rel_media(ela_path)

        freq_path = freq_result.get('freq_image_path')
        if freq_path and os.path.exists(freq_path):
            forensic.freq_image = _to_rel_media(freq_path)

        noise_path = noise_result.get('noise_image_path')
        if noise_path and os.path.exists(noise_path):
            forensic.noise_image = _to_rel_media(noise_path)

        copy_move_path = copy_move_result.get('copy_move_image_path')
        if copy_move_path and os.path.exists(copy_move_path):
            forensic.copy_move_image = _to_rel_media(copy_move_path)

        boundary_path = boundary_result.get('face_boundary_image_path')
        if boundary_path and os.path.exists(boundary_path):
            forensic.face_boundary_image = _to_rel_media(boundary_path)

        resampling_path = resampling_result.get('resampling_image_path')
        if resampling_path and os.path.exists(resampling_path):
            forensic.resampling_image = _to_rel_media(resampling_path)

        forensic.save()

        logger.info(f"ForensicAnalysis #{forensic.pk} saved. Risk: {forensic.overall_risk}")

    except Exception as e:
        logger.error(f"Failed to save ForensicAnalysis record: {e}")
    return failed_modules
