"""
AI analysis pipeline for one uploaded image.

    original file (never modified)
      -> processing copy (EXIF-rotated, RGB, metadata-free; saved only if it differs)
      -> face detection -> per face: alignment/crop (PNG) -> quality flags
      -> every model on every face (timeout-protected, failures isolated)
      -> per-face ensemble -> image-level summary

Pure function of its inputs: no database access, so Celery workers (Phase 7) can call it.
"""

import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

import cv2
import numpy as np
from django.conf import settings
from PIL import Image, ImageOps

from ai_models import ensemble
from ai_models.face_detection import align_and_crop, crop_geometry, get_face_detector, quality_flags
from ai_models.registry import get_registry
from ai_models.types import AVAILABLE, TIMEOUT, ModelOutput

logger = logging.getLogger('deepshield.pipeline')
PIPELINE_VERSION = 'pipeline-2'


class PipelineCancelled(Exception):
    """Raised between steps when the caller asks to stop (cooperative cancellation)."""


def _ms(t0):
    return round((time.perf_counter() - t0) * 1000, 2)


def load_processing_image(path):
    """Decode once, normalise orientation/colour. Returns (rgb uint8, changes list, processed PIL)."""
    changes = []
    with Image.open(path) as im:
        if im.getexif().get(0x0112, 1) not in (0, 1):
            changes.append('exif_orientation_applied')
        im = ImageOps.exif_transpose(im)
        if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
            rgba = im.convert('RGBA')
            bg = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, rgba)
            changes.append('alpha_composited_on_white')
        if im.mode != 'RGB':
            changes.append(f'converted_from_{im.mode}')
            im = im.convert('RGB')
        rgb = np.asarray(im, dtype=np.uint8).copy()       # fresh array: carries no metadata
    return rgb, changes


def run_with_timeout(fn, timeout):
    """Run fn() with a wall-clock limit. A timed-out thread cannot be killed in Python and
    is abandoned (Celery hard time limits terminate such workers in production)."""
    ex = ThreadPoolExecutor(max_workers=1)
    try:
        return ex.submit(fn).result(timeout=timeout)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def _rel(path):
    return os.path.relpath(path, settings.MEDIA_ROOT).replace('\\', '/') if path else None


def _save_png(rgb, folder, prefix):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f'{prefix}_{uuid.uuid4().hex[:12]}.png')
    if not cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError('could not write image')
    return path


def run_ai_analysis(image_path, faces_dir, heatmaps_dir, processed_dir=None,
                    registry=None, detector=None, ensemble_cfg=None, on_stage=None, should_cancel=None, on_event=None):
    stage = on_stage or (lambda name: None)
    emit = on_event or (lambda kind, data: None)
    check_cancel = should_cancel or (lambda: False)
    t_total = time.perf_counter()
    timings = {'models_ms': {}}
    registry = registry or get_registry()
    cfg = ensemble_cfg or ensemble.load_ensemble_config()
    result = {'pipeline_version': PIPELINE_VERSION, 'status': 'completed', 'faces': [],
              'face_count': 0, 'timings_ms': timings, 'error': ''}

    t0 = time.perf_counter()
    try:
        rgb, changes = load_processing_image(image_path)
    except Exception:
        logger.exception('could not decode image for analysis')
        return {**result, 'status': 'failed', 'error': 'The image could not be decoded.',
                'image_level': {'label': ensemble.ANALYSIS_FAILED, 'score': None,
                                'representative_face': None, 'explanation': 'Analysis failed.'}}
    timings['decode_ms'] = _ms(t0)
    H, W = rgb.shape[:2]
    processing_copy = None
    if changes and processed_dir:
        processing_copy = _rel(_save_png(rgb, processed_dir, 'proc'))
    result['image'] = {'width': W, 'height': H, 'processing_changes': changes,
                       'processing_copy': processing_copy}

    stage('FACE_DETECTION')
    t0 = time.perf_counter()
    det = detector or get_face_detector()
    faces = det.detect(rgb)
    timings['detect_ms'] = _ms(t0)
    result['detector'] = {'name': det.name, 'min_face_size': settings.FACE_MIN_SIZE,
                          'confidence_threshold': settings.FACE_CONFIDENCE_THRESHOLD
                          if det.name == 'yunet' else None,
                          'padding': settings.FACE_PADDING,
                          'alignment': 'eye-level (landmarks)' if det.supports_landmarks
                          else 'none (detector provides no landmarks)'}
    result['face_count'] = len(faces)
    emit('faces', {'image': {'width': W, 'height': H},
                   'detector': det.name,
                   'faces': [{'index': f.index, 'bbox': list(f.bbox), 'confidence': f.confidence,
                              'landmarks': (np.asarray(f.landmarks).round(1).tolist()
                                            if f.landmarks is not None else None)} for f in faces]})

    stage('AI_ANALYSIS')
    detectors = registry.ordered()
    expected_models = sum(1 for detector in detectors if detector.spec.enabled)
    face_ensembles = []
    for face in faces:
        if check_cancel():
            raise PipelineCancelled()
        crop, aligned = align_and_crop(rgb, face)
        flags = quality_flags(face, rgb.shape)
        crop_path = _save_png(crop, faces_dir, f'face{face.index}')
        outputs, explanations = [], {}
        for d in detectors:
            if check_cancel():
                raise PipelineCancelled()
            t1 = time.perf_counter()
            try:
                out = run_with_timeout(lambda d=d, crop=crop: d.predict(crop), settings.AI_MODEL_TIMEOUT)
            except FuturesTimeout:
                out = ModelOutput(key=d.spec.key, name=d.spec.display_name, status=TIMEOUT,
                                  message='Model exceeded the time limit.', version=d.spec.version,
                                  threshold=d.spec.threshold, scope=d.spec.scope)
            timings['models_ms'][d.spec.key] = timings['models_ms'].get(d.spec.key, 0.0) + _ms(t1)
            if out.status in ('FAILED', TIMEOUT):
                from detector import metrics
                metrics.record_model_failure(d.spec.key)
            outputs.append(out)
            emit('model', {'face': face.index, 'key': d.spec.key, 'name': d.spec.display_name,
                           'status': out.status, 'ms': _ms(t1)})
            if (settings.AI_EXPLAIN_ENABLED and out.status == AVAILABLE
                    and face.index < settings.AI_EXPLAIN_MAX_FACES):
                try:
                    explanations[d.spec.key] = run_with_timeout(
                        lambda d=d, crop=crop: d.explain(crop, heatmaps_dir), settings.AI_MODEL_TIMEOUT)
                except Exception:
                    explanations[d.spec.key] = {'available': False, 'method': None,
                                                'reason': 'Explanation failed or timed out.'}
        t1 = time.perf_counter()
        fe = ensemble.combine(outputs, cfg, flags, expected_models=expected_models)
        timings['ensemble_ms'] = timings.get('ensemble_ms', 0.0) + _ms(t1)
        face_ensembles.append(fe)
        result['faces'].append({
            'index': face.index, 'face_id': uuid.uuid4().hex, 'bbox': list(face.bbox),
            'detector_confidence': face.confidence, 'aligned': aligned, 'quality_flags': flags,
            'landmarks': (np.asarray(face.landmarks).round(1).tolist() if face.landmarks is not None else None),
            'crop_geo': crop_geometry(rgb.shape, face),
            'crop_path': _rel(crop_path), 'crop_size': list(crop.shape[:2]),
            'model_outputs': [o.as_dict() for o in outputs], 'ensemble': fe,
            'explanations': explanations,
        })

    result['image_level'] = ensemble.summarise_faces(face_ensembles)
    if not faces:
        result['status'] = 'no_face'
    result['models'] = [d.health_check() for d in detectors]
    result['model_metadata'] = {d.spec.key: d.spec.metadata() for d in detectors}
    result['ensemble_config'] = {'method': cfg.method, 'version': cfg.version,
                                 'validated': cfg.validated, 'calibrated': cfg.calibrated}
    timings['total_ms'] = _ms(t_total)
    return result
