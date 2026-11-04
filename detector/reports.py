"""
Evidence reports: reproducible content, SHA-256 content hash and HMAC signature.

The hash proves the stored report was not altered after creation; the HMAC (keyed with
REPORT_SIGNING_KEY, default derived from SECRET_KEY) additionally proves it was created by this
installation. Rotating the key makes older signatures unverifiable (hash still verifies).
"""

import hashlib
import hmac
import json

from django.conf import settings
from django.utils import timezone

from detector.models import Face, ForensicAnalysis, ModelResult, Report

REPORT_VERSION = 'report-1'
LIMITATIONS = [
    'Deepfake detection is probabilistic; false positives and false negatives occur.',
    'Image manipulation techniques evolve and can evade detectors trained on older data.',
    'Compression, resizing and editing can change detection results.',
    'Forensic indicators are supporting evidence only; no single indicator establishes authenticity '
    'or manipulation.',
    'Scores are model outputs; they are not calibrated probabilities unless the ensemble states that '
    'it was validated and calibrated.',
]


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False, default=str)


def content_hash(content: dict) -> str:
    return hashlib.sha256(canonical_json(content).encode('utf-8')).hexdigest()


def sign(digest: str) -> str:
    key = getattr(settings, 'REPORT_SIGNING_KEY', '') or settings.SECRET_KEY
    return hmac.new(key.encode(), digest.encode(), hashlib.sha256).hexdigest()


def _strip_artifacts(evidence: dict) -> dict:
    """Reports never embed server file paths."""
    ev = json.loads(canonical_json(evidence or {}))
    for ind in (ev.get('forensic_indicators') or {}).values():
        ind['artifact'] = bool(ind.get('artifact'))
    return ev


def build_report_content(analysis) -> dict:
    up = analysis.uploaded_image
    info = analysis.pipeline_info or {}
    forensic = ForensicAnalysis.objects.filter(analysis=analysis).first()
    faces = []
    versions = {}
    for face in Face.objects.filter(analysis=analysis).order_by('face_index'):
        models_ = []
        for r in ModelResult.objects.filter(analysis=analysis, face_index=face.face_index).order_by('model_key'):
            d = r.details or {}
            models_.append({'key': r.model_key, 'name': r.model_name, 'status': d.get('status', r.status),
                            'score': None if r.status != 'success' else round(r.fake_probability, 4),
                            'label': d.get('label'), 'threshold': d.get('threshold'),
                            'version': d.get('version', ''), 'weights_sha256': d.get('weights_sha256', ''),
                            'preprocessing_version': d.get('preprocessing_version', ''),
                            'score_type': d.get('score_type', '')})
            if d.get('version'):
                versions[(r.model_key, d['version'])] = {'key': r.model_key, 'version': d['version'],
                                                         'weights_sha256': d.get('weights_sha256', '')}
        faces.append({'index': face.face_index, 'face_id': str(face.face_id),
                      'bbox': [face.x, face.y, face.w, face.h], 'detector_confidence': face.detector_confidence,
                      'aligned': face.aligned, 'quality_flags': face.quality_flags,
                      'ensemble': {k: v for k, v in (face.ensemble or {}).items() if k != 'explanation'},
                      'ensemble_explanation': (face.ensemble or {}).get('explanation', ''),
                      'models': models_})
    evidence = _strip_artifacts(forensic.evidence_report if forensic else {})
    return {
        'report_version': REPORT_VERSION,
        'software_version': settings.DEEPSHIELD_VERSION,
        'analysis_id': str(up.public_id),
        'created_at': timezone.now().isoformat(),
        'file': {'sha256': up.sha256_hash, 'sha256_algorithm': 'SHA-256', 'size_bytes': up.file_size,
                 'width': (info.get('image') or {}).get('width'), 'height': (info.get('image') or {}).get('height')},
        'ai_detection': {'image_level': info.get('image_level'), 'detector': info.get('detector'),
                         'ensemble': info.get('ensemble_config')},
        'faces': faces,
        'model_versions': sorted(versions.values(), key=lambda v: (v['key'], v['version'])),
        'forensic_evidence': evidence,
        'assessment': (evidence.get('summary') or {}),
        'limitations': LIMITATIONS,
    }


def create_report(analysis) -> Report:
    content = build_report_content(analysis)
    digest = content_hash(content)
    report, _ = Report.objects.update_or_create(
        analysis=analysis,
        defaults={'content': content, 'content_sha256': digest, 'signature': sign(digest),
                  'software_version': settings.DEEPSHIELD_VERSION})
    return report


def verify_report(report: Report) -> dict:
    digest = content_hash(report.content)
    return {'hash_valid': hmac.compare_digest(digest, report.content_sha256),
            'signature_valid': bool(report.signature) and hmac.compare_digest(sign(report.content_sha256),
                                                                              report.signature)}
