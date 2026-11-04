"""
forensics/evidence.py — transparent evidence aggregation.

There is NO weighted "risk score" built from unvalidated components. Instead:

  * every forensic test yields an INDICATOR: status, severity, description, limitations,
    reliability and an artifact reference — never a probability;
  * five categories are kept separate: AI detection, forensic indicators, provenance,
    file integrity, explainability;
  * the final assessment follows explicit, readable rules (see `assess`) and always lists
    its reasoning. Forensic indicators alone never establish manipulation.

Legacy columns (evidence_* and overall_risk*) are still filled for the existing UI:
  evidence_* = display value of that test's LEVEL (0 / 50 / 100), not a measurement;
  overall_risk_score = percentage of applicable indicators that were flagged (a count, NOT a
  probability) — the UI will be relabelled in Phase 9.
"""

import logging
from datetime import datetime, timezone

from forensics.catalog import CATALOG

logger = logging.getLogger('deepshield.forensics')

EVIDENCE_VERSION = 'evidence-2'

# configurable result labels
LABELS = {
    'AUTHENTIC': 'AUTHENTIC / NO STRONG MANIPULATION INDICATORS',
    'POTENTIALLY_MANIPULATED': 'POTENTIALLY MANIPULATED',
    'UNCERTAIN': 'UNCERTAIN',
    'NO_FACE': 'NO FACE DETECTED',
    'FAILED': 'ANALYSIS FAILED',
}

LEVEL_TO_DISPLAY = {'LOW': 0.0, 'MEDIUM': 50.0, 'HIGH': 100.0, 'UNAVAILABLE': 0.0, 'N/A': 0.0}
STATISTICAL = ('ela', 'frequency', 'noise', 'copy_move', 'resampling', 'face_boundary')
_SEVERITY_ORDER = ['none', 'info', 'low', 'moderate']


def _cap(severity, max_severity):
    order = _SEVERITY_ORDER
    return order[min(order.index(severity), order.index(max_severity if max_severity in order else 'low'))]


def _indicator(key, level, description, artifact=None, details=None, applicable=True):
    cat = CATALOG[key]
    if level in (None, 'UNAVAILABLE'):
        status, severity = ('not_applicable' if not applicable else 'unavailable'), 'none'
    elif level == 'LOW':
        status, severity = 'no_indication', 'none'
    else:   # MEDIUM / HIGH: reported, but severity is capped by the test's measured reliability
        status = 'indication'
        severity = _cap('moderate' if level == 'HIGH' else 'low', cat['max_severity'])
    return {'name': cat['name'], 'status': status, 'severity': severity, 'level': level,
            'confidence': None,                       # no calibrated confidence exists for these tests
            'reliability': cat['reliability'], 'description': description,
            'limitations': cat['cannot_prove'], 'false_positive_risk': cat['false_positive'],
            'false_negative_risk': cat['false_negative'], 'artifact': artifact, 'details': details or {}}


def assess(ai_label, indicators, models_available=0):
    """Return (assessment_key, reasoning list). Rules, in order:
       1 NO_FACE            -> NO_FACE (forensics still reported)
       2 AI potentially manipulated -> POTENTIALLY_MANIPULATED
       3 AI authentic       -> AUTHENTIC unless a verified copy-move or >=2 other indications conflict
       4 no AI verdict      -> UNCERTAIN (forensic indicators alone never decide)"""
    flagged = [k for k, v in indicators.items() if v['status'] == 'indication']
    names = ', '.join(indicators[k]['name'] for k in flagged)
    if ai_label == 'NO_FACE':
        r = ['No detectable face, so no AI face assessment was possible.']
        if flagged:
            r.append(f'Image-level indicators reported: {names}.')
        return 'NO_FACE', r
    if ai_label == 'ANALYSIS_FAILED':
        return 'FAILED', ['The analysis could not be completed.']
    if ai_label == 'POTENTIALLY_MANIPULATED':
        r = ['The AI ensemble indicates possible manipulation.']
        r.append(f'Supporting forensic indicators: {names}.' if flagged else
                 'No forensic indicator corroborates this; the AI result stands alone.')
        return 'POTENTIALLY_MANIPULATED', r
    if ai_label == 'AUTHENTIC':
        conflict = 'copy_move' in flagged or len([k for k in flagged if k != 'copy_move']) >= 2
        if conflict:
            return 'UNCERTAIN', ['The AI found no strong manipulation indicators, but forensic indicators '
                                 f'disagree: {names}.']
        r = ['The AI models found no strong manipulation indicators.']
        if flagged:
            r.append(f'One weak forensic indicator was noted ({names}); it is supporting context only.')
        return 'AUTHENTIC', r
    r = ['The AI result is uncertain or unavailable; forensic indicators alone cannot establish manipulation '
         'or authenticity.']
    if models_available == 0:
        r.append('No AI model produced a result for this face.')
    if flagged:
        r.append(f'Indicators reported: {names}.')
    return 'UNCERTAIN', r


def build_evidence_report(ai_info, forensic, sha256='', file_size=None, explainability=None):
    """ai_info: Analysis.pipeline_info['image_level'] merged with the representative face ensemble
    (keys: label, score, score_kind, validated, models_available, models_expected, disagreement,
    reasons, explanation). forensic: dict of module result dicts keyed ela/frequency/noise/copy_move/
    resampling/face_boundary/metadata/c2pa."""
    ind = {}
    specs = [('ela', 'ela_level', 'ela_interpretation', 'ela_image_path'),
             ('frequency', 'freq_level', 'freq_interpretation', 'freq_image_path'),
             ('noise', 'noise_level', 'noise_interpretation', 'noise_image_path'),
             ('copy_move', 'copy_move_level', 'copy_move_interpretation', 'copy_move_image_path'),
             ('resampling', 'resampling_level', 'resampling_interpretation', 'resampling_image_path'),
             ('face_boundary', 'face_boundary_level', 'face_boundary_interpretation', 'face_boundary_image_path')]
    for key, lvl, interp, art in specs:
        res = forensic.get(key) or {}
        level = res.get(lvl, 'UNAVAILABLE')
        applicable = not (key == 'ela' and (res.get('details') or {}).get('applicable') is False)
        ind[key] = _indicator(key, level, res.get(interp, '') or 'Not available.', res.get(art),
                              res.get('details'), applicable)

    c2 = forensic.get('c2pa') or {}
    cstatus = c2.get('c2pa_status', 'ERROR')
    cmap = {'VALID': ('credential_valid', 'A valid Content Credential was found.'),
            'INVALID': ('credential_invalid', 'A Content Credential was found but is invalid or unreadable.'),
            'NOT_PRESENT': ('no_credential', 'No valid Content Credentials were found.'),
            'LIBRARY_MISSING': ('unavailable', 'Content Credentials could not be checked (verifier not installed).'),
            'ERROR': ('unavailable', 'Content Credentials check failed.')}
    cs, ctext = cmap.get(cstatus, cmap['ERROR'])
    if cstatus == 'VALID' and c2.get('c2pa_signer_trusted'):
        cs = 'credential_verified'
    provenance = {'c2pa': {'name': CATALOG['c2pa']['name'], 'status': cs, 'description': ctext,
                           'tool': c2.get('c2pa_claim_generator', ''), 'signer_trusted': bool(c2.get('c2pa_signer_trusted')),
                           'limitations': CATALOG['c2pa']['cannot_prove']},
                  'metadata': {'name': CATALOG['metadata']['name'],
                               'status': 'present' if (forensic.get('metadata') or {}).get('meta_has_exif') else 'limited_or_absent',
                               'editing_software_tag': (forensic.get('metadata') or {}).get('meta_editing_software_name', ''),
                               'description': (forensic.get('metadata') or {}).get('meta_note', ''),
                               'limitations': CATALOG['metadata']['cannot_prove']}}
    integrity = {'algorithm': 'SHA-256', 'sha256': sha256, 'file_size_bytes': file_size,
                 'computed_at': datetime.now(timezone.utc).isoformat(),
                 'limitations': CATALOG['sha256']['cannot_prove']}
    expl = explainability or {'available': False, 'methods': []}
    expl = {**expl, 'limitations': 'Visualization indicates regions influencing the model prediction; '
                                   'it is not proof of manipulation.'}

    ai = dict(ai_info or {})
    key, reasoning = assess(ai.get('label', 'UNCERTAIN'), ind, ai.get('models_available', 0))
    return {
        'version': EVIDENCE_VERSION,
        'ai_detection': ai,
        'forensic_indicators': ind,
        'provenance': provenance,
        'integrity': integrity,
        'explainability': expl,
        'summary': {'assessment': key, 'label': LABELS[key], 'reasoning': reasoning,
                    'limitations': ['Deepfake detection is probabilistic; false positives and negatives occur.',
                                    'Forensic indicators are supporting evidence, never proof.',
                                    'No single indicator establishes authenticity or manipulation.']},
    }


# ---------------------------------------------------------------- legacy columns
def calculate_evidence_scores(ai_confidence, ai_prediction, ela_level, freq_level, noise_level,
                              meta_has_editing_sw, meta_has_exif, meta_format,
                              copy_move_level='UNAVAILABLE', face_boundary_level='UNAVAILABLE',
                              resampling_level='UNAVAILABLE'):
    """Fill the legacy evidence_* / overall_risk* columns from the same transparent rules.
    Missing metadata and editing-software tags do NOT contribute (they are context, not indicators)."""
    levels = {'ela': ela_level, 'frequency': freq_level, 'noise': noise_level, 'copy_move': copy_move_level,
              'resampling': resampling_level, 'face_boundary': face_boundary_level}
    applicable = [k for k, v in levels.items() if v not in ('UNAVAILABLE', 'N/A', None)]
    flagged = [k for k in applicable if levels[k] in ('MEDIUM', 'HIGH')]
    ai_label = {'LIKELY_MANIPULATED': 'POTENTIALLY_MANIPULATED', 'LIKELY_AUTHENTIC': 'AUTHENTIC',
                'NO_FACE': 'NO_FACE'}.get(ai_prediction, 'UNCERTAIN')
    if ai_label == 'POTENTIALLY_MANIPULATED':
        risk = 'HIGH' if flagged else 'MEDIUM'
    elif ai_label == 'AUTHENTIC':
        risk = 'MEDIUM' if ('copy_move' in flagged or len(flagged) >= 2) else 'LOW'
    else:
        risk = 'MEDIUM' if len(flagged) >= 2 else 'LOW'
    ratio = round(100.0 * len(flagged) / len(applicable), 2) if applicable else 0.0
    ai_val = 0.0
    if ai_label == 'POTENTIALLY_MANIPULATED':
        ai_val = min(100.0, float(ai_confidence))
    elif ai_label == 'AUTHENTIC':
        ai_val = max(0.0, 100.0 - float(ai_confidence))
    disp = lambda lv: LEVEL_TO_DISPLAY.get(lv, 0.0)
    text = (f'AI result: {ai_label.replace("_", " ").lower()}. {len(flagged)} of {len(applicable)} applicable '
            f'forensic indicator(s) flagged' + (f' ({", ".join(flagged)})' if flagged else '') +
            '. The percentage shown is a count of flagged indicators, not a probability. '
            'Indicators are supporting evidence only and none proves manipulation.')
    return {'evidence_ai': round(ai_val, 2), 'evidence_ela': disp(ela_level), 'evidence_freq': disp(freq_level),
            'evidence_noise': disp(noise_level), 'evidence_meta': 0.0,
            'evidence_copy_move': disp(copy_move_level), 'evidence_boundary': disp(face_boundary_level),
            'evidence_resampling': disp(resampling_level), 'overall_risk_score': ratio,
            'overall_risk': risk, 'risk_interpretation': text}
