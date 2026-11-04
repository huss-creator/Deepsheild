"""
Ensemble policy: combines per-model manipulation scores into ONE image-face decision.

Design rules
  * Scores are RAW model outputs. Nothing is called a "probability" unless a
    validated ensemble config proves calibration (see evaluation.py).
  * Defaults are deliberately simple and UNVALIDATED (mean with SDXL counted double, 0.5
    threshold, +-0.10 uncertainty band). No invented per-model weights.
  * `manage.py evaluate_models` fits/validates a config on validation data and writes
    ensemble_config.json (method, weights/coefficients, threshold, calibration).
  * The decision can be UNCERTAIN: too few models, poor face quality, strong model
    disagreement, stacking inputs missing, or score inside the uncertainty band.

Labels (configurable constants below): AUTHENTIC, POTENTIALLY_MANIPULATED,
UNCERTAIN, NO_FACE, ANALYSIS_FAILED.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from django.conf import settings

from ai_models.types import AVAILABLE, ModelOutput

logger = logging.getLogger('deepshield.models')

AUTHENTIC = 'AUTHENTIC'
POTENTIALLY_MANIPULATED = 'POTENTIALLY_MANIPULATED'
UNCERTAIN = 'UNCERTAIN'
NO_FACE = 'NO_FACE'
ANALYSIS_FAILED = 'ANALYSIS_FAILED'

ENSEMBLE_SOFTWARE_VERSION = 'ensemble-3-domain'

DOMAIN_FACESWAP = 'faceswap'
DOMAIN_AI = 'ai_generated'
DOMAIN_TITLES = {DOMAIN_FACESWAP: 'Face-swap / deepfake check',
                 DOMAIN_AI: 'AI-generated image check'}


def _in_domain(o, domain: str) -> bool:
    return getattr(o, 'scope', 'both') in (domain, 'both')


def _domain_summary(live: List[ModelOutput], strong: float = 0.85) -> dict:
    """Per-domain view: which models are trained for it and how many of them flag."""
    res = {}
    for dom in (DOMAIN_FACESWAP, DOMAIN_AI):
        ms = [o for o in live if _in_domain(o, dom)]
        fl = [o for o in ms if o.score >= o.threshold]
        state = ('NO_MODELS' if not ms else 'CLEAR' if not fl
                 else 'FLAGGED' if (len(fl) * 2 > len(ms) or any(
                     o.scope != 'both' and o.score >= max(o.threshold, strong) for o in fl)) else 'PARTIAL')
        res[dom] = {'title': DOMAIN_TITLES[dom], 'models': len(ms), 'flagged': len(fl),
                    'state': state, 'flagged_names': [o.name for o in fl]}
    return res


PRIORITY_WEIGHT = 2.0     # weight of the SDXL detector in the default (unvalidated) mean


@dataclass
class EnsembleConfig:
    method: str = 'mean'                       # mean | weighted | stacking
    version: str = 'default-unvalidated-1'
    validated: bool = False
    calibrated: bool = False                   # True only if held-out ECE proved it
    decision_threshold: float = 0.5
    uncertainty_band: float = 0.10
    min_models: int = 3
    specialist_flag_score: float = 0.85       # a specialist this sure overrides the majority
    max_std: float = 0.30                      # score spread above this = disagreement
    disagreement_split: float = 0.34           # minority vote share that counts as a split
    weights: Dict[str, float] = field(default_factory=dict)
    coefficients: Dict[str, float] = field(default_factory=dict)   # stacking, on logit(score)
    intercept: float = 0.0
    validation: dict = field(default_factory=dict)


def load_ensemble_config(path: Optional[str] = None) -> EnsembleConfig:
    path = path or getattr(settings, 'ENSEMBLE_CONFIG_PATH', '')
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            raw = json.load(fh)
        fields = EnsembleConfig.__dataclass_fields__
        return EnsembleConfig(**{k: v for k, v in raw.items() if k in fields})
    except FileNotFoundError:
        return EnsembleConfig()
    except (OSError, ValueError, TypeError):
        logger.error('ensemble config invalid; using unvalidated defaults')
        return EnsembleConfig()


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _as_output(o) -> ModelOutput:
    return o if isinstance(o, ModelOutput) else ModelOutput(**o)


def combine(outputs: List, cfg: Optional[EnsembleConfig] = None,
            quality_flags: Optional[List[str]] = None, expected_models: Optional[int] = None) -> dict:
    cfg = cfg or load_ensemble_config()
    outs = [_as_output(o) for o in outputs]
    live = [o for o in outs if o.status == AVAILABLE and o.score is not None]
    expected = expected_models if expected_models is not None else len(outs)
    scores = np.array([o.score for o in live], dtype=float)
    votes_m = sum(1 for o in live if o.score >= o.threshold)
    votes_a = len(live) - votes_m
    reasons: List[str] = []
    score: Optional[float] = None

    if live:
        if cfg.method == 'weighted' and cfg.weights:
            w = np.array([cfg.weights.get(o.key, 0.0) for o in live], dtype=float)
            score = float(np.dot(w, scores) / w.sum()) if w.sum() > 0 else float(scores.mean())
        elif cfg.method == 'stacking':
            if all(k in cfg.coefficients for k in (o.key for o in live)) and \
                    len(live) == len(cfg.coefficients):
                z = cfg.intercept + sum(cfg.coefficients[o.key] * _logit(o.score) for o in live)
                score = 1.0 / (1.0 + math.exp(-z))
            else:
                reasons.append('stacking_inputs_missing')
        else:
            # SDXL detector has priority: its score counts PRIORITY_WEIGHT times as much as the others.
            w = np.array([PRIORITY_WEIGHT if o.key == 'sdxl_detector' else 1.0 for o in live], dtype=float)
            score = float(np.dot(w, scores) / w.sum())

    std = float(scores.std()) if len(scores) > 1 else 0.0

    # Domain awareness: a model only votes "authentic" for what it was trained to see. A strong
    # flag from a specialist (scope != both) drives the headline; models trained on the OTHER
    # domain are 'not applicable' instead of counting as authentic votes.
    domains = _domain_summary(live, cfg.specialist_flag_score)
    specialists = [o for o in live if getattr(o, 'scope', 'both') != 'both'
                   and o.score >= max(o.threshold, cfg.specialist_flag_score)]
    flagged_domains = {o.scope for o in specialists}
    na_votes = 0
    if specialists:
        na = [o for o in live if o.score < o.threshold and not any(_in_domain(o, d) for d in flagged_domains)]
        na_votes = len(na)
        votes_a -= na_votes
    disagreement = bool(len(live) > 1 and not specialists and
                        (std > cfg.max_std or min(votes_m, votes_a) / len(live) >= cfg.disagreement_split))

    if len(live) < cfg.min_models:
        reasons.append('insufficient_models')
    if quality_flags:
        reasons.append('poor_face_quality')
    if disagreement:
        reasons.append('model_disagreement')

    thr, band = cfg.decision_threshold, cfg.uncertainty_band
    if specialists and not ({'insufficient_models', 'poor_face_quality'} & set(reasons)):
        label = POTENTIALLY_MANIPULATED
        reasons = [r for r in reasons if r != 'stacking_inputs_missing']
        reasons.append('specialist_flag')
    elif reasons or score is None:
        label = UNCERTAIN
    elif score >= thr + band:
        label = POTENTIALLY_MANIPULATED
    elif score <= thr - band:
        label = AUTHENTIC
    else:
        label = UNCERTAIN
        reasons.append('score_in_uncertainty_band')

    score_kind = ('calibrated probability' if (cfg.calibrated and cfg.validated)
                  else 'prediction score (uncalibrated)')
    return {
        'label': label,
        'score': None if score is None else round(score, 4),
        'score_kind': score_kind,
        'method': cfg.method, 'ensemble_version': cfg.version,
        'software_version': ENSEMBLE_SOFTWARE_VERSION, 'validated': cfg.validated,
        'models_available': len(live), 'models_expected': expected,
        'manipulated_votes': votes_m, 'authentic_votes': votes_a, 'not_applicable_votes': na_votes,
        'domains': domains,
        'specialist_flags': [{'key': o.key, 'name': o.name, 'scope': o.scope} for o in specialists],
        'score_std': round(std, 4), 'disagreement': disagreement,
        'reasons': reasons, 'quality_flags': list(quality_flags or []),
        'explanation': _explain(label, score, score_kind, cfg, len(live), expected, reasons, specialists),
    }


_REASON_TEXT = {
    'insufficient_models': 'too few models produced a result',
    'poor_face_quality': 'the face is too small or partially visible for a reliable result',
    'model_disagreement': 'the models disagree with each other',
    'stacking_inputs_missing': 'a model required by the validated combiner is unavailable',
    'specialist_flag': 'a specialist model trained for this manipulation type flagged the image',
    'score_in_uncertainty_band': 'the score is close to the decision threshold',
}


def _explain(label, score, kind, cfg, n_live, n_expected, reasons, specialists=()) -> str:
    if n_live == 0:
        return (f'No AI model produced a result ({n_expected} configured/expected). '
                'No AI assessment is available for this face.')
    head = {AUTHENTIC: 'No strong manipulation indicators were found by the AI models.',
            POTENTIALLY_MANIPULATED: 'The AI models indicate possible manipulation.',
            UNCERTAIN: 'The AI result is uncertain'}[label]
    if label == UNCERTAIN and reasons:
        head += ': ' + '; '.join(_REASON_TEXT.get(r, r) for r in reasons) + '.'
    if specialists:
        head += (' Flagged by the specialist model(s) trained for this kind of manipulation: '
                 + ', '.join(o.name for o in specialists) + '. Models trained for a different kind of image are not counted as authentic votes.')
    text = (f'{head} {n_live} of {n_expected} models contributed'
            + (f'; ensemble {kind} {score:.2f}' if score is not None else '') + f' (method: {cfg.method}).')
    if not cfg.validated:
        text += ' The ensemble has not been validated on your data; treat this as a rough indication.'
    return text


def summarise_faces(face_ensembles: List[dict]) -> dict:
    """Image-level summary. Most severe face wins: any POTENTIALLY_MANIPULATED face makes the
    image potentially manipulated; else any UNCERTAIN; else AUTHENTIC. No faces -> NO_FACE."""
    if not face_ensembles:
        return {'label': NO_FACE, 'score': None, 'representative_face': None,
                'explanation': 'No detectable face was found in the image.'}
    order = {POTENTIALLY_MANIPULATED: 2, UNCERTAIN: 1, AUTHENTIC: 0}
    rep = max(range(len(face_ensembles)),
              key=lambda i: (order[face_ensembles[i]['label']], face_ensembles[i]['score'] or 0.0))
    label = face_ensembles[rep]['label']
    n = len(face_ensembles)
    text = face_ensembles[rep]['explanation']
    if n > 1:
        text = f'{n} faces analysed independently; the most severe result is shown. ' + text
    return {'label': label, 'score': face_ensembles[rep]['score'], 'representative_face': rep,
            'faces_analysed': n, 'explanation': text}


# ---------------------------------------------------------------------------
# Legacy compatibility (old API used by tests/templates until Phase 6/9)
# ---------------------------------------------------------------------------
MANIPULATION_LABELS = {'FAKE', 'TAMPERED', 'DEEPFAKE', 'MANIPULATED'}
AUTHENTIC_LABELS = {'REAL', 'ORIGINAL', 'AUTHENTIC', 'GENUINE'}
INVALID_LABELS = {'NOT_CONFIGURED', 'ERROR', 'UNCERTAIN'}


def normalize_prediction(raw_prediction):
    """Map any model label to FAKE / TAMPERED / REAL / UNCERTAIN / NOT_CONFIGURED / ERROR."""
    if not raw_prediction:
        return 'UNCERTAIN'
    p = raw_prediction.upper().strip()
    if p in ('FAKE', 'DEEPFAKE', 'MANIPULATED', 'ARTIFICIAL', 'GENERATED', 'SYNTHETIC', '1', 'FAKE_FACE'):
        return 'FAKE'
    if p in ('TAMPERED', 'FORGED', 'ALTERED', 'SPLICED', 'EDITED'):
        return 'TAMPERED'
    if p in ('REAL', 'GENUINE', 'AUTHENTIC', 'ORIGINAL', '0', 'REAL_FACE', 'UNTOUCHED', 'UNMODIFIED'):
        return 'REAL'
    if p in ('NOT_CONFIGURED', 'STANDBY', 'ERROR', 'UNCERTAIN'):
        return p
    logger.warning("normalize_prediction: unknown label '%s' -> UNCERTAIN", raw_prediction)
    return 'UNCERTAIN'


_LEGACY_LABEL = {AUTHENTIC: 'LIKELY_AUTHENTIC', POTENTIALLY_MANIPULATED: 'LIKELY_MANIPULATED',
                 UNCERTAIN: 'UNCERTAIN'}


def legacy_final_prediction(label: str) -> str:
    return _LEGACY_LABEL.get(label, label)


def calculate_ensemble_result(model_results, model_weights=None):
    """DEPRECATED wrapper kept for the old result shape. `model_weights` is ignored
    (weights now come from a validated EnsembleConfig). Uses min_models=1 so that the
    old semantics (any executed model counts) are preserved."""
    outs = []
    for r in model_results:
        ok = (r.get('status') in ('success', AVAILABLE)
              and normalize_prediction(r.get('prediction')) in ('FAKE', 'TAMPERED', 'REAL'))
        outs.append(ModelOutput(key=r.get('model_key', ''), name=r.get('model_name', ''),
                                status=AVAILABLE if ok else 'NOT_CONFIGURED',
                                score=float(r.get('fake_probability', 0.0)) if ok else None,
                                threshold=0.5))
    cfg = load_ensemble_config()
    cfg.min_models = 1
    res = combine(outs, cfg)
    score = res['score'] or 0.0
    conf = (1 - score) * 100 if res['label'] == AUTHENTIC else score * 100
    return {
        'final_prediction': legacy_final_prediction(res['label']),
        'overall_confidence': round(conf if res['models_available'] else 0.0, 2),
        'authentic_votes': res['authentic_votes'], 'manipulation_votes': res['manipulated_votes'],
        'executed_models': res['models_available'],
        'unavailable_models': len(outs) - res['models_available'],
        'explanation': res['explanation'],
    }
