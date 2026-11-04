"""
Evaluation toolkit: metrics, leakage audit, threshold selection on VALIDATION data only,
and validated ensemble fitting. Used by `manage.py evaluate_models`.

Manifest CSV columns: path, label (real|fake|0|1), split (train|validation|test|external),
group_id (source video / identity id; REQUIRED to audit same-source leakage),
dataset (name; optional), sha256 (optional, computed if absent).
"""

import csv
import hashlib
import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold

SPLITS = ('train', 'validation', 'test', 'external')
FAKE_WORDS = {'fake', '1', 'manipulated', 'deepfake'}
REAL_WORDS = {'real', '0', 'authentic', 'original'}


def parse_label(raw: str) -> int:
    v = str(raw).strip().lower()
    if v in FAKE_WORDS:
        return 1
    if v in REAL_WORDS:
        return 0
    raise ValueError(f'unrecognised label {raw!r}')


def read_manifest(path: str) -> List[dict]:
    with open(path, newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r['label'] = parse_label(r['label'])
        r['split'] = r['split'].strip().lower()
        if r['split'] not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {r['split']!r}")
    return rows


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- leakage
def leakage_audit(rows: List[dict]) -> dict:
    """Detect exact-duplicate files and same-group (video/identity) leakage across splits.
    External data must share nothing with train/validation (it is never used for
    training or threshold tuning)."""
    problems = []
    by_hash, by_group = defaultdict(set), defaultdict(set)
    for r in rows:
        if r.get('sha256'):
            by_hash[r['sha256']].add(r['split'])
        if r.get('group_id'):
            by_group[r['group_id']].add(r['split'])
    for h, splits in by_hash.items():
        if len(splits) > 1:
            problems.append({'type': 'duplicate_file_across_splits', 'sha256': h[:16], 'splits': sorted(splits)})
    for g, splits in by_group.items():
        if len(splits) > 1:
            problems.append({'type': 'group_across_splits', 'group_id': g, 'splits': sorted(splits)})
    missing_groups = sum(1 for r in rows if not r.get('group_id'))
    warnings = []
    if missing_groups:
        warnings.append(f'{missing_groups} rows have no group_id: same-video/identity leakage cannot be '
                        'ruled out for them.')
    return {'ok': not problems, 'problems': problems[:50], 'n_problems': len(problems),
            'warnings': warnings}


# ---------------------------------------------------------------- metrics
def _ece(y, s, bins=10) -> float:
    y, s = np.asarray(y), np.asarray(s)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(s, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - s[m].mean())
    return float(e)


def compute_metrics(y_true, scores, threshold: float = 0.5) -> dict:
    y, s = np.asarray(y_true, dtype=int), np.asarray(scores, dtype=float)
    pred = (s >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    div = lambda a, b: round(a / b, 4) if b else None
    single = len(set(y.tolist())) < 2
    return {
        'n': int(len(y)), 'n_fake': int(y.sum()), 'n_real': int((1 - y).sum()), 'threshold': threshold,
        'accuracy': div(tp + tn, len(y)), 'precision': div(tp, tp + fp), 'recall': div(tp, tp + fn),
        'f1': div(2 * tp, 2 * tp + fp + fn), 'specificity': div(tn, tn + fp),
        'false_positive_rate': div(fp, fp + tn), 'false_negative_rate': div(fn, fn + tp),
        'roc_auc': None if single else round(float(roc_auc_score(y, s)), 4),
        'pr_auc': None if single else round(float(average_precision_score(y, s)), 4),
        'brier': round(float(brier_score_loss(y, np.clip(s, 0, 1))), 4),
        'ece': round(_ece(y, s), 4),
        'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
    }


def eer_and_tpr(y_true, scores, target_fpr: float = 0.05) -> dict:
    """Threshold-free extras: equal error rate and TPR at a fixed false-positive rate (read off the ROC curve)."""
    y, s = np.asarray(y_true, dtype=int), np.asarray(scores, dtype=float)
    if len(set(y.tolist())) < 2:
        return {'eer': None, 'tpr_at_fpr': None, 'target_fpr': target_fpr}
    fpr, tpr, _ = roc_curve(y, s)
    i = int(np.nanargmin(np.abs((1 - tpr) - fpr)))
    ok = np.where(fpr <= target_fpr)[0]
    return {'eer': round(float((fpr[i] + (1 - tpr[i])) / 2), 4),
            'tpr_at_fpr': round(float(tpr[ok].max()), 4) if len(ok) else 0.0, 'target_fpr': target_fpr}


def apply_ensemble(cfg: dict, keys: List[str], scores: Dict[str, List[float]]) -> np.ndarray:
    """Score rows with a fitted ensemble config (mean, or logistic stacking on logits)."""
    X = np.column_stack([scores[k] for k in keys])
    if cfg.get('method') == 'stacking':
        w = np.array([cfg['coefficients'][k] for k in keys])
        return 1 / (1 + np.exp(-(cfg.get('intercept', 0.0) + _logit(X) @ w)))
    return X.mean(axis=1)


def curves(y_true, scores, max_points: int = 200) -> dict:
    y, s = np.asarray(y_true, dtype=int), np.asarray(scores, dtype=float)
    if len(set(y.tolist())) < 2:
        return {}
    def thin(a, b):
        step = max(1, len(a) // max_points)
        return [[round(float(x), 4), round(float(z), 4)] for x, z in zip(a[::step], b[::step])]
    fpr, tpr, _ = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)
    return {'roc': thin(fpr, tpr), 'precision_recall': thin(rec, prec)}


def select_threshold(y_val, s_val, method: str = 'youden', target_fpr: float = 0.05) -> float:
    """Pick a threshold from VALIDATION data only."""
    y, s = np.asarray(y_val, dtype=int), np.asarray(s_val, dtype=float)
    if len(set(y.tolist())) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, s)
    if method == 'fpr':
        ok = np.where(fpr <= target_fpr)[0]
        i = ok[np.argmax(tpr[ok])]
    else:
        i = int(np.argmax(tpr - fpr))
    t = float(thr[i])
    return 0.5 if not math.isfinite(t) else round(min(max(t, 0.0), 1.0), 4)


# --------------------------------------------------------------- ensemble
def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_ensemble(val_scores: Dict[str, List[float]], y_val, seed: int = 42,
                 test_scores: Optional[Dict[str, List[float]]] = None, y_test=None) -> dict:
    """Compare equal-weight mean vs logistic stacking by cross-validated log-loss on the
    validation split; choose the simpler mean unless stacking is clearly better. Returns an
    ensemble_config dict. `calibrated` is set only if held-out TEST ECE <= 0.05."""
    keys = sorted(val_scores)
    X = np.column_stack([val_scores[k] for k in keys])
    y = np.asarray(y_val, dtype=int)
    if len(y) < 20 or len(set(y.tolist())) < 2:
        raise ValueError('Need >=20 validation samples with both classes to fit an ensemble.')
    n_splits = int(min(5, np.bincount(y).min()))
    skf = StratifiedKFold(n_splits=max(n_splits, 2), shuffle=True, random_state=seed)
    ll_mean, ll_stack = [], []
    for tr, te in skf.split(X, y):
        m = np.clip(X[te].mean(axis=1), 1e-6, 1 - 1e-6)
        ll_mean.append(float(-np.mean(y[te] * np.log(m) + (1 - y[te]) * np.log(1 - m))))
        lr = LogisticRegression(C=1.0, random_state=seed, max_iter=1000).fit(_logit(X[tr]), y[tr])
        p = np.clip(lr.predict_proba(_logit(X[te]))[:, 1], 1e-6, 1 - 1e-6)
        ll_stack.append(float(-np.mean(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p))))
    cv = {'mean_logloss': round(float(np.mean(ll_mean)), 4), 'stacking_logloss': round(float(np.mean(ll_stack)), 4)}
    use_stack = cv['stacking_logloss'] < 0.95 * cv['mean_logloss']       # must be clearly better
    cfg = {'method': 'mean', 'validated': True, 'calibrated': False, 'cv': cv,
           'version': datetime.now(timezone.utc).strftime('ens-%Y%m%d-%H%M%S'),
           'validation': {'n': int(len(y)), 'seed': seed, 'models': keys,
                          'date': datetime.now(timezone.utc).isoformat()}}
    if use_stack:
        lr = LogisticRegression(C=1.0, random_state=seed, max_iter=1000).fit(_logit(X), y)
        cfg.update(method='stacking', coefficients={k: round(float(c), 6) for k, c in zip(keys, lr.coef_[0])},
                   intercept=round(float(lr.intercept_[0]), 6))
        val_final = lr.predict_proba(_logit(X))[:, 1]
    else:
        val_final = X.mean(axis=1)
    cfg['decision_threshold'] = select_threshold(y, val_final)
    if test_scores is not None and y_test is not None and len(y_test) >= 20:
        Xt = np.column_stack([test_scores[k] for k in keys])
        ft = (1 / (1 + np.exp(-(cfg.get('intercept', 0) + _logit(Xt) @ np.array(
            [cfg['coefficients'][k] for k in keys])))) if use_stack else Xt.mean(axis=1))
        ece = _ece(y_test, ft)
        cfg['validation']['test_ece'] = round(ece, 4)
        cfg['calibrated'] = bool(use_stack and ece <= 0.05)
    return cfg


# ------------------------------------------------------------------- driver
def run_evaluation(rows: List[dict], score_fn: Callable[[dict], Dict[str, Optional[float]]],
                   model_keys: List[str], thresholds: Optional[Dict[str, float]] = None,
                   seed: int = 42) -> dict:
    """score_fn(row) -> {model_key: score or None}. Thresholds are tuned on validation only,
    then applied unchanged to test and external."""
    np.random.seed(seed)
    scores = {k: {sp: ([], []) for sp in SPLITS} for k in model_keys}   # (y, s)
    for r in rows:
        if r['split'] == 'train':
            continue
        out = score_fn(r)
        for k in model_keys:
            if out.get(k) is not None:
                scores[k][r['split']][0].append(r['label'])
                scores[k][r['split']][1].append(out[k])
    report = {'seed': seed, 'created': datetime.now(timezone.utc).isoformat(),
              'leakage_audit': leakage_audit(rows), 'models': {}, 'ensemble': None}
    for k in model_keys:
        yv, sv = scores[k]['validation']
        thr = (thresholds or {}).get(k) or (select_threshold(yv, sv) if sv else 0.5)
        entry = {'threshold_tuned_on': 'validation', 'threshold': thr, 'splits': {}}
        for sp in ('validation', 'test', 'external'):
            y, s = scores[k][sp]
            if y:
                entry['splits'][sp] = {**compute_metrics(y, s, thr), 'curves': curves(y, s)}
        t, e = entry['splits'].get('test'), entry['splits'].get('external')
        if t and e and t.get('roc_auc') is not None and e.get('roc_auc') is not None:
            entry['generalisation_gap_auc'] = round(t['roc_auc'] - e['roc_auc'], 4)
        report['models'][k] = entry
    return report
