"""
Model monitoring from data the system already stores (no images, no user identifiers).

Per model version (key + version + weights checksum) over a time window:
  n, failure_rate, mean/std/percentiles of the model score, flagged_rate (share of successful runs
  that labelled FAKE/TAMPERED) and the same numbers for the PREVIOUS window of equal length.

A change in flagged_rate / mean score between windows is a *prompt to investigate*: it can mean model
degradation, a change in upload population, a new camera/codec, or a new manipulation family. It is
NOT proof the model got worse; without labels, accuracy cannot be measured here. Accuracy needs a
labelled sample — see docs/model_management.md.
"""

from datetime import timedelta

import numpy as np
from django.utils import timezone

from detector.models import ModelResult

MIN_N_FOR_DRIFT = 30
DRIFT_FLAGGED_DELTA = 0.15
DRIFT_MEAN_DELTA = 0.10


def _window_stats(qs):
    rows = list(qs.values_list('status', 'prediction', 'fake_probability'))
    n = len(rows)
    ok = [(p, s) for st, p, s in rows if st == ModelResult.STATUS_SUCCESS]
    failed = sum(1 for st, _, _ in rows if st == ModelResult.STATUS_ERROR)
    out = {'n': n, 'n_success': len(ok), 'failure_rate': round(failed / n, 4) if n else None}
    if ok:
        scores = np.array([s for _, s in ok], dtype=float)
        out.update(mean_score=round(float(scores.mean()), 4), std_score=round(float(scores.std()), 4),
                   p10=round(float(np.percentile(scores, 10)), 4), p50=round(float(np.percentile(scores, 50)), 4),
                   p90=round(float(np.percentile(scores, 90)), 4),
                   flagged_rate=round(sum(1 for p, _ in ok if p in ('FAKE', 'TAMPERED')) / len(ok), 4))
    return out


def summarise(days=7, now=None):
    now = now or timezone.now()
    cur_from, prev_from = now - timedelta(days=days), now - timedelta(days=2 * days)
    report = []
    keys = (ModelResult.objects.filter(created_at__gte=prev_from, model_version__isnull=False)
            .values_list('model_key', 'model_version__version', 'model_version__weights_sha256').distinct())
    for key, version, sha in sorted(keys):
        base = ModelResult.objects.filter(model_key=key, model_version__version=version,
                                          model_version__weights_sha256=sha)
        cur = _window_stats(base.filter(created_at__gte=cur_from, created_at__lte=now))
        prev = _window_stats(base.filter(created_at__gte=prev_from, created_at__lt=cur_from))
        alerts = []
        if cur['n'] >= MIN_N_FOR_DRIFT and prev['n'] >= MIN_N_FOR_DRIFT and cur['n_success'] and prev['n_success']:
            if abs(cur['flagged_rate'] - prev['flagged_rate']) >= DRIFT_FLAGGED_DELTA:
                alerts.append(f"flagged rate moved {prev['flagged_rate']:.2f} -> {cur['flagged_rate']:.2f}")
            if abs(cur['mean_score'] - prev['mean_score']) >= DRIFT_MEAN_DELTA:
                alerts.append(f"mean score moved {prev['mean_score']:.2f} -> {cur['mean_score']:.2f}")
        if cur['failure_rate'] and cur['failure_rate'] >= 0.05 and cur['n'] >= MIN_N_FOR_DRIFT:
            alerts.append(f"failure rate {cur['failure_rate']:.0%}")
        report.append({'model': key, 'version': version, 'weights_sha256': (sha or '')[:12],
                       'current': cur, 'previous': prev, 'alerts': alerts,
                       'enough_data': cur['n'] >= MIN_N_FOR_DRIFT and prev['n'] >= MIN_N_FOR_DRIFT})
    return report
