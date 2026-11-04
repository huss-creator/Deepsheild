"""
Evaluate the six models on validation / test / external splits and fit the ensemble.

    python manage.py evaluate_models --manifest manifest.csv --out reports/eval1 --seed 42

Faces: rows are expected to be face crops (set --detect to run the face detector and use
the largest face instead). Writes evaluation_report.json, scores.csv and ensemble_config.json
(copy/point ENSEMBLE_CONFIG_PATH at it to activate the validated ensemble).
"""

import csv
import json
import os

import numpy as np
from django.core.management.base import BaseCommand, CommandError
from PIL import Image

from ai_models import evaluation as ev
from ai_models.face_detection import align_and_crop, get_face_detector
from ai_models.registry import MODEL_ORDER, get_registry
from ai_models.types import AVAILABLE


class Command(BaseCommand):
    help = 'Evaluate models, audit leakage, tune thresholds on validation, fit the ensemble.'

    def add_arguments(self, parser):
        parser.add_argument('--manifest', required=True)
        parser.add_argument('--out', required=True)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--detect', action='store_true', help='detect+crop the largest face first (all splits)')
        parser.add_argument('--models', default='', help='comma-separated model keys to evaluate and ensemble '
                            '(default: every AVAILABLE model), e.g. efficientnet_b4,xception,vit_b16')
        parser.add_argument('--detect-external', action='store_true',
                            help='detect+crop only for split=external (full frames, e.g. Celeb-DF); the other splits '
                                 'are already face crops')

    def handle(self, *a, manifest, out, seed, detect, detect_external=False, models='', **o):
        rows = ev.read_manifest(manifest)
        base = os.path.dirname(os.path.abspath(manifest))
        for r in rows:
            r['abs'] = r['path'] if os.path.isabs(r['path']) else os.path.join(base, r['path'])
            if not r.get('sha256'):
                r['sha256'] = ev.sha256_file(r['abs'])
        audit = ev.leakage_audit(rows)
        if not audit['ok']:
            raise CommandError(f"Data leakage detected ({audit['n_problems']} problems): "
                               f"{audit['problems'][:3]}. Fix the splits first.")
        registry = get_registry()
        dets = {d.spec.key: d for d in registry.ordered() if d.status == AVAILABLE}
        wanted = [m.strip() for m in models.split(',') if m.strip()]
        if wanted:
            unknown = [m for m in wanted if m not in MODEL_ORDER]
            absent = [m for m in wanted if m in MODEL_ORDER and m not in dets]
            if unknown or absent:
                raise CommandError(f'--models: unknown={unknown} not AVAILABLE={absent}. Run `manage.py check_models`.')
            dets = {k: d for k, d in dets.items() if k in wanted}
        if not dets:
            raise CommandError('No model is AVAILABLE. Run `manage.py check_models`.')
        face_det = get_face_detector() if (detect or detect_external) else None
        n_done = [0]; n_noface = [0]
        cache = {}

        def score_fn(row):
            n_done[0] += 1
            if n_done[0] % 200 == 0:
                self.stdout.write(f'  scored {n_done[0]} images (no face found: {n_noface[0]})')
            img = np.asarray(Image.open(row['abs']).convert('RGB'))
            if face_det and (detect or row['split'] == 'external'):
                faces = face_det.detect(img)
                if not faces:
                    n_noface[0] += 1
                    return {}
                img, _ = align_and_crop(img, faces[0])
            res = {k: d.predict(img).score for k, d in dets.items()}
            cache[row['abs']] = res
            return res

        keys = [k for k in MODEL_ORDER if k in dets]
        # thresholds are always tuned on the VALIDATION split (never taken from config) so metrics stay honest
        report = ev.run_evaluation(rows, score_fn, keys, seed=seed)
        # ensemble fit on validation (models available for all validation rows only)
        val = [r for r in rows if r['split'] == 'validation' and cache.get(r['abs']) and
               all(cache[r['abs']].get(k) is not None for k in keys)]
        tst = [r for r in rows if r['split'] == 'test' and cache.get(r['abs']) and
               all(cache[r['abs']].get(k) is not None for k in keys)]
        os.makedirs(out, exist_ok=True)
        try:
            cfg = ev.fit_ensemble({k: [cache[r['abs']][k] for r in val] for k in keys},
                                  [r['label'] for r in val], seed,
                                  {k: [cache[r['abs']][k] for r in tst] for k in keys} if tst else None,
                                  [r['label'] for r in tst] if tst else None)
            cfg['models'] = {k: dets[k].spec.metadata() for k in keys}
            report['ensemble'] = cfg
            with open(os.path.join(out, 'ensemble_config.json'), 'w') as fh:
                json.dump(cfg, fh, indent=2)
        except ValueError as exc:
            report['ensemble'] = {'error': str(exc)}
        # threshold-free extras (EER, TPR@5%FPR) per model/split, and the fitted ensemble vs the equal-mean baseline
        # on test and EXTERNAL data. Thresholds come from validation only; nothing is tuned on external data.
        def split_rows(sp):
            return [r for r in rows if r['split'] == sp and cache.get(r['abs'])
                    and all(cache[r['abs']].get(k) is not None for k in keys)]
        for k in keys:
            for sp, m in report['models'][k]['splits'].items():
                rr = [r for r in rows if r['split'] == sp and cache.get(r['abs']) and cache[r['abs']].get(k) is not None]
                m.update(ev.eer_and_tpr([r['label'] for r in rr], [cache[r['abs']][k] for r in rr]))
        ens = report.get('ensemble') or {}
        if ens and 'error' not in ens:
            ens['evaluation'] = {}
            for sp in ('validation', 'test', 'external'):
                rr = split_rows(sp)
                if len(rr) < 20:
                    continue
                y = [r['label'] for r in rr]
                sc = {k: [cache[r['abs']][k] for r in rr] for k in keys}
                fitted, mean = ev.apply_ensemble(ens, keys, sc), ev.apply_ensemble({'method': 'mean'}, keys, sc)
                mean_thr = ev.select_threshold([r['label'] for r in val],
                                               ev.apply_ensemble({'method': 'mean'}, keys,
                                                                 {k: [cache[r['abs']][k] for r in val] for k in keys}))
                ens['evaluation'][sp] = {
                    'fitted': {**ev.compute_metrics(y, fitted, ens['decision_threshold']), **ev.eer_and_tpr(y, fitted)},
                    'equal_mean': {**ev.compute_metrics(y, mean, mean_thr), **ev.eer_and_tpr(y, mean)}}
            with open(os.path.join(out, 'ensemble_config.json'), 'w') as fh:
                json.dump(ens, fh, indent=2)
        report['model_metadata'] = {k: dets[k].spec.metadata() for k in keys}
        with open(os.path.join(out, 'evaluation_report.json'), 'w') as fh:
            json.dump(report, fh, indent=2)
        with open(os.path.join(out, 'scores.csv'), 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['sha256', 'split', 'label', 'group_id'] + keys)
            for r in rows:
                if r['abs'] in cache:
                    w.writerow([r['sha256'], r['split'], r['label'], r.get('group_id', '')]
                               + [cache[r['abs']].get(k) for k in keys])
        self.stdout.write(self.style.SUCCESS(f'Report written to {out}'))
        for k, m in report['models'].items():
            t, e = m['splits'].get('test', {}), m['splits'].get('external', {})
            self.stdout.write(f"{k:<16} test AUC={t.get('roc_auc')}  external AUC={e.get('roc_auc')}")
        for sp, m in ((report.get('ensemble') or {}).get('evaluation') or {}).items():
            self.stdout.write(f"ensemble[{sp}] fitted AUC={m['fitted']['roc_auc']}  equal-mean AUC={m['equal_mean']['roc_auc']}")
