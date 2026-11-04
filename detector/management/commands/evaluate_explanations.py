"""Which explanation method is the most faithful for each model?  (deletion test with a random-mask control)

    python manage.py evaluate_explanations --images D:/faces_test --n 60 --out reports/explain

For each model (explained straight from its deployed ONNX; no extra files) and each candidate method (Grad-CAM, Grad-CAM++, + attention rollout for
ViT): take the model's top-20 % most important cells, blur them out, and measure how much the *deployed* model's
'manipulated' score falls.  Control: blur the same number of random cells.  A method is faithful if its drop is
clearly larger than the random drop.  Only images the model itself calls manipulated are used (deletion is
meaningless on images it already calls authentic).  Writes explain_report.json / .md and prints the recommendation.
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np
from django.core.management.base import BaseCommand, CommandError

from ai_models.explainers import CANDIDATES, cam_explainer_for
from ai_models.registry import get_registry

EXT = {'.jpg', '.jpeg', '.png', '.webp'}


def _blur(img):
    return cv2.GaussianBlur(img, (0, 0), max(img.shape[:2]) / 8)


def _apply(img, blurred, cell_mask):
    m = cv2.resize(cell_mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return np.where(m[..., None], blurred, img)


class Command(BaseCommand):
    help = 'Pick the most faithful explanation method per model (deletion test vs random control).'

    def add_arguments(self, p):
        p.add_argument('--images', help='folder of face crops (searched recursively)')
        p.add_argument('--manifest', help='CSV with a `path` column (optional `split`; only split==test is used)')
        p.add_argument('--n', type=int, default=60, help='max images per model')
        p.add_argument('--fraction', type=float, default=0.2)
        p.add_argument('--random-repeats', type=int, default=5)
        p.add_argument('--models', default='')
        p.add_argument('--out', default='explain_report')
        p.add_argument('--seed', type=int, default=42)

    def _images(self, o):
        if o['manifest']:
            import pandas as pd
            df = pd.read_csv(o['manifest'])
            if 'split' in df:
                df = df[df.split.isin(['test', 'validation', 'val'])]
            paths = df['path'].tolist()
        elif o['images']:
            paths = [str(x) for x in Path(o['images']).rglob('*') if x.suffix.lower() in EXT]
        else:
            raise CommandError('give --images or --manifest')
        rng = np.random.default_rng(o['seed'])
        rng.shuffle(paths)
        return paths

    def handle(self, *a, **o):
        paths = self._images(o)
        wanted = [m for m in o['models'].split(',') if m]
        rng = np.random.default_rng(o['seed'])
        report = {}
        for det in get_registry().ordered():
            key = det.spec.key
            ex = cam_explainer_for(det)
            if det.status != 'AVAILABLE' or ex is None or (wanted and key not in wanted):
                continue
            try:
                ex._ensure()
            except RuntimeError as exc:
                self.stdout.write(self.style.ERROR(f'{key}: {exc}'))
                continue
            faces = []
            for p in paths:
                img = cv2.imread(p)
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                out = det.predict(rgb)
                if out.status == 'AVAILABLE' and out.score >= det.spec.threshold:
                    faces.append((rgb, out.score))
                if len(faces) >= o['n']:
                    break
            self.stdout.write(f'{key}: {len(faces)} images the model calls manipulated')
            rows = {}
            for method in CANDIDATES[key]:
                gains, wins, drops, rdrops = [], 0, [], []
                for rgb, base in faces:
                    cam = ex.cam(rgb, method)
                    k = max(1, math.ceil(o['fraction'] * cam.size))
                    blurred = _blur(rgb)
                    top = np.zeros(cam.size, bool)
                    top[np.argsort(cam.ravel())[-k:]] = True
                    d_cam = base - det.predict(_apply(rgb, blurred, top.reshape(cam.shape))).score
                    d_rand = []
                    for _ in range(o['random_repeats']):
                        m = np.zeros(cam.size, bool)
                        m[rng.choice(cam.size, k, replace=False)] = True
                        d_rand.append(base - det.predict(_apply(rgb, blurred, m.reshape(cam.shape))).score)
                    drops.append(d_cam); rdrops.append(float(np.mean(d_rand)))
                    gains.append(d_cam - rdrops[-1]); wins += d_cam > rdrops[-1]
                if not faces:
                    continue
                g = np.array(gains)
                rows[method] = {'n': len(faces), 'mean_drop': round(float(np.mean(drops)), 4),
                                'mean_random_drop': round(float(np.mean(rdrops)), 4), 'mean_gain': round(float(g.mean()), 4),
                                'ci95': round(float(1.96 * g.std(ddof=1) / math.sqrt(len(g))), 4) if len(g) > 1 else None,
                                'win_rate': round(wins / len(faces), 3)}
                self.stdout.write(f'  {method:10s} drop={rows[method]["mean_drop"]:.3f}  random={rows[method]["mean_random_drop"]:.3f}  '
                                  f'gain={rows[method]["mean_gain"]:+.3f}±{rows[method]["ci95"]}  wins={rows[method]["win_rate"]:.0%}')
            if rows:
                best = max(rows, key=lambda m: rows[m]['mean_gain'])
                report[key] = {'methods': rows, 'recommended': best,
                               'faithful': rows[best]['mean_gain'] > (rows[best]['ci95'] or 0)}
                self.stdout.write(self.style.SUCCESS(f'  -> best for {key}: {best}'))
        out = Path(o['out']); out.mkdir(parents=True, exist_ok=True)
        (out / 'explain_report.json').write_text(json.dumps(report, indent=2))
        lines = ['| model | method | mean drop | random drop | gain | win rate | |', '|---|---|---|---|---|---|---|']
        for k, v in report.items():
            for m, r in v['methods'].items():
                lines.append(f'| {k} | {m} | {r["mean_drop"]} | {r["mean_random_drop"]} | {r["mean_gain"]:+} ± {r["ci95"]} | {r["win_rate"]:.0%} | {"**best**" if m == v["recommended"] else ""} |')
        (out / 'explain_report.md').write_text('\n'.join(lines) + '\n\nDeletion test: top-20% cells blurred; drop = fall of the deployed model\'s manipulated score; gain = drop - random-mask drop.\n')
        self.stdout.write(f'written to {out}/')
