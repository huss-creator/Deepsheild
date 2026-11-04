"""OPTIONAL. Heat-maps work without this (the deployed .onnx is converted automatically). Use it only if you have the
training notebooks' PyTorch state_dict and prefer to explain from those exact weights.

    python manage.py attach_explain xception=C:/path/xception_state_dict.pth vit_b16=... efficientnet_b4=...
    python manage.py attach_explain --method vit_b16=rollout xception=...        (choose a method)

Copies each file to trained_models/<model>/explain.pth, pins its SHA-256 in models.json (`explain` block) and verifies it
reproduces the deployed model's score on a probe image. A backup models.json.bak is written first.
"""
import json
import shutil
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from ai_models.detectors import file_sha256
from ai_models.explainers import ARCH, CANDIDATES, DEFAULT_METHOD
from ai_models.registry import get_registry, reset_registry


class Command(BaseCommand):
    help = 'Attach PyTorch .pth files (state_dict) used only to draw Grad-CAM style explanations.'

    def add_arguments(self, parser):
        parser.add_argument('pairs', nargs='+', help='key=path/to/state_dict.pth')
        parser.add_argument('--method', action='append', default=[], help='key=gradcam|gradcampp|rollout')

    def handle(self, *args, **o):
        cfg_path = Path(settings.MODEL_CONFIG_FILE)
        if not cfg_path.is_file():
            raise CommandError(f'{cfg_path} not found')
        cfg = json.loads(cfg_path.read_text(encoding='utf-8-sig'))
        methods = dict(m.split('=', 1) for m in o['method'])
        shutil.copy2(cfg_path, str(cfg_path) + '.bak')
        for pair in o['pairs']:
            key, _, src = pair.partition('=')
            if key not in ARCH or key not in cfg:
                raise CommandError(f'{key!r}: expected one of {sorted(set(ARCH) & set(cfg))}')
            src = Path(src.strip('"'))
            if not src.is_file():
                raise CommandError(f'{src} not found')
            method = methods.get(key, DEFAULT_METHOD[key])
            if method not in CANDIDATES[key]:
                raise CommandError(f'{key}: method must be one of {CANDIDATES[key]}')
            dst = Path(settings.MODEL_DIR) / key / 'explain.pth'
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            cfg[key]['explain'] = {'method': method, 'weights': f'{key}/explain.pth', 'sha256': file_sha256(dst)}
            self.stdout.write(f'{key}: attached ({method})')
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
        reset_registry()
        bad = 0
        for d in get_registry().ordered():
            if not d.spec.explain or d.status != 'AVAILABLE':
                continue
            from ai_models.explainers import CamExplainer
            ex = CamExplainer(d)
            try:
                ex._ensure()
                self.stdout.write(self.style.SUCCESS(f'  {d.spec.key}: weights match the deployed model (parity OK)'))
            except RuntimeError as exc:
                bad += 1
                self.stdout.write(self.style.ERROR(f'  {d.spec.key}: {exc}'))
        if bad:
            raise CommandError(f'{bad} model(s) failed verification; restore models.json.bak or fix the .pth')
