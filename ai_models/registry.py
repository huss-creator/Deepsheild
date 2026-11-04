"""
Model registry: turns configuration into ModelSpec objects and owns the loaded
detectors (one instance per process, loaded lazily and only once).

Nothing model-specific is hardcoded. Drop checkpoints into MODEL_DIR and describe
them in models.json (see trained_models/models.example.json).
"""

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from django.conf import settings


logger = logging.getLogger('deepshield.models')

# key -> (display name, family, typical input size, folder, env prefix)
ARCHITECTURES = {
    'xception':        ('XceptionNet',      'CNN',           299, 'xception',        'XCEPTION'),
    'efficientnet_b4': ('EfficientNet-B4',  'CNN',           380, 'efficientnet_b4', 'EFFICIENTNET'),
    'vit_b16':         ('ViT-B/16',         'Transformer',   224, 'vit_b16',         'VIT'),
    'freq_svm':        ('FFT/DCT + SVM',    'Frequency+SVM', 256, 'freq_svm',        'FREQUENCY'),
    'sdxl_detector':   ('SDXL Detector',   'Transformer',   224, 'sdxl_detector',   'SDXL'),
}
MODEL_ORDER = list(ARCHITECTURES)
SCOPES = ('faceswap', 'ai_generated', 'both')

EXT_FRAMEWORK = {'.onnx': 'onnx', '.pt': 'torch', '.pth': 'torch', '.h5': 'keras',
                 '.keras': 'keras', '.joblib': 'sklearn', '.pkl': 'sklearn'}
NORMALIZATIONS = ('zero_one', 'minus_one_one', 'imagenet', 'none')
OUTPUT_MODES = ('sigmoid', 'logit', 'softmax', 'logits')
PICKLE_FRAMEWORKS = ('sklearn',)   # arbitrary-code-on-load formats: checksum mandatory


@dataclass
class ModelSpec:
    key: str
    display_name: str
    family: str
    enabled: bool = True
    path: Optional[Path] = None
    framework: str = ''
    input_size: int = 224
    normalization: str = ''
    output_mode: str = ''
    fake_index: Optional[int] = None     # which output index means "manipulated"
    threshold: float = 0.5
    threshold_version: str = 'default'
    device: str = 'auto'
    precision: str = 'fp32'
    version: str = ''
    preprocessing_version: str = 'v1'
    interpolation: str = 'bilinear'
    sha256: str = ''
    num_outputs: int = 2                 # torch state_dict architectures only
    gradcam_layer: str = ''
    explain: dict = field(default_factory=dict)   # {'method','weights','sha256','layer'}: PyTorch copy used only for heat-maps
    feature_method: str = ''             # freq_svm only
    fake_label: Optional[int] = None     # freq_svm only (class label meaning fake)
    scope: str = 'both'                  # faceswap | ai_generated | both (training domain)
    training: dict = field(default_factory=dict)   # dataset, dataset_version, date, parameters, metrics

    def problems(self) -> List[str]:
        """Configuration errors that would make results silently wrong."""
        issues = []
        if self.framework not in ('onnx', 'torch', 'keras', 'sklearn'):
            issues.append('framework missing/unsupported')
        if self.key == 'freq_svm':
            if not self.feature_method:
                issues.append('feature_method is required')
            if self.fake_label is None:
                issues.append('fake_label is required')
            if not self.sha256:
                issues.append('sha256 is required for pickle-based models')
        else:
            if self.normalization not in NORMALIZATIONS:
                issues.append(f'normalization must be one of {NORMALIZATIONS}')
            if self.output_mode not in OUTPUT_MODES:
                issues.append(f'output_mode must be one of {OUTPUT_MODES}')
            if self.output_mode in ('softmax', 'logits') and self.fake_index is None:
                issues.append('fake_index is required for softmax/logits outputs')
        if self.framework in PICKLE_FRAMEWORKS and not self.sha256:
            issues.append('sha256 is required for pickle-based models')
        if getattr(settings, 'MODEL_REQUIRE_CHECKSUM', False) and not self.sha256:
            issues.append('sha256 is required (MODEL_REQUIRE_CHECKSUM)')
        if not self.version:
            issues.append('version is required')
        return issues

    def metadata(self) -> dict:
        """Reproducibility metadata recorded with every analysis result."""
        t = self.training or {}
        return {
            'model_name': self.display_name, 'model_version': self.version, 'scope': self.scope,
            'training_dataset': t.get('dataset', ''),
            'training_dataset_version': t.get('dataset_version', ''),
            'training_date': t.get('date', ''), 'framework': self.framework,
            'input_size': self.input_size, 'preprocessing_version': self.preprocessing_version,
            'threshold': self.threshold, 'threshold_version': self.threshold_version,
            'weights_checksum': self.sha256, 'training_parameters': t.get('parameters', {}),
            'evaluation_metrics': t.get('metrics', {}),
        }


def _read_json(path: str) -> dict:
    try:
        with open(path, 'r', encoding='utf-8-sig') as fh:   # utf-8-sig: Windows Notepad adds a BOM
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.error('model config file is unreadable or invalid JSON: %s', exc)
        return {}


def _env_bool(name: str) -> Optional[bool]:
    v = os.environ.get(name)
    return None if v is None else v.strip().lower() in ('1', 'true', 'yes', 'on')


def _discover_file(model_dir: Path, folder: str) -> Optional[Path]:
    d = model_dir / folder
    if d.is_dir():
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in EXT_FRAMEWORK:
                return f
    return None


def build_specs() -> Dict[str, ModelSpec]:
    model_dir = Path(settings.MODEL_DIR)
    cfg = _read_json(settings.MODEL_CONFIG_FILE)
    specs = {}
    for key, (name, family, default_size, folder, env_prefix) in ARCHITECTURES.items():
        c = dict(cfg.get(key, {}))
        enabled = c.get('enabled', True)
        env_enabled = _env_bool(f'{env_prefix}_ENABLED')
        if env_enabled is not None:
            enabled = env_enabled
        raw_path = os.environ.get(f'{env_prefix}_MODEL_PATH') or c.get('file')
        path = None
        if raw_path:
            p = Path(raw_path)
            path = p if p.is_absolute() else model_dir / p
        else:
            path = _discover_file(model_dir, folder)
        framework = c.get('framework') or (EXT_FRAMEWORK.get(path.suffix.lower(), '') if path else '')
        t = c.get('training') or {}
        specs[key] = ModelSpec(
            key=key, display_name=name, family=family, enabled=bool(enabled), path=path,
            framework=framework, input_size=int(c.get('input_size', default_size)),
            normalization=c.get('normalization', ''), output_mode=c.get('output_mode', ''),
            fake_index=c.get('fake_index'), threshold=float(c.get('threshold', 0.5)),
            threshold_version=str(c.get('threshold_version', 'default')),
            device=c.get('device', settings.DEVICE), precision=c.get('precision', 'fp32'),
            version=str(c.get('version', '')),
            preprocessing_version=str(c.get('preprocessing_version', 'v1')),
            interpolation=c.get('interpolation', 'bilinear'),
            sha256=str(c.get('sha256', '')).lower(), num_outputs=int(c.get('num_outputs', 2)),
            gradcam_layer=c.get('gradcam_layer', ''), explain=dict(c.get('explain') or {}), feature_method=c.get('feature_method', ''),
            fake_label=c.get('fake_label'), training=t,
            scope=c.get('scope', 'both') if c.get('scope', 'both') in SCOPES else 'both')
    return specs


class ModelRegistry:
    """Holds one detector per model key. Thread-safe lazy loading."""

    def __init__(self, specs: Optional[Dict[str, ModelSpec]] = None):
        from ai_models.detectors import create_detector   # deferred: heavy imports
        self._create = create_detector
        self.specs = specs if specs is not None else build_specs()
        self.detectors = {}
        self._lock = threading.Lock()
        self._loaded = False

    def load_all(self):
        with self._lock:
            if self._loaded:
                return
            for key in MODEL_ORDER:
                det = self._create(self.specs[key])
                det.load()
                self.detectors[key] = det
                logger.info('model %s status=%s', key, det.status)
            self._loaded = True

    def ordered(self):
        self.load_all()
        return [self.detectors[k] for k in MODEL_ORDER]

    def health(self) -> List[dict]:
        """Public-safe health summary (no filesystem paths)."""
        return [d.health_check() for d in self.ordered()]

    def available_count(self) -> int:
        return sum(1 for d in self.ordered() if d.status == 'AVAILABLE')


_registry: Optional[ModelRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> ModelRegistry:
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = ModelRegistry()
        return _registry


def reset_registry(registry: Optional[ModelRegistry] = None) -> None:
    """Replace/clear the process-wide registry (tests, admin model rollback)."""
    global _registry
    with _registry_lock:
        _registry = registry
