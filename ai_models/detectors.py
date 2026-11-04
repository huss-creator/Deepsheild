"""
Detector abstraction. Every model exposes the same lifecycle:

    load()          verify file/checksum, load weights, run a self-test
    preprocess()    face crop (RGB uint8) -> model input
    predict()       full inference -> ModelOutput (never raises)
    postprocess()   raw model output -> manipulation score in [0, 1]
    explain()       explanation artefact where technically supported
    health_check()  public-safe status (no filesystem paths)

Backends: ONNX (framework-agnostic), PyTorch (TorchScript or state_dict), Keras,
and scikit-learn (FFT/DCT + SVM). A failing model never crashes the service; it
reports FAILED/TIMEOUT/DISABLED/NOT_CONFIGURED and the ensemble policy decides.
Models are loaded once per process and are read-only during inference, so they
are safe to share between threads.
"""

import hashlib
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from django.conf import settings

from ai_models.registry import ModelSpec
from ai_models.types import (AVAILABLE, DISABLED, FAILED, LABEL_AUTHENTIC, LABEL_MANIPULATED,
                             NOT_CONFIGURED, ModelOutput)

logger = logging.getLogger('deepshield.models')

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_INTERP = {'bilinear': cv2.INTER_LINEAR, 'bicubic': cv2.INTER_CUBIC, 'area': cv2.INTER_AREA,
           'nearest': cv2.INTER_NEAREST, 'lanczos': cv2.INTER_LANCZOS4}


class ModelConfigError(Exception):
    """Configuration/loading problem whose message is safe to show to users."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def resolve_device(requested: str) -> str:
    """'auto' -> cuda if a GPU is usable else cpu. Never raises if CUDA is missing."""
    requested = (requested or 'auto').lower()
    if requested == 'cpu':
        return 'cpu'
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    if requested == 'cuda' and not has_cuda:
        logger.warning('CUDA requested but unavailable; falling back to CPU')
        return 'cpu'
    return 'cuda' if has_cuda else 'cpu'


class BaseDetector:
    layout = 'NHWC'                      # tensor layout this backend feeds the model
    score_type = 'uncalibrated model output'

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.status = NOT_CONFIGURED
        self.message = ''
        self.device = 'cpu'
        self._model = None

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        spec = self.spec
        if not spec.enabled:
            self.status, self.message = DISABLED, 'Disabled in configuration.'
            return
        if spec.path is None:
            self.status, self.message = NOT_CONFIGURED, 'No model file has been provided.'
            return
        try:
            issues = spec.problems()
            if issues:
                raise ModelConfigError('Invalid model configuration: ' + '; '.join(issues))
            path = Path(spec.path)
            model_dir = Path(settings.MODEL_DIR).resolve()
            if model_dir not in path.resolve().parents:
                raise ModelConfigError('Model file must be located inside MODEL_DIR.')
            if not path.is_file():
                raise ModelConfigError('Model file not found.')
            if spec.sha256 and file_sha256(path) != spec.sha256:
                raise ModelConfigError('Model checksum does not match the configured SHA-256.')
            self.device = resolve_device(spec.device)
            self._load_backend(path)
            self._self_test()
            self.status, self.message = AVAILABLE, ''
        except ModelConfigError as exc:
            self.status, self.message = FAILED, str(exc)
            logger.error('model %s failed to load: %s', spec.key, exc)
        except Exception:
            self.status, self.message = FAILED, 'Model failed to load (see server log).'
            logger.exception('model %s failed to load', spec.key)

    def _load_backend(self, path: Path) -> None:          # pragma: no cover - abstract
        raise NotImplementedError

    def _forward(self, batch: np.ndarray) -> np.ndarray:   # pragma: no cover - abstract
        raise NotImplementedError

    def _self_test(self) -> None:
        """Dry run on a blank face: verifies shapes and output semantics at startup."""
        size = self.spec.input_size
        score = self.postprocess(self._forward(self.preprocess(np.zeros((size, size, 3), np.uint8))))
        if not (0.0 <= score <= 1.0):
            raise ModelConfigError('Model output is outside [0, 1].')

    # ------------------------------------------------------------ inference
    def preprocess(self, face_rgb: np.ndarray) -> np.ndarray:
        spec = self.spec
        size = spec.input_size
        interp = _INTERP.get(spec.interpolation, cv2.INTER_LINEAR)
        x = cv2.resize(face_rgb, (size, size), interpolation=interp).astype(np.float32)
        if spec.normalization == 'zero_one':
            x = x / 255.0
        elif spec.normalization == 'minus_one_one':
            x = x / 127.5 - 1.0
        elif spec.normalization == 'imagenet':
            x = (x / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        # 'none': raw 0..255 float
        if self.layout == 'NCHW':
            x = np.transpose(x, (2, 0, 1))
        return x[np.newaxis, ...]

    def postprocess(self, raw) -> float:
        """Raw output -> manipulation score in [0, 1], following spec.output_mode exactly."""
        out = np.asarray(raw, dtype=np.float64).ravel()
        mode, idx = self.spec.output_mode, self.spec.fake_index
        if mode in ('sigmoid', 'logit'):
            if out.size != 1:
                raise ModelConfigError(f'output_mode={mode} expects 1 output, model returns {out.size}.')
            if mode == 'sigmoid' and not (-1e-6 <= out[0] <= 1 + 1e-6):
                raise ModelConfigError('output_mode=sigmoid but model output is not a probability.')
            score = out[0] if mode == 'sigmoid' else _sigmoid(out[0])
        else:
            if out.size < 2 or idx is None or idx >= out.size:
                raise ModelConfigError(f'output_mode={mode} needs >=2 outputs and a valid fake_index.')
            if mode == 'softmax':
                if abs(out.sum() - 1.0) > 1e-3 or out.min() < -1e-6:
                    raise ModelConfigError('output_mode=softmax but outputs are not probabilities.')
                probs = out
            else:
                e = np.exp(out - out.max())
                probs = e / e.sum()
            score = probs[idx]
        return float(np.clip(score, 0.0, 1.0))

    def predict(self, face_rgb: np.ndarray) -> ModelOutput:
        spec = self.spec
        base = dict(key=spec.key, name=spec.display_name, version=spec.version,
                    weights_sha256=spec.sha256, preprocessing_version=spec.preprocessing_version,
                    threshold=spec.threshold, score_type=self.score_type, scope=spec.scope)
        if self.status != AVAILABLE:
            return ModelOutput(status=self.status, message=self.message, **base)
        t0 = time.perf_counter()
        try:
            score = self.postprocess(self._forward(self.preprocess(face_rgb)))
        except ModelConfigError as exc:
            return ModelOutput(status=FAILED, message=str(exc), **base)
        except Exception:
            logger.exception('inference failed for %s', spec.key)
            return ModelOutput(status=FAILED, message='Inference failed (see server log).', **base)
        label = LABEL_MANIPULATED if score >= spec.threshold else LABEL_AUTHENTIC
        return ModelOutput(status=AVAILABLE, score=round(score, 6), label=label,
                           inference_ms=round((time.perf_counter() - t0) * 1000, 2), **base)

    def explain(self, face_rgb: np.ndarray, save_dir: str) -> dict:
        return {'available': False, 'method': None,
                'reason': 'No explanation method is implemented for this model type.'}

    def health_check(self) -> dict:
        s = self.spec
        return {'key': s.key, 'name': s.display_name, 'status': self.status,
                'message': self.message, 'version': s.version, 'framework': s.framework,
                'device': self.device if self.status == AVAILABLE else None,
                'input_size': s.input_size}


# ------------------------------------------------------------------- ONNX
class OnnxDetector(BaseDetector):
    def _load_backend(self, path):
        import onnxruntime as ort
        providers = ['CPUExecutionProvider']
        if self.device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
        self._model = ort.InferenceSession(str(path), providers=providers)
        inp = self._model.get_inputs()[0]
        self._input_name, shape = inp.name, inp.shape
        self._fp16 = 'float16' in inp.type
        if len(shape) != 4:
            raise ModelConfigError('ONNX model input must be a 4-D image tensor.')
        if shape[1] == 3:
            self.layout, hw = 'NCHW', shape[2:]
        elif shape[3] == 3:
            self.layout, hw = 'NHWC', shape[1:3]
        else:
            raise ModelConfigError('Cannot determine ONNX channel layout (no dimension equal to 3).')
        for dim in hw:
            if isinstance(dim, int) and dim != self.spec.input_size:
                raise ModelConfigError(f'input_size {self.spec.input_size} does not match the '
                                       f'model input size {dim}.')

    def _forward(self, batch):
        batch = batch.astype(np.float16 if self._fp16 else np.float32)
        return self._model.run(None, {self._input_name: batch})[0]

    def explain(self, face_rgb, save_dir):
        from ai_models.explainers import explain_for
        return explain_for(self, face_rgb, save_dir)


# ------------------------------------------------------------------ PyTorch
class TorchDetector(BaseDetector):
    """TorchScript (.pt) or a plain state_dict for timm/torchvision architectures.
    Full-model pickles are NOT accepted (arbitrary code execution on load).
    NOTE: implemented to spec but not executed in the development sandbox (no torch)."""
    layout = 'NCHW'

    _TIMM_NAMES = {'xception': 'legacy_xception', 'efficientnet_b4': 'efficientnet_b4',
                   'vit_b16': 'vit_base_patch16_224'}

    def _load_backend(self, path):
        import torch
        self._torch = torch
        dev = torch.device(self.device)
        try:
            model = torch.jit.load(str(path), map_location=dev)
        except Exception:
            state = torch.load(str(path), map_location='cpu', weights_only=True)
            state = state.get('state_dict', state) if isinstance(state, dict) else state
            state = {k.replace('module.', '', 1): v for k, v in state.items()}
            name = self._TIMM_NAMES.get(self.spec.key)
            if not name:
                raise ModelConfigError('State-dict loading is not available for this architecture; '
                                       'export TorchScript or ONNX instead.')
            import timm
            model = timm.create_model(name, pretrained=False, num_classes=self.spec.num_outputs,
                                      img_size=self.spec.input_size) if self.spec.key == 'vit_b16' \
                else timm.create_model(name, pretrained=False, num_classes=self.spec.num_outputs)
            model.load_state_dict(state, strict=True)
        model.to(dev).eval()
        if self.spec.precision == 'fp16' and self.device == 'cuda':
            model.half()
        self._model = model

    def _forward(self, batch):
        torch = self._torch
        x = torch.from_numpy(batch).to(self.device)
        if self.spec.precision == 'fp16' and self.device == 'cuda':
            x = x.half()
        with torch.inference_mode():
            return self._model(x).float().cpu().numpy()

    def explain(self, face_rgb, save_dir):
        from ai_models.explainers import explain_for
        return explain_for(self, face_rgb, save_dir)


# -------------------------------------------------------------------- Keras
class KerasDetector(BaseDetector):
    """Keras .keras/.h5 models. NOTE: implemented to spec but not executed in the
    development sandbox (no tensorflow). .h5 files can embed Lambda layers that run
    code on load, so provide a sha256 and only load files you trained yourself."""
    layout = 'NHWC'

    def _load_backend(self, path):
        import tensorflow as tf
        self._tf = tf
        self._model = tf.keras.models.load_model(str(path), compile=False)
        shape = self._model.input_shape
        if len(shape) != 4 or shape[-1] != 3:
            raise ModelConfigError('Keras model input must be (H, W, 3).')
        for dim in shape[1:3]:
            if dim is not None and dim != self.spec.input_size:
                raise ModelConfigError(f'input_size {self.spec.input_size} does not match the '
                                       f'model input size {dim}.')

    def _forward(self, batch):
        return np.asarray(self._model(batch, training=False))

    def explain(self, face_rgb, save_dir):
        from ai_models.explain import keras_gradcam
        return keras_gradcam(self, face_rgb, save_dir)


# --------------------------------------------------------- FFT/DCT + SVM
class FrequencySVMDetector(BaseDetector):
    """preprocessing -> transform -> features -> (scaler -> SVM) -> score.
    The joblib file must be a scikit-learn Pipeline/estimator with exactly 2 classes.
    Pickle can execute code on load, so a SHA-256 is mandatory."""
    score_type = 'SVM score (uncalibrated unless Platt-scaled)'

    def _load_backend(self, path):
        import joblib
        model = joblib.load(path)
        classes = list(getattr(model, 'classes_', []))
        if len(classes) != 2:
            raise ModelConfigError('SVM must be a binary classifier with classes_.')
        if self.spec.fake_label not in classes:
            raise ModelConfigError('fake_label is not one of the SVM classes.')
        self._model, self._classes = model, classes
        self.score_type = ('SVM Platt-scaled probability' if hasattr(model, 'predict_proba')
                           else 'sigmoid(SVM margin), uncalibrated')

    def preprocess(self, face_rgb):
        from ai_models.frequency_features import extract
        return extract(face_rgb, self.spec.feature_method, self.spec.input_size)

    def _forward(self, feats):
        n_expected = getattr(self._model, 'n_features_in_', feats.shape[1])
        if feats.shape[1] != n_expected:
            raise ModelConfigError(f'Feature length {feats.shape[1]} != SVM expectation {n_expected}; '
                                   'feature_method does not match the trained model.')
        fake_idx = self._classes.index(self.spec.fake_label)
        if hasattr(self._model, 'predict_proba'):
            p_fake = float(self._model.predict_proba(feats)[0][fake_idx])
        else:
            p_pos = float(_sigmoid(self._model.decision_function(feats)[0]))
            p_fake = p_pos if fake_idx == 1 else 1.0 - p_pos
        return np.array([1.0 - p_fake, p_fake])

    def postprocess(self, raw):
        return float(np.clip(np.asarray(raw).ravel()[1], 0.0, 1.0))

    def explain(self, face_rgb, save_dir):
        from ai_models.explainers import explain_for
        return explain_for(self, face_rgb, save_dir)

    def _self_test(self):
        size = self.spec.input_size
        rng = np.random.default_rng(0)
        self.postprocess(self._forward(self.preprocess(
            rng.integers(0, 255, (size, size, 3), dtype=np.uint8))))


_BACKENDS = {'onnx': OnnxDetector, 'torch': TorchDetector, 'keras': KerasDetector,
             'sklearn': FrequencySVMDetector}


def create_detector(spec: ModelSpec) -> BaseDetector:
    cls = _BACKENDS.get(spec.framework, BaseDetector)
    return cls(spec)
