"""
Grad-CAM explanations (CNNs only) + overlay helper.

A heat-map shows image regions that influenced the model's output. It is NOT proof
of manipulation and NOT a ground-truth localisation. Transformers and ONNX/SVM
models report explanations as unavailable instead of faking them.

The Keras and PyTorch implementations are backend-specific; ONNX models use the
parity-checked differentiable converter in `ai_models.explainers`.
"""

import logging
import os
import uuid

import cv2
import numpy as np

logger = logging.getLogger('deepshield.models')

DISCLAIMER = ('Visualization indicates regions influencing the model prediction; '
              'it is not proof of manipulation.')


def overlay_heatmap(face_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a [0,1] class-activation map over the RGB face; returns RGB uint8."""
    h, w = face_rgb.shape[:2]
    cam = np.clip(cam.astype(np.float32), 0, None)
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    colour = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(face_rgb, 1 - alpha, colour, alpha, 0)


def _save(overlay_rgb, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'gradcam_{uuid.uuid4().hex[:12]}.png')
    cv2.imwrite(path, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
    return path


def _result(detector, layer, path):
    return {'available': True, 'method': 'grad-cam', 'target_class': 'manipulated', 'layer': layer,
            'image_path': path, 'model_version': detector.spec.version, 'disclaimer': DISCLAIMER}


def keras_gradcam(detector, face_rgb, save_dir):
    try:
        tf = detector._tf
        model = detector._model
        layer_name = detector.spec.gradcam_layer
        if not layer_name:
            convs = [l.name for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
            if not convs:
                return {'available': False, 'method': None,
                        'reason': 'No top-level Conv2D layer found; set gradcam_layer.'}
            layer_name = convs[-1]
        grad_model = tf.keras.models.Model(model.inputs,
                                           [model.get_layer(layer_name).output, model.output])
        x = tf.convert_to_tensor(detector.preprocess(face_rgb))
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(x, training=False)
            idx = detector.spec.fake_index if detector.spec.output_mode in ('softmax', 'logits') else 0
            target = preds[:, idx or 0]
        grads = tape.gradient(target, conv_out)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.nn.relu(tf.reduce_sum(conv_out[0] * weights, axis=-1)).numpy()
        size = detector.spec.input_size
        base = cv2.resize(face_rgb, (size, size))
        return _result(detector, layer_name, _save(overlay_heatmap(base, cam), save_dir))
    except Exception:
        logger.exception('keras grad-cam failed')
        return {'available': False, 'method': None, 'reason': 'Grad-CAM generation failed.'}


def torch_gradcam(detector, face_rgb, save_dir):
    try:
        torch = detector._torch
        model = detector._model
        if not hasattr(model, 'named_modules') or isinstance(model, torch.jit.ScriptModule):
            return {'available': False, 'method': None,
                    'reason': 'Grad-CAM needs a state_dict-loaded model; TorchScript hooks unsupported.'}
        name = detector.spec.gradcam_layer
        modules = dict(model.named_modules())
        if not name:
            convs = [n for n, m in modules.items() if isinstance(m, torch.nn.Conv2d)]
            if not convs:
                return {'available': False, 'method': None, 'reason': 'No Conv2d layer found.'}
            name = convs[-1]
        store = {}
        h1 = modules[name].register_forward_hook(lambda m, i, o: store.__setitem__('act', o))
        h2 = modules[name].register_full_backward_hook(lambda m, gi, go: store.__setitem__('grad', go[0]))
        try:
            x = torch.from_numpy(detector.preprocess(face_rgb)).to(detector.device).float()
            model.float()
            out = model(x).float()
            idx = detector.spec.fake_index if detector.spec.output_mode in ('softmax', 'logits') else 0
            model.zero_grad()
            out.ravel()[idx or 0].backward()
        finally:
            h1.remove(); h2.remove()
        w = store['grad'].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * store['act']).sum(dim=1))[0].detach().cpu().numpy()
        size = detector.spec.input_size
        base = cv2.resize(face_rgb, (size, size))
        return _result(detector, name, _save(overlay_heatmap(base, cam), save_dir))
    except Exception:
        logger.exception('torch grad-cam failed')
        return {'available': False, 'method': None, 'reason': 'Grad-CAM generation failed.'}
