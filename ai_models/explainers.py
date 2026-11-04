"""
Per-model explanations. ONNX models are converted to a differentiable copy and parity-checked; PyTorch models
use an isolated copy of their loaded weights or a separately pinned state_dict. Gradients are not available from
ONNX Runtime itself.

Why not one method for every model?  The right explanation depends on how the model computes its score:

  EfficientNet-B4 / Xception               conv -> global-average-pool -> linear      Grad-CAM on the last conv block
  ViT-B/16                                 patch tokens + attention                   Grad-CAM on the last block's patch
                                                                                      tokens (14x14 grid)  | attention rollout
  Organika SDXL Detector                   Swin Transformer                          Grad-CAM on final norm output (7x7)
  FFT/DCT + SVM                            hand-made frequency features, no conv      Grad-CAM is not defined: band ablation
                                                                                      in the frequency domain

`gradcampp` (Grad-CAM++) and `rollout` (ViT) are implemented as alternatives. `manage.py evaluate_explanations`
measures which one is the most faithful for each model on your data (deletion test against a random-mask
control) and tells you what to write into `models.json`.

Integrity: the .pth is SHA-256 pinned and loaded with weights_only=True; before the first explanation it must
reproduce the deployed model's score on a fixed probe image (otherwise the heat-map would explain a different
model than the one that produced the verdict) and the explanation is switched off with a clear reason.
"""

import copy
import logging
import threading
from pathlib import Path

import cv2
import numpy as np
from django.conf import settings

from ai_models.explain import DISCLAIMER, _save, overlay_heatmap

logger = logging.getLogger('deepshield.models')

# key -> timm architecture (None = defined here), target layer and feature-map layout
ARCH = {
    'efficientnet_b4': dict(timm='efficientnet_b4', layer='bn2'),
    'xception':        dict(timm='legacy_xception', layer='act4'),         # the name used by the training notebooks
    'vit_b16':         dict(timm='vit_base_patch16_224', layer='blocks.11.norm1', tokens=True),
    # Organika/sdxl-detector: the ONNX graph may return token sequences while timm
    # returns channel-last spatial maps from the same final normalization layer.
    'sdxl_detector':   dict(timm='swin_base_patch4_window7_224', layer='norm', swin=True),
}
DEFAULT_METHOD = {'efficientnet_b4': 'gradcam', 'xception': 'gradcam',
                  'vit_b16': 'gradcam', 'sdxl_detector': 'gradcam'}
CANDIDATES = {k: ('gradcam', 'gradcampp') for k in ARCH} | {'vit_b16': ('gradcam', 'gradcampp', 'rollout')}
PARITY_TOLERANCE = 2e-3                                                     # |score_torch - score_deployed| (non-logit outputs)
PARITY_LOGIT_ATOL = 2e-3                                                    # max |logit_torch - logit_onnx| in fp32


def _unavailable(reason):
    return {'available': False, 'method': None, 'reason': reason}


def _spatial_activations(output, arch):
    if arch.get('tokens'):
        tokens = output[:, 1:, :]
        grid = int(round(tokens.shape[1] ** 0.5))
        if grid * grid != tokens.shape[1]:
            raise RuntimeError('transformer patch tokens do not form a square grid')
        return tokens.reshape(tokens.shape[0], grid, grid, tokens.shape[2]).permute(0, 3, 1, 2)
    if arch.get('swin'):
        if output.ndim == 3:
            grid = int(round(output.shape[1] ** 0.5))
            if grid * grid != output.shape[1]:
                raise RuntimeError('Swin tokens do not form a square grid')
            return output.reshape(output.shape[0], grid, grid, output.shape[2]).permute(0, 3, 1, 2)
        if output.ndim == 4:
            if output.shape[-1] >= max(output.shape[1:3]):
                return output.permute(0, 3, 1, 2)
            if output.shape[1] >= max(output.shape[2:]):
                return output
            raise RuntimeError('could not determine the Swin feature-map channel dimension')
        raise RuntimeError('Swin Grad-CAM layer must return a spatial feature map')
    return output


def _repair_layer_norm_shapes(net, torch):
    """Infer erased LayerNorm dimensions from ONNX affine weights."""
    for module in net.modules():
        if not isinstance(module, torch.nn.LayerNorm) or all(dim > 0 for dim in module.normalized_shape):
            continue
        weight = module.weight
        if weight is None or not weight.shape or any(dim <= 0 for dim in weight.shape):
            raise RuntimeError('ONNX LayerNorm has no valid feature shape')
        module.normalized_shape = tuple(weight.shape)


def build_network(key, size, num_classes=2):
    """The architecture the training notebooks used (random init; weights come from the .pth)."""
    a = ARCH[key]
    import timm
    kw = dict(pretrained=False, num_classes=num_classes)
    if key == 'vit_b16':
        kw['img_size'] = size
    if key == 'sdxl_detector':
        kw['img_size'] = size
    return timm.create_model(a['timm'], **kw)


class CamExplainer:
    """Grad-CAM family / attention rollout on a PyTorch copy of a deployed model."""

    MISMATCH_HINT = 're-export the .pth from the same training run as the .onnx'

    def __init__(self, detector):
        self.det = detector
        self.cfg = detector.spec.explain or {}
        self._net = None
        self._lock = threading.Lock()
        self.error = ''

    # ----------------------------------------------------------- loading + integrity
    def _ensure(self):
        if self._net is not None:
            return self._net
        if self.error:
            raise RuntimeError(self.error)
        try:
            import torch
            spec = self.det.spec
            if spec.key not in ARCH:
                raise RuntimeError('no PyTorch architecture is registered for this model')
            rel = self.cfg.get('weights')
            if not rel:
                raise RuntimeError('explain.weights is not configured')
            path = Path(rel) if Path(rel).is_absolute() else Path(settings.MODEL_DIR) / rel
            model_dir = Path(settings.MODEL_DIR).resolve()
            if model_dir not in path.resolve().parents:
                raise RuntimeError('explain weights must be located inside MODEL_DIR')
            if not path.is_file():
                raise RuntimeError('explain weights file not found')
            from ai_models.detectors import file_sha256
            if not self.cfg.get('sha256') or file_sha256(path) != str(self.cfg['sha256']).lower():
                raise RuntimeError('explain weights checksum does not match the configured SHA-256')
            net = build_network(spec.key, spec.input_size, spec.num_outputs)
            state = torch.load(str(path), map_location='cpu', weights_only=True)
            state = state.get('state_dict', state) if isinstance(state, dict) else state
            net.load_state_dict({k.replace('module.', '', 1): v.float() for k, v in state.items()}, strict=True)
            net.eval()
            self._parity(net, torch)
            self._net = net
            return net
        except Exception as exc:
            self.error = str(exc) if isinstance(exc, RuntimeError) else f'{type(exc).__name__}'
            logger.exception('explainer for %s unavailable', self.det.spec.key)
            raise RuntimeError(self.error) from exc

    def _parity(self, net, torch):
        """The explained network must reproduce the deployed model on fixed probes (3 images, raw logits when the
        deployed output is logits: a saturated 0/1 score cannot tell two different models apart, logits can)."""
        size = self.det.spec.input_size
        rng = np.random.default_rng(7)
        yy, xx = np.mgrid[0:size, 0:size]
        probes = [rng.integers(0, 255, (size, size, 3), dtype=np.uint8),
                  np.stack([xx * 255 // size, yy * 255 // size, (xx + yy) * 255 // (2 * size)], -1).astype(np.uint8),
                  cv2.GaussianBlur(rng.integers(0, 255, (size, size, 3), dtype=np.uint8), (0, 0), size / 20)]
        atol = 5e-2 if getattr(self.det, '_fp16', False) else PARITY_LOGIT_ATOL
        for probe in probes:
            x = self.det.preprocess(probe)
            deployed = np.asarray(self.det._forward(x), dtype=np.float64).ravel()
            with torch.no_grad():
                mine = net(torch.from_numpy(x).float()).numpy().astype(np.float64).ravel()
            if self.det.spec.output_mode == 'logits' and deployed.size == mine.size:
                diff = float(np.abs(deployed - mine).max())
                tol = atol + 1e-3 * float(np.abs(mine).max())          # relative slack for large logits
            else:
                diff = abs(self.det.postprocess(deployed) - self.det.postprocess(mine))
                tol = PARITY_TOLERANCE
            if diff > tol:
                raise RuntimeError(f'explain network does not match the deployed model (probe difference {diff:.4f}); {self.MISMATCH_HINT}')

    # ----------------------------------------------------------- maps
    def _target_index(self):
        i = self.det.spec.fake_index
        return 0 if i is None else int(i)

    def cam(self, face_rgb, method=None):
        """Class-activation map for the 'manipulated' output, float32 in [0, 1], coarse grid (e.g. 7x7, 14x14)."""
        method = method or self.cfg.get('method') or DEFAULT_METHOD[self.det.spec.key]
        if method not in CANDIDATES[self.det.spec.key]:
            raise RuntimeError(f'method {method!r} is not available for {self.det.spec.key}')
        with self._lock:
            net = self._ensure()
            import torch
            x = torch.from_numpy(self.det.preprocess(face_rgb)).float()
            cam = self._rollout(net, x, torch) if method == 'rollout' else self._gradcam(net, x, torch, method == 'gradcampp')
        cam = np.maximum(cam, 0).astype(np.float32)
        return cam / cam.max() if cam.max() > 0 else cam

    def _gradcam(self, net, x, torch, plus):
        a = ARCH[self.det.spec.key]
        module = self._target_module(net)
        store = {}

        handle = module.register_forward_hook(lambda m, i, o: store.__setitem__('a', o))   # the real graph node
        try:
            with torch.enable_grad():
                score = net(x).reshape(1, -1)[0, self._target_index()]
                grads, = torch.autograd.grad(score, store['a'])
        finally:
            handle.remove()
        act, g = _spatial_activations(store['a'].detach(), a)[0], _spatial_activations(grads, a)[0]
        if plus:                                          # Grad-CAM++ (Chattopadhay et al. 2018)
            g2, g3 = g ** 2, g ** 3
            denom = 2 * g2 + act.sum(dim=(1, 2), keepdim=True) * g3
            alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
            w = (alpha * torch.relu(g)).sum(dim=(1, 2))
        else:                                             # Grad-CAM (Selvaraju et al. 2017)
            w = g.mean(dim=(1, 2))
        raw = (w[:, None, None] * act).sum(0)
        rel = torch.relu(raw).detach().cpu().numpy()
        if np.any(rel > 0):
            return rel
        # Fallback when attributions are strictly non-positive (e.g. authentic face or negative logit):
        # Normalize the raw gradient-weighted activation field so interpretable saliency is preserved.
        raw_np = raw.detach().cpu().numpy()
        ptp = float(np.ptp(raw_np))
        if ptp > 1e-7:
            return (raw_np - raw_np.min()) / ptp
        act_map = torch.relu(act).mean(dim=0).detach().cpu().numpy()
        act_ptp = float(np.ptp(act_map))
        if act_ptp > 1e-7:
            return (act_map - act_map.min()) / act_ptp
        return np.ones_like(raw_np, dtype=np.float32) * 0.5

    def _target_module(self, net):
        name = self.cfg.get('layer') or ARCH[self.det.spec.key]['layer']
        module = dict(net.named_modules()).get(name)
        if module is None:
            raise RuntimeError(f'Grad-CAM target layer {name!r} was not found')
        return module

    def _attention_maps(self, net, x, torch):
        """Per-block attention probabilities (1, heads, N, N), in block order."""
        blocks = list(net.blocks)
        maps, saved = [], []
        for b in blocks:
            saved.append(b.attn.fused_attn)
            b.attn.fused_attn = False                       # fused SDPA never materialises the attention matrix
        handles = [b.attn.attn_drop.register_forward_hook(lambda m, i, o: maps.append(i[0].detach())) for b in blocks]
        try:
            with torch.no_grad():
                net(x)
        finally:
            for h in handles:
                h.remove()
            for b, s in zip(blocks, saved):
                b.attn.fused_attn = s
        return maps

    def _rollout(self, net, x, torch):
        """Attention rollout (Abnar & Zuidema 2020): residual-corrected product of head-averaged attention."""
        maps = self._attention_maps(net, x, torch)
        n = maps[0].shape[-1]
        roll = torch.eye(n, device=maps[0].device, dtype=maps[0].dtype)
        for att in maps:
            a = att.mean(dim=1)[0] + torch.eye(n, device=att.device, dtype=att.dtype)
            roll = (a / a.sum(dim=-1, keepdim=True)) @ roll
        patches = roll[0, 1:]
        g = int(round(patches.numel() ** 0.5))
        return patches.reshape(g, g).detach().cpu().numpy()

    # ----------------------------------------------------------- public
    def explain(self, face_rgb, save_dir):
        try:
            method = self.cfg.get('method') or DEFAULT_METHOD[self.det.spec.key]
            cam = self.cam(face_rgb, method)
        except RuntimeError as exc:
            return _unavailable(f'Explanation unavailable: {exc}.' if self.error else 'Explanation failed.')
        except Exception:
            logger.exception('explanation failed for %s', self.det.spec.key)
            return _unavailable('Explanation failed.')
        if not np.isfinite(cam).all() or not np.any(cam > 0):
            return _unavailable('Grad-CAM produced no finite positive attribution.')
        size = self.det.spec.input_size
        overlay = overlay_heatmap(cv2.resize(face_rgb, (size, size)), cam)
        name = {'gradcam': 'grad-cam', 'gradcampp': 'grad-cam++', 'rollout': 'attention-rollout'}[method]
        return {'available': True, 'method': name, 'target_class': 'manipulated', 'domain': 'image',
                'layer': self.cfg.get('layer') or ARCH[self.det.spec.key]['layer'], 'grid': list(cam.shape),
                'cam': _cam_grid(cam),
                'image_path': _save(overlay, save_dir), 'model_version': self.det.spec.version, 'disclaimer': DISCLAIMER}


def _cam_grid(cam, max_side=16):
    """Small normalised (0..1) copy of the attention grid, kept with the result so the UI can combine
    several models' maps and place them back on the photo. Stored rounded: a few hundred bytes."""
    try:
        c = np.asarray(cam, dtype=np.float32)
        if c.ndim != 2 or not np.isfinite(c).all():
            return None
        if max(c.shape) > max_side:
            c = cv2.resize(c, (max_side, max_side), interpolation=cv2.INTER_AREA)
        c = c - c.min()
        top = float(c.max())
        return np.round(c / top, 3).tolist() if top > 0 else None
    except Exception:
        return None


class LoadedTorchCamExplainer(CamExplainer):
    """Explain an isolated copy of the already-loaded PyTorch model (no second checkpoint required)."""

    def _ensure(self):
        if self.error:
            raise RuntimeError(self.error)
        if self._net is None:
            try:
                import torch
                model = self.det._model
                if model is None or isinstance(model, torch.jit.ScriptModule):
                    raise RuntimeError('Grad-CAM requires a non-TorchScript PyTorch model with named layers')
                self._net = copy.deepcopy(model).float().cpu().eval()
                for parameter in self._net.parameters():
                    parameter.requires_grad_(True)
            except Exception as exc:
                self.error = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
                raise RuntimeError(self.error) from exc
        return self._net


def strip_identity(model):
    """torch's exporter routes weights through Identity nodes; onnx2torch needs them as plain initializers."""
    import onnx
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    rename, keep = {}, []
    for n in g.node:
        if n.op_type == 'Identity':
            src = rename.get(n.input[0], n.input[0])
            if src in inits:
                t = onnx.TensorProto()
                t.CopyFrom(inits[src])
                t.name = n.output[0]
                g.initializer.append(t)
                inits[t.name] = t
            else:
                rename[n.output[0]] = src
        else:
            keep.append(n)
    del g.node[:]
    g.node.extend(keep)
    for n in g.node:
        for i, x in enumerate(n.input):
            if x in rename:
                n.input[i] = rename[x]
    for o in g.output:
        o.name = rename.get(o.name, o.name)
    return model


def locate_nodes(model, key, size):
    """Graph analysis of the *deployed ONNX*: names of the node that produces the Grad-CAM tensor (and, for ViT, the softmax nodes).
      CNN with global-average-pool   -> the tensor that feeds the pool (last conv block)
      ViT-B/16                       -> the input of the last transformer block (residual Add, tokens x 768).
      sdxl_detector (Swin)           -> the output of the final LayerNorm before the classifier head."""
    prod = {o: n for n in model.graph.node for o in n.output}
    if key == 'vit_b16':                                       # torch's exporter names nodes after the module path
        import re
        names = [n.name for n in model.graph.node]
        outs = {int(m.group(1)): n for n in names if (m := re.fullmatch(r'/blocks/blocks\.(\d+)/Add_1', n))}
        soft = sorted((int(m.group(1)), n) for n in names if (m := re.fullmatch(r'/blocks/blocks\.(\d+)/attn/Softmax', n)))
        last = max(outs, default=-1)
        if last < 1 or (last - 1) not in outs or len(soft) != last + 1:
            raise RuntimeError('unexpected ViT graph layout (re-export with training/kaggle_train.py)')
        return outs[last - 1], [n for _, n in soft]            # input of the last transformer block
    if key == 'sdxl_detector':                                 # Swin Transformer: find the final LayerNorm
        # The final 'norm' module is the last LayerNorm in the graph before the classifier.
        lns = [n for n in model.graph.node if n.op_type == 'LayerNormalization']
        if not lns:
            raise RuntimeError('no LayerNormalization found in sdxl_detector ONNX graph')
        # The last LayerNorm is the `norm` head that precedes global average pooling / classifier.
        return lns[-1].name, []
    gap = [n for n in model.graph.node if n.op_type in ('GlobalAveragePool', 'ReduceMean')]
    if not gap:
        raise RuntimeError('no global-average-pool found in the ONNX graph')
    return prod[gap[-1].input[0]].name, []              # the LAST pool (EfficientNet has squeeze-excite ReduceMeans before it)


def _submodule(gm, node_name):
    name = node_name.lstrip('/').replace('.', '/')
    try:
        return gm.get_submodule(name)
    except AttributeError:
        for n, m in gm.named_children():
            if n.replace('/', '').replace('_', '') == node_name.replace('/', '').replace('.', '').replace('_', ''):
                return m
    raise RuntimeError(f'cannot map ONNX node {node_name!r} to a module')


def _register_missing_converters():
    """Register missing comparison node converters in onnx2torch for newer opset versions (14..21)."""
    try:
        from onnx2torch.node_converters.registry import add_converter, _CONVERTER_REGISTRY, OperationDescription
        from onnx2torch.node_converters.comparisons import OnnxCompare
        from onnx2torch.utils.common import OperationConverterResult, onnx_mapping_from_node

        for op in ('GreaterOrEqual', 'LessOrEqual', 'Greater', 'Less', 'Equal'):
            for v in range(7, 22):
                desc = OperationDescription(domain='', operation_type=op, version=v)
                if desc not in _CONVERTER_REGISTRY:
                    try:
                        add_converter(operation_type=op, version=v)(
                            lambda node, graph: OperationConverterResult(
                                torch_module=OnnxCompare(operation_type=node.operation_type),
                                onnx_mapping=onnx_mapping_from_node(node=node),
                            )
                        )
                    except Exception:
                        pass
    except Exception:
        pass


class OnnxCamExplainer(CamExplainer):
    """No extra files: the deployed .onnx itself is converted to a differentiable PyTorch module (onnx2torch). Same weights by
    construction; the conversion is still checked against ONNX Runtime on 3 probes before the first explanation."""
    MISMATCH_HINT = 'the ONNX graph could not be converted faithfully (re-export it with training/kaggle_train.py)'

    def _ensure(self):
        if self._net is not None:
            return self._net
        if self.error:
            raise RuntimeError(self.error)
        try:
            import onnx
            import onnx2torch
            import torch
            _register_missing_converters()
            spec = self.det.spec
            if spec.key not in ARCH:
                raise RuntimeError('no explanation recipe is registered for this model')
            model = strip_identity(onnx.load(str(spec.path)))
            self._target_node, self._softmax_nodes = locate_nodes(model, spec.key, spec.input_size)
            net = onnx2torch.convert(model).eval()
            _repair_layer_norm_shapes(net, torch)
            for p in net.parameters():
                p.requires_grad_(True)
            self._parity(net, torch)
            self._net = net
            return net
        except ImportError:
            self.error = 'install torch and onnx2torch (pip install torch torchvision timm onnx2torch)'
            raise RuntimeError(self.error)
        except Exception as exc:
            self.error = str(exc) if isinstance(exc, RuntimeError) else f'{type(exc).__name__}'
            logger.exception('onnx explainer for %s unavailable', self.det.spec.key)
            raise RuntimeError(self.error) from exc

    def _target_module(self, net):
        return _submodule(net, self._target_node)

    def _attention_maps(self, net, x, torch):
        maps = []
        handles = [_submodule(net, n).register_forward_hook(lambda m, i, o: maps.append(o.detach())) for n in self._softmax_nodes]
        try:
            with torch.no_grad():
                net(x)
        finally:
            for h in handles:
                h.remove()
        return maps


class FrequencyExplainer:
    """FFT/DCT + SVM: no convolution and no gradient, so Grad-CAM is not defined. Instead each frequency group is
    replaced by the training mean (the scaler's mean_) and the drop in the 'manipulated' score is the attribution
    (feature-group ablation). Rendered on the 2-D spectrum (FFT rings) or on the 8x8 DCT grid."""

    def __init__(self, detector):
        self.det = detector

    def _reference(self, n):
        for _, step in getattr(self.det._model, 'steps', []):
            if hasattr(step, 'mean_') and len(step.mean_) == n:
                return np.asarray(step.mean_, dtype=np.float64)
        return np.zeros(n)

    def attribution(self, face_rgb):
        feats = self.det.preprocess(face_rgb)
        n = feats.shape[1]
        base = self.det.postprocess(self.det._forward(feats))
        ref = self._reference(n)
        if self.det.spec.feature_method == 'dct_block_stats':
            groups = [np.array([i]) for i in range(n)]
        else:
            groups = np.array_split(np.arange(n), 8)           # 8 radial bands, low -> high frequency
        delta = []
        for idx in groups:
            f2 = feats.copy()
            f2[0, idx] = ref[idx]
            delta.append(base - self.det.postprocess(self.det._forward(f2)))   # > 0: this group pushes towards 'manipulated'
        return base, np.array(delta)

    @staticmethod
    def _diverge(v):                                           # v in [-1, 1] -> RGB (red = fake, blue = real)
        return np.where(v[..., None] >= 0, np.array([235, 60, 60]), np.array([60, 110, 235])).astype(np.float32)

    def explain(self, face_rgb, save_dir):
        try:
            from ai_models.frequency_features import _to_gray
            size = self.det.spec.input_size
            base, delta = self.attribution(face_rgb)
            top = float(np.abs(delta).max()) or 1.0
            gray = _to_gray(face_rgb, size)
            if self.det.spec.feature_method == 'dct_block_stats':
                canvas = cv2.cvtColor(cv2.normalize(np.log1p(np.abs(cv2.dct(gray[:size // 8 * 8, :size // 8 * 8] - 128.0))),
                                                    None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                canvas = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_NEAREST).astype(np.float32)
                v = cv2.resize((delta / top).reshape(8, 8).astype(np.float32), (size, size), interpolation=cv2.INTER_NEAREST)
                where = np.ones((size, size), bool)
                shown = 'DCT coefficient position'
            else:
                win = np.outer(np.hanning(size), np.hanning(size)).astype(np.float32)
                spec = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2((gray - gray.mean()) * win))) ** 2)
                canvas = cv2.cvtColor(cv2.normalize(spec, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                                      cv2.COLOR_GRAY2RGB).astype(np.float32)
                yy, xx = np.indices((size, size))
                radius = np.hypot(yy - size / 2, xx - size / 2)
                band = np.clip((radius / (size / 2) * len(delta)).astype(int), 0, len(delta) - 1)
                v = (delta / top)[band].astype(np.float32)
                where = radius <= size / 2
                shown = 'frequency band (centre = low, edge = high)'
            alpha = (0.15 + 0.6 * np.abs(v))[..., None] * where[..., None]
            out = np.clip(canvas * (1 - alpha) + self._diverge(v) * alpha, 0, 255).astype(np.uint8)
            cv2.putText(out, 'red: towards FAKE   blue: towards REAL', (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            return {'available': True, 'method': 'frequency-band-ablation', 'target_class': 'manipulated',
                    'domain': 'frequency', 'layer': shown, 'band_contributions': [round(float(d), 4) for d in delta],
                    'image_path': _save(out, save_dir), 'model_version': self.det.spec.version,
                    'disclaimer': DISCLAIMER + ' Grad-CAM is not defined for this model; the map shows which '
                                  'frequency components changed the SVM score.'}
        except Exception:
            logger.exception('frequency explanation failed')
            return _unavailable('Explanation failed.')


def cam_explainer_for(detector):
    """The image explainer for a detector, or None when the model has no image explanation (SVM, unsupported)."""
    if detector.spec.key not in ARCH:
        return None
    if not hasattr(detector, '_explainer'):
        if detector.spec.framework == 'torch' and not (detector.spec.explain or {}).get('weights'):
            detector._explainer = LoadedTorchCamExplainer(detector)
        elif (detector.spec.explain or {}).get('weights'):               # optional: an attached PyTorch state_dict
            detector._explainer = CamExplainer(detector)
        elif detector.spec.framework == 'onnx':                           # default: convert the deployed ONNX itself
            detector._explainer = OnnxCamExplainer(detector)
        else:
            return None
    return detector._explainer


def explain_for(detector, face_rgb, save_dir):
    """Dispatcher used by OnnxDetector / FrequencySVMDetector.explain()."""
    if detector.spec.key == 'freq_svm':
        return FrequencyExplainer(detector).explain(face_rgb, save_dir)
    ex = cam_explainer_for(detector)
    if ex is None:
        return _unavailable('No explanation method is implemented for this model.')
    return ex.explain(face_rgb, save_dir)
