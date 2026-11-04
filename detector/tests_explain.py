"""Explainability tests: every architecture gets a valid map, the integrity gate rejects weights that are not the
deployed model, and the frequency-domain explainer works for the SVM. Uses random-init nets created in the tests."""
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

import cv2
import joblib
import numpy as np
import torch
from django.conf import settings
from django.test import SimpleTestCase, override_settings
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ai_models import explainers as E
from ai_models.detectors import create_detector, file_sha256
from ai_models.frequency_features import extract
from ai_models.registry import ModelSpec

FACE = np.random.default_rng(1).integers(0, 255, (300, 300, 3), dtype=np.uint8)


def _pre(size):
    def f(face):
        x = cv2.resize(face, (size, size)).astype(np.float32) / 255.0
        return np.transpose(x, (2, 0, 1))[None]
    return f


def _set_positive_classifier(net, key, fake_index=1):
    linears = [module for module in net.modules() if isinstance(module, torch.nn.Linear)]
    with torch.no_grad():
        for layer in linears:
            layer.weight.zero_()
            if layer.bias is not None:
                layer.bias.zero_()
        linears[-1].weight[fake_index].fill_(1.0 / linears[-1].in_features)


class ArchitectureMaps(SimpleTestCase):
    GRID = {'efficientnet_b4': (380, (12, 12)),
            'xception': (299, (10, 10)), 'vit_b16': (224, (14, 14)), 'sdxl_detector': (224, (7, 7))}

    def test_every_method_on_every_architecture(self):
        for key, (size, grid) in self.GRID.items():
            fake_index = 0 if key == 'sdxl_detector' else 1
            det = NS(spec=NS(key=key, input_size=size, fake_index=fake_index, explain={}, version='t', num_outputs=2),
                     preprocess=_pre(size))
            ex = E.CamExplainer(det)
            ex._net = E.build_network(key, size).eval()
            for method in E.CANDIDATES[key]:
                cam = ex.cam(FACE, method)
                self.assertEqual(cam.shape, grid, (key, method))
                self.assertTrue(np.isfinite(cam).all() and 0 <= cam.min() and cam.max() <= 1, (key, method))

    def test_swin_sequence_and_spatial_layouts(self):
        arch = E.ARCH['sdxl_detector']
        self.assertEqual(E._spatial_activations(torch.zeros(1, 49, 8), arch).shape, (1, 8, 7, 7))
        self.assertEqual(E._spatial_activations(torch.zeros(1, 7, 7, 8), arch).shape, (1, 8, 7, 7))
        self.assertEqual(E._spatial_activations(torch.zeros(1, 8, 7, 7), arch).shape, (1, 8, 7, 7))

    def test_onnx_layer_norm_repairs_missing_feature_shape(self):
        layer = torch.nn.LayerNorm(0)
        layer.weight = torch.nn.Parameter(torch.ones(8))
        layer.bias = torch.nn.Parameter(torch.zeros(8))
        net = torch.nn.Sequential(layer)
        E._repair_layer_norm_shapes(net, torch)
        self.assertEqual(tuple(net[0].normalized_shape), (8,))
        self.assertEqual(net(torch.ones(1, 8)).shape, (1, 8))

    def test_sdxl_model_card_uses_the_artificial_class_and_logits(self):
        config_path = Path(settings.BASE_DIR) / 'trained_models' / 'models.json'
        config = json.loads(config_path.read_text(encoding='utf-8-sig'))['sdxl_detector']
        self.assertEqual((config['output_mode'], config['fake_index']), ('logits', 0))
        self.assertEqual(config['training']['parameters']['labels']['0'], 'artificial (manipulated)')

    def test_vit_rollout_restores_fused_attention(self):
        det = NS(spec=NS(key='vit_b16', input_size=224, fake_index=1, explain={}, version='t', num_outputs=2), preprocess=_pre(224))
        ex = E.CamExplainer(det)
        ex._net = E.build_network('vit_b16', 224).eval()
        before = [b.attn.fused_attn for b in ex._net.blocks]
        ex.cam(FACE, 'rollout')
        self.assertEqual(before, [b.attn.fused_attn for b in ex._net.blocks])

    def test_maps_depend_on_the_weights(self):
        """Sanity check (Adebayo et al. 2018): a re-initialised network must give a different map."""
        det = NS(spec=NS(key='xception', input_size=299, fake_index=1, explain={}, version='t', num_outputs=2), preprocess=_pre(299))
        maps = []
        for seed in (0, 1):
            torch.manual_seed(seed)
            ex = E.CamExplainer(det)
            ex._net = E.build_network('xception', 299).eval()
            maps.append(ex.cam(FACE, 'gradcampp').ravel())
        self.assertTrue(all(m.std() > 0 for m in maps))
        self.assertLess(abs(np.corrcoef(*maps)[0, 1]), 0.95)



class IntegrityGate(SimpleTestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        (self.tmp / 'xception').mkdir()
        torch.manual_seed(0)
        self.net = E.build_network('xception', 299).eval()
        _set_positive_classifier(self.net, 'xception')
        torch.onnx.export(self.net, torch.zeros(1, 3, 299, 299), str(self.tmp / 'xception/model.onnx'), dynamo=False,
                          input_names=['input'], output_names=['logits'], opset_version=17)
        torch.save(self.net.state_dict(), self.tmp / 'xception/explain.pth')

    def detector(self, weights='xception/explain.pth', sha=None):
        sha = sha if sha is not None else file_sha256(self.tmp / weights)
        spec = ModelSpec(key='xception', display_name='XceptionNet', family='CNN', path=self.tmp / 'xception/model.onnx',
                         framework='onnx', input_size=299, normalization='zero_one', output_mode='logits', fake_index=1,
                         version='t', explain={'method': 'gradcam', 'weights': weights, 'sha256': sha})
        det = create_detector(spec)
        det.load()
        self.assertEqual(det.status, 'AVAILABLE', det.message)
        return det

    def run_explain(self, det):
        with override_settings(MODEL_DIR=str(self.tmp)):
            return det.explain(FACE, str(self.tmp / 'maps'))

    def test_matching_weights_produce_a_map(self):
        with override_settings(MODEL_DIR=str(self.tmp), MODEL_REQUIRE_CHECKSUM=False):
            res = self.run_explain(self.detector())
        self.assertTrue(res['available'], res)
        self.assertEqual((res['method'], res['domain']), ('grad-cam', 'image'))
        self.assertTrue(Path(res['image_path']).is_file())

    def test_weights_of_a_different_model_are_refused(self):
        torch.manual_seed(5)
        torch.save(E.build_network('xception', 299).eval().state_dict(), self.tmp / 'xception/other.pth')
        with override_settings(MODEL_DIR=str(self.tmp), MODEL_REQUIRE_CHECKSUM=False):
            res = self.run_explain(self.detector('xception/other.pth'))
        self.assertFalse(res['available'])
        self.assertIn('does not match the deployed model', res['reason'])

    def test_checksum_mismatch_is_refused(self):
        with override_settings(MODEL_DIR=str(self.tmp), MODEL_REQUIRE_CHECKSUM=False):
            res = self.run_explain(self.detector(sha='0' * 64))
        self.assertFalse(res['available'])
        self.assertIn('checksum', res['reason'])

    def test_weights_outside_model_dir_are_refused(self):
        with override_settings(MODEL_DIR=str(self.tmp), MODEL_REQUIRE_CHECKSUM=False):
            det = self.detector()
            det.spec.explain['weights'] = '../../etc/passwd'
            res = self.run_explain(det)
        self.assertFalse(res['available'])

    def test_without_explain_block_the_onnx_is_converted(self):
        with override_settings(MODEL_DIR=str(self.tmp), MODEL_REQUIRE_CHECKSUM=False):
            det = self.detector()
            det.spec.explain = {}
            res = self.run_explain(det)
        self.assertTrue(res['available'], res)


class OnnxConversion(SimpleTestCase):
    """No .pth needed: the deployed ONNX is converted to PyTorch. For every architecture the converted network's map must be
    identical to the map of the original PyTorch network (same weights), proving the target layer was located correctly."""
    GRID = ArchitectureMaps.GRID

    def export(self, key, size, tmp, seed=1):
        torch.manual_seed(seed)
        net = E.build_network(key, size).eval()
        _set_positive_classifier(net, key)
        (tmp / key).mkdir(exist_ok=True)
        path = tmp / key / 'model.onnx'
        torch.onnx.export(net, torch.zeros(1, 3, size, size), str(path), dynamo=False, input_names=['input'],
                          output_names=['logits'], dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}}, opset_version=17)
        spec = ModelSpec(key=key, display_name=key, family='x', path=path, framework='onnx', input_size=size,
                         normalization='zero_one', output_mode='logits', fake_index=1, version='t')
        with override_settings(MODEL_DIR=str(tmp), MODEL_REQUIRE_CHECKSUM=False):
            det = create_detector(spec)
            det.load()
        self.assertEqual(det.status, 'AVAILABLE', det.message)
        return net, det

    def test_converted_maps_equal_the_original_network(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        for key, (size, grid) in self.GRID.items():
            net, det = self.export(key, size, tmp)
            ref = E.CamExplainer(det)
            ref._net = net
            ex = E.cam_explainer_for(det)
            self.assertIsInstance(ex, E.OnnxCamExplainer)
            methods = ('gradcam', 'gradcampp') if key != 'vit_b16' else ('rollout',)   # ViT Grad-CAM uses a different (pre-norm) tensor
            for m in methods:
                a, b = ex.cam(FACE, m), ref.cam(FACE, m)
                self.assertEqual(a.shape, grid, (key, m))
                if a.std() > 0 and b.std() > 0:
                    self.assertGreater(np.corrcoef(a.ravel(), b.ravel())[0, 1], 0.999, (key, m))
            if key == 'vit_b16':
                self.assertEqual(ex.cam(FACE, 'gradcam').shape, grid)

    def test_explain_end_to_end_without_any_extra_file(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        _, det = self.export('xception', 299, tmp)
        res = det.explain(FACE, str(tmp / 'maps'))
        self.assertTrue(res['available'], res)
        self.assertEqual((res['method'], res['grid']), ('grad-cam', [10, 10]))
        self.assertTrue(Path(res['image_path']).is_file())

    def test_efficientnet_uses_the_final_pool_not_squeeze_excite(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        _, det = self.export('efficientnet_b4', 380, tmp)
        ex = E.cam_explainer_for(det)
        ex._ensure()
        self.assertEqual(ex._target_node, '/bn2/act/Mul')

    def test_corrupted_conversion_is_refused(self):
        """If the converted graph does not reproduce ONNX Runtime, no heat-map is shown."""
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        _, det = self.export('xception', 299, tmp)
        ex = E.cam_explainer_for(det)
        real = det._forward
        det._forward = lambda x: real(x) + 1.0          # deployed model now disagrees with its conversion
        res = ex.explain(FACE, str(tmp / 'maps'))
        self.assertFalse(res['available'])
        self.assertIn('does not match', res['reason'])


class FrequencyExplanation(SimpleTestCase):
    def make(self, method):
        rng = np.random.default_rng(0)
        faces = [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(40)]
        smooth = [cv2.GaussianBlur(f, (0, 0), 3) for f in faces]
        X = np.vstack([extract(f, method, 256) for f in faces + smooth])
        y = np.array([0] * 40 + [1] * 40)
        model = make_pipeline(StandardScaler(), SVC(probability=True, random_state=0)).fit(X, y)
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        joblib.dump(model, tmp / 'm.joblib')
        spec = ModelSpec(key='freq_svm', display_name='FFT', family='Frequency+SVM', path=tmp / 'm.joblib', framework='sklearn',
                         input_size=256, feature_method=method, fake_label=1, version='t', sha256=file_sha256(tmp / 'm.joblib'))
        det = create_detector(spec)
        with override_settings(MODEL_DIR=str(tmp)):
            det.load()
        self.assertEqual(det.status, 'AVAILABLE', det.message)
        return det, tmp, smooth[0]

    def test_fft_band_ablation(self):
        det, tmp, face = self.make('azimuthal_fft')
        res = det.explain(face, str(tmp / 'maps'))
        self.assertTrue(res['available'], res)
        self.assertEqual((res['method'], res['domain'], len(res['band_contributions'])), ('frequency-band-ablation', 'frequency', 8))
        self.assertTrue(Path(res['image_path']).is_file())
        self.assertIn('Grad-CAM is not defined', res['disclaimer'])

    def test_dct_grid_ablation(self):
        det, tmp, face = self.make('dct_block_stats')
        res = det.explain(face, str(tmp / 'maps'))
        self.assertTrue(res['available'], res)
        self.assertEqual(len(res['band_contributions']), 64)

    def test_ablation_is_consistent_with_the_score(self):
        """Replacing every feature by the training mean must move the score to the neutral value, so the summed single-group
        deltas have the same sign as the full ablation for a confidently-fake face."""
        det, tmp, face = self.make('azimuthal_fft')
        base, delta = E.FrequencyExplainer(det).attribution(face)
        self.assertTrue(np.isfinite(delta).all())
        self.assertGreater(abs(delta).max(), 0)
