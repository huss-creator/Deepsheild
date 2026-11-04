"""
Phase 4 tests: model registry/config safety, real ONNX + scikit-learn inference on tiny
throw-away models created inside the tests (NOT shipped), ensemble policy, face
alignment, multi-face pipeline, failure isolation, persistence, evaluation toolkit.
Torch/Keras backends are not executed here (frameworks not installed in CI sandbox).
"""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import onnx
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from onnx import TensorProto, helper
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ai_models import ensemble as ens
from ai_models import evaluation as ev
from ai_models.detectors import file_sha256
from ai_models.explain import overlay_heatmap
from ai_models.face_detection import (Face, HaarDetector, align_and_crop, eye_angle_degrees,
                                      quality_flags)
from ai_models.frequency_features import azimuthal_fft, extract
from ai_models.pipeline import load_processing_image, run_ai_analysis
from ai_models.registry import ModelRegistry, build_specs
from ai_models.types import AVAILABLE, DISABLED, FAILED, NOT_CONFIGURED, TIMEOUT, ModelOutput
from detector.models import UploadedImage
from detector.services import persist_ai_results

BRIGHT = np.full((80, 80, 3), 230, np.uint8)
DARK = np.full((80, 80, 3), 25, np.uint8)


def make_onnx(path, size=32, outputs=1, softmax=False, w=10.0, b=-15.0):
    """score = sigmoid(w * sum(mean RGB in 0..1) + b): bright -> ~1, dark -> ~0."""
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, size, size])
    nodes = [helper.make_node('ReduceMean', ['x'], ['m'], axes=[2, 3], keepdims=0)]
    if outputs == 1:
        W = helper.make_tensor('W', TensorProto.FLOAT, [3, 1], [w] * 3)
        B = helper.make_tensor('B', TensorProto.FLOAT, [1], [b])
        nodes += [helper.make_node('MatMul', ['m', 'W'], ['z']), helper.make_node('Add', ['z', 'B'], ['a']),
                  helper.make_node('Sigmoid', ['a'], ['y'])]
        y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1])
    else:
        W = helper.make_tensor('W', TensorProto.FLOAT, [3, 2], [0, w, 0, w, 0, w])
        B = helper.make_tensor('B', TensorProto.FLOAT, [2], [0, b])
        nodes += [helper.make_node('MatMul', ['m', 'W'], ['z']),
                  helper.make_node('Add', ['z', 'B'], ['a' if softmax else 'y'])]
        if softmax:
            nodes.append(helper.make_node('Softmax', ['a'], ['y'], axis=1))
        y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2])
    g = helper.make_graph(nodes, 'tiny', [x], [y], [W, B])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 8
    onnx.save(m, str(path))


class ModelDirTestCase(TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix='ds_models_'))
        self.media = tempfile.mkdtemp(prefix='ds_media_')
        self.ov = override_settings(MODEL_DIR=self.dir, MODEL_CONFIG_FILE=str(self.dir / 'models.json'),
                                    MODEL_REQUIRE_CHECKSUM=False, MEDIA_ROOT=self.media,
                                    ENSEMBLE_CONFIG_PATH=str(self.dir / 'none.json'))
        self.ov.enable()
        self.cfg = {}

    def tearDown(self):
        self.ov.disable()
        shutil.rmtree(self.dir, ignore_errors=True)
        shutil.rmtree(self.media, ignore_errors=True)

    def add_onnx(self, key, outputs=1, softmax=False, size=32, **cfg):
        folder = self.dir / key
        folder.mkdir(exist_ok=True)
        f = folder / 'model.onnx'
        make_onnx(f, size=size, outputs=outputs, softmax=softmax)
        base = {'file': f'{key}/model.onnx', 'framework': 'onnx', 'input_size': size,
                'normalization': 'zero_one', 'version': '9.9.9', 'sha256': file_sha256(f),
                'output_mode': 'sigmoid' if outputs == 1 else ('softmax' if softmax else 'logits')}
        if outputs == 2:
            base['fake_index'] = 1
        base.update(cfg)
        self.cfg[key] = base
        return f

    def registry(self):
        (self.dir / 'models.json').write_text(json.dumps(self.cfg))
        reg = ModelRegistry(build_specs())
        reg.load_all()
        return reg


# ============================================================== registry / onnx
class RegistryAndOnnxTests(ModelDirTestCase):
    def test_no_files_means_not_configured_and_no_fake_scores(self):
        reg = self.registry()
        self.assertEqual({d.status for d in reg.ordered()}, {NOT_CONFIGURED})
        out = reg.detectors['vit_b16'].predict(BRIGHT)
        self.assertEqual(out.status, NOT_CONFIGURED)
        self.assertIsNone(out.score)
        self.assertIsNone(out.label)

    def test_models_json_with_windows_bom_is_read(self):
        self.add_onnx('vit_b16', version='1.2.3', threshold=0.5)
        (self.dir / 'models.json').write_bytes(b'\xef\xbb\xbf' + json.dumps(self.cfg).encode('utf-8'))
        reg = ModelRegistry(build_specs()); reg.load_all()
        self.assertEqual(reg.detectors['vit_b16'].status, AVAILABLE)

    def test_invalid_models_json_is_reported_not_crashing(self):
        (self.dir / 'models.json').write_text('{"efficientnet_b4": {,}')
        with self.assertLogs('deepshield.models', level='ERROR') as cm:
            reg = ModelRegistry(build_specs()); reg.load_all()
        self.assertIn('invalid JSON', cm.output[0])
        self.assertEqual({d.status for d in reg.ordered()}, {NOT_CONFIGURED})

    def test_onnx_sigmoid_model_runs_real_inference(self):
        self.add_onnx('vit_b16', version='1.2.3', threshold=0.5)
        d = self.registry().detectors['vit_b16']
        self.assertEqual(d.status, AVAILABLE)
        hi, lo = d.predict(BRIGHT), d.predict(DARK)
        self.assertGreater(hi.score, 0.95)
        self.assertLess(lo.score, 0.05)
        self.assertEqual((hi.label, lo.label), ('MANIPULATED', 'AUTHENTIC'))
        self.assertEqual(hi.version, '1.2.3')
        self.assertEqual(len(hi.weights_sha256), 64)
        self.assertGreater(hi.inference_ms, 0)
        self.assertEqual(d.layout, 'NCHW')          # detected from the model, not guessed

    def test_class_mapping_uses_fake_index(self):
        self.add_onnx('xception', outputs=2, softmax=True, fake_index=1)
        self.add_onnx('efficientnet_b4', outputs=2, softmax=True, fake_index=0)
        reg = self.registry()
        a, b = reg.detectors['xception'].predict(BRIGHT), reg.detectors['efficientnet_b4'].predict(BRIGHT)
        self.assertAlmostEqual(a.score + b.score, 1.0, places=4)
        self.assertGreater(a.score, 0.95)

    def test_logits_mode_applies_softmax(self):
        self.add_onnx('efficientnet_b4', outputs=2)
        out = self.registry().detectors['efficientnet_b4'].predict(BRIGHT)
        self.assertEqual(out.status, AVAILABLE)
        self.assertGreater(out.score, 0.95)

    def test_threshold_is_per_model(self):
        self.add_onnx('vit_b16', threshold=0.9999999)
        self.assertEqual(self.registry().detectors['vit_b16'].predict(BRIGHT).label, 'AUTHENTIC')

    def test_checksum_mismatch_fails_without_crashing(self):
        self.add_onnx('vit_b16', sha256='0' * 64)
        d = self.registry().detectors['vit_b16']
        self.assertEqual(d.status, FAILED)
        self.assertIn('checksum', d.message)
        self.assertEqual(d.predict(BRIGHT).status, FAILED)

    def test_missing_semantics_are_refused_not_guessed(self):
        self.add_onnx('vit_b16', normalization='', output_mode='')
        d = self.registry().detectors['vit_b16']
        self.assertEqual(d.status, FAILED)
        self.assertIn('normalization', d.message)
        self.assertIn('output_mode', d.message)

    def test_fake_index_required_for_softmax(self):
        self.add_onnx('xception', outputs=2, softmax=True)
        self.cfg['xception'].pop('fake_index')
        self.assertIn('fake_index', self.registry().detectors['xception'].message)

    def test_input_size_mismatch_detected(self):
        self.add_onnx('vit_b16', size=32, input_size=64)
        d = self.registry().detectors['vit_b16']
        self.assertEqual(d.status, FAILED)
        self.assertIn('input_size', d.message)

    def test_output_mode_mismatch_detected_by_self_test(self):
        self.add_onnx('xception', outputs=2, softmax=True, output_mode='sigmoid')
        d = self.registry().detectors['xception']
        self.assertEqual(d.status, FAILED)

    def test_model_outside_model_dir_refused(self):
        other = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, other, True)
        make_onnx(other / 'm.onnx')
        self.add_onnx('vit_b16', file=str(other / 'm.onnx'))
        d = self.registry().detectors['vit_b16']
        self.assertEqual(d.status, FAILED)
        self.assertIn('MODEL_DIR', d.message)

    def test_disabled_model(self):
        self.add_onnx('vit_b16', enabled=False)
        self.assertEqual(self.registry().detectors['vit_b16'].status, DISABLED)

    def test_env_path_override_and_checksum_required_setting(self):
        self.add_onnx('vit_b16')
        self.cfg['vit_b16'].pop('sha256')
        with override_settings(MODEL_REQUIRE_CHECKSUM=True):
            d = self.registry().detectors['vit_b16']
        self.assertEqual(d.status, FAILED)
        self.assertIn('sha256', d.message)

    def test_health_check_exposes_no_paths(self):
        self.add_onnx('vit_b16')
        blob = json.dumps(self.registry().health())
        self.assertNotIn(str(self.dir), blob)
        self.assertNotIn('model.onnx', blob)

    def test_metadata_for_reproducibility(self):
        self.add_onnx('vit_b16', training={'dataset': 'DS', 'dataset_version': '3', 'date': '2026-01-01',
                                            'parameters': {'lr': 1e-4}, 'metrics': {'auc': 0.9}})
        meta = self.registry().specs['vit_b16'].metadata()
        for k in ('model_name', 'model_version', 'training_dataset', 'training_dataset_version',
                  'training_date', 'framework', 'input_size', 'preprocessing_version',
                  'threshold_version', 'weights_checksum', 'training_parameters', 'evaluation_metrics'):
            self.assertIn(k, meta)
        self.assertEqual(meta['training_dataset'], 'DS')


# ============================================================ frequency + SVM
class FrequencySVMTests(ModelDirTestCase):
    SIZE = 64

    def _faces(self, n, noisy, seed):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            a = rng.integers(0, 255, (self.SIZE, self.SIZE), dtype=np.uint8)
            if not noisy:
                a = cv2.GaussianBlur(a, (0, 0), 3)
            out.append(np.stack([a] * 3, -1))
        return out

    def _train(self, path):
        X = [extract(f, 'azimuthal_fft', self.SIZE)[0] for f in self._faces(30, False, 1) + self._faces(30, True, 2)]
        y = [0] * 30 + [1] * 30
        joblib.dump(make_pipeline(StandardScaler(), SVC(probability=True, random_state=0)).fit(X, y), path)

    def _cfg(self, **kw):
        (self.dir / 'freq_svm').mkdir(exist_ok=True)
        f = self.dir / 'freq_svm' / 'model.joblib'
        self._train(f)
        self.cfg['freq_svm'] = {'file': 'freq_svm/model.joblib', 'framework': 'sklearn', 'input_size': self.SIZE,
                                'feature_method': 'azimuthal_fft', 'fake_label': 1, 'version': '0.1',
                                'sha256': file_sha256(f), **kw}

    def test_features_are_deterministic(self):
        f = self._faces(1, True, 5)[0]
        self.assertTrue(np.array_equal(extract(f, 'azimuthal_fft', 64), extract(f, 'azimuthal_fft', 64)))
        self.assertTrue(np.all(np.isfinite(azimuthal_fft(np.zeros((50, 50, 3), np.uint8), 64))))

    def test_svm_scores_real_signal(self):
        self._cfg()
        d = self.registry().detectors['freq_svm']
        self.assertEqual(d.status, AVAILABLE, d.message)
        noisy = np.mean([d.predict(f).score for f in self._faces(10, True, 99)])
        smooth = np.mean([d.predict(f).score for f in self._faces(10, False, 98)])
        self.assertGreater(noisy, 0.8)
        self.assertLess(smooth, 0.2)
        self.assertIn('Platt', d.predict(self._faces(1, True, 7)[0]).score_type)

    def test_pickle_requires_checksum(self):
        self._cfg(sha256='')
        d = self.registry().detectors['freq_svm']
        self.assertEqual(d.status, FAILED)
        self.assertIn('sha256', d.message)

    def test_tampered_pickle_not_loaded(self):
        self._cfg()
        (self.dir / 'freq_svm' / 'model.joblib').write_bytes(b'evil pickle')
        d = self.registry().detectors['freq_svm']
        self.assertEqual(d.status, FAILED)
        self.assertIn('checksum', d.message)

    def test_unknown_feature_method_fails(self):
        self._cfg(feature_method='made_up')
        self.assertEqual(self.registry().detectors['freq_svm'].status, FAILED)

    def test_wrong_fake_label_fails(self):
        self._cfg(fake_label=7)
        self.assertEqual(self.registry().detectors['freq_svm'].status, FAILED)


# ================================================================== ensemble
def mo(key, score, status=AVAILABLE, thr=0.5):
    return ModelOutput(key=key, name=key, status=status, score=score if status == AVAILABLE else None,
                       threshold=thr)


class EnsembleTests(TestCase):
    def cfg(self, **kw):
        return ens.EnsembleConfig(**kw)

    def test_clear_manipulated_and_authentic(self):
        hi = ens.combine([mo(k, .9) for k in 'abcd'], self.cfg())
        lo = ens.combine([mo(k, .1) for k in 'abcd'], self.cfg())
        self.assertEqual(hi['label'], ens.POTENTIALLY_MANIPULATED)
        self.assertEqual(lo['label'], ens.AUTHENTIC)
        self.assertEqual(hi['manipulated_votes'], 4)

    def test_unavailable_models_do_not_contribute(self):
        r = ens.combine([mo('a', .9), mo('b', .9), mo('c', .9), mo('d', 0, NOT_CONFIGURED),
                         mo('e', 0, FAILED), mo('f', 0, TIMEOUT)], self.cfg(), expected_models=6)
        self.assertEqual((r['models_available'], r['models_expected']), (3, 6))
        self.assertEqual(r['label'], ens.POTENTIALLY_MANIPULATED)

    def test_no_models_gives_uncertain_without_score(self):
        r = ens.combine([mo(k, 0, NOT_CONFIGURED) for k in 'abcdef'], self.cfg())
        self.assertEqual(r['label'], ens.UNCERTAIN)
        self.assertIsNone(r['score'])
        self.assertIn('No AI model produced a result', r['explanation'])

    def test_too_few_models_is_uncertain(self):
        r = ens.combine([mo('a', .95), mo('b', .95)], self.cfg(min_models=3))
        self.assertEqual(r['label'], ens.UNCERTAIN)
        self.assertIn('insufficient_models', r['reasons'])

    def test_disagreement_is_uncertain(self):
        r = ens.combine([mo('a', .95), mo('b', .95), mo('c', .95), mo('d', .05), mo('e', .05), mo('f', .95)],
                        self.cfg())
        self.assertTrue(r['disagreement'])
        self.assertEqual(r['label'], ens.UNCERTAIN)

    def test_poor_face_quality_is_uncertain_even_if_models_agree(self):
        r = ens.combine([mo(k, .95) for k in 'abcd'], self.cfg(), quality_flags=['face_too_small'])
        self.assertEqual(r['label'], ens.UNCERTAIN)
        self.assertIn('poor_face_quality', r['reasons'])

    def test_uncertainty_band(self):
        r = ens.combine([mo(k, .53) for k in 'abcd'], self.cfg())
        self.assertEqual(r['label'], ens.UNCERTAIN)
        self.assertIn('score_in_uncertainty_band', r['reasons'])

    def test_unvalidated_ensemble_never_claims_probability(self):
        r = ens.combine([mo(k, .9) for k in 'abcd'], self.cfg())
        self.assertNotIn('calibrated probability', r['score_kind'])
        self.assertFalse(r['validated'])
        self.assertIn('not been validated', r['explanation'])
        ok = ens.combine([mo(k, .9) for k in 'abcd'], self.cfg(validated=True, calibrated=True))
        self.assertEqual(ok['score_kind'], 'calibrated probability')

    def test_stacking_needs_all_its_inputs(self):
        cfg = self.cfg(method='stacking', validated=True,
                       coefficients={'a': 1.0, 'b': 1.0, 'c': 1.0, 'd': 1.0}, intercept=-2.0, min_models=1)
        full = ens.combine([mo(k, .9) for k in 'abcd'], cfg)
        self.assertEqual(full['label'], ens.POTENTIALLY_MANIPULATED)
        part = ens.combine([mo(k, .9) for k in 'abc'], cfg)
        self.assertEqual(part['label'], ens.UNCERTAIN)
        self.assertIn('stacking_inputs_missing', part['reasons'])

    def test_weighted_method(self):
        cfg = self.cfg(method='weighted', weights={'a': 1.0, 'b': 0.0, 'c': 0.0}, min_models=1)
        self.assertAlmostEqual(ens.combine([mo('a', .9), mo('b', .1), mo('c', .1)], cfg, )['score'], .9, 3)

    def test_image_level_summary(self):
        def f(l, s=.5):
            return {'label': l, 'score': s, 'explanation': 'x'}
        self.assertEqual(ens.summarise_faces([])['label'], ens.NO_FACE)
        s = ens.summarise_faces([f(ens.AUTHENTIC, .1), f(ens.POTENTIALLY_MANIPULATED, .9), f(ens.UNCERTAIN)])
        self.assertEqual((s['label'], s['representative_face']), (ens.POTENTIALLY_MANIPULATED, 1))
        self.assertIn('3 faces analysed independently', s['explanation'])

    def test_config_file_is_loaded(self):
        p = Path(tempfile.mkdtemp()) / 'e.json'
        p.write_text(json.dumps({'method': 'weighted', 'validated': True, 'decision_threshold': .6,
                                 'unknown_key': 1}))
        c = ens.load_ensemble_config(str(p))
        self.assertEqual((c.method, c.validated, c.decision_threshold), ('weighted', True, .6))
        self.assertFalse(ens.load_ensemble_config('/nonexistent.json').validated)


# ============================================================ faces / pipeline
class StubDetector:
    name = 'stub'
    supports_landmarks = False

    def __init__(self, faces):
        self._faces = faces

    def detect(self, rgb):
        return self._faces


class FaceTests(TestCase):
    def test_alignment_levels_the_eyes(self):
        lm = np.array([[100, 100], [200, 130], [150, 150], [110, 190], [190, 200]], np.float32)
        angle = eye_angle_degrees(lm)
        M = cv2.getRotationMatrix2D((150, 115), angle, 1.0)
        pts = cv2.transform(lm[None, :2], M)[0]
        self.assertAlmostEqual(float(pts[0][1]), float(pts[1][1]), delta=0.01)

    def test_align_and_crop_with_and_without_landmarks(self):
        img = np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8)
        plain, aligned = align_and_crop(img, Face(0, 100, 100, 100, 100))
        self.assertFalse(aligned)
        self.assertEqual(plain.shape[:2], (140, 140))               # 20% padding each side
        tilted = Face(0, 100, 100, 100, 100, 0.9, np.array([[120, 120], [180, 140], [150, 160], [125, 185], [175, 190]], np.float32))
        crop, aligned = align_and_crop(img, tilted)
        self.assertTrue(aligned)
        self.assertGreater(crop.size, 0)

    def test_quality_flags(self):
        self.assertIn('face_too_small', quality_flags(Face(0, 100, 100, 40, 40), (400, 400, 3)))
        self.assertIn('face_near_image_edge', quality_flags(Face(0, 0, 100, 100, 100), (400, 400, 3)))
        self.assertEqual(quality_flags(Face(0, 100, 100, 100, 100), (400, 400, 3)), [])

    def test_haar_finds_nothing_in_noise(self):
        noise = np.random.default_rng(3).integers(0, 255, (240, 240, 3), dtype=np.uint8)
        self.assertEqual(HaarDetector().detect(noise), [])

    def test_haar_rejects_background_candidate_without_plausible_eyes(self):
        class FaceCascade:
            def __init__(self, boxes):
                self.boxes = np.asarray(boxes, dtype=np.int32)

            def detectMultiScale(self, *args, **kwargs):
                return self.boxes

        class EyeCascade:
            def empty(self):
                return False

            def detectMultiScale(self, roi, *args, **kwargs):
                if roi.mean() > 180:
                    return np.asarray([[15, 10, 10, 10], [50, 10, 10, 10]], dtype=np.int32)
                return np.empty((0, 4), dtype=np.int32)

        detector = HaarDetector.__new__(HaarDetector)
        detector._cascades = [
            FaceCascade([[10, 10, 80, 80], [120, 10, 80, 80]]),
            FaceCascade([[12, 12, 76, 76]]),  # duplicate detection of the first face
        ]
        detector._profile_cascade = None
        detector._eye_cascade = EyeCascade()
        image = np.zeros((200, 250, 3), dtype=np.uint8)
        image[10:90, 10:90] = 220

        with override_settings(FACE_MIN_SIZE=30, FACE_MIN_AREA_RATIO=0.005, FACE_HAAR_MIN_NEIGHBORS=7,
                              FACE_HAAR_VALIDATE_EYES=True, AI_MAX_FACES=10):
            faces = detector.detect(image)

        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0].bbox, (10, 10, 80, 80))

    def test_exif_orientation_is_normalised_in_processing_copy(self):
        exif = Image.Exif(); exif[0x0112] = 6
        p = os.path.join(tempfile.mkdtemp(), 'r.jpg')
        Image.fromarray(np.zeros((100, 200, 3), np.uint8)).save(p, exif=exif)
        rgb, changes = load_processing_image(p)
        self.assertEqual(rgb.shape[:2], (200, 100))
        self.assertIn('exif_orientation_applied', changes)

    def test_transparent_png_is_composited_on_white(self):
        p = os.path.join(tempfile.mkdtemp(), 't.png')
        Image.new('RGBA', (50, 50), (0, 0, 0, 0)).save(p)
        rgb, changes = load_processing_image(p)
        self.assertEqual(int(rgb.min()), 255)
        self.assertIn('alpha_composited_on_white', changes)

    def test_overlay_heatmap(self):
        out = overlay_heatmap(np.zeros((40, 50, 3), np.uint8), np.random.rand(7, 7))
        self.assertEqual((out.shape, out.dtype), ((40, 50, 3), np.uint8))
        self.assertEqual(overlay_heatmap(np.zeros((8, 8, 3), np.uint8), np.zeros((3, 3))).shape, (8, 8, 3))


class PipelineTests(ModelDirTestCase):
    def setUp(self):
        super().setUp()
        self.add_onnx('vit_b16'); self.add_onnx('xception', outputs=2, softmax=True)
        self.add_onnx('efficientnet_b4', outputs=2)
        arr = np.full((300, 420, 3), 128, np.uint8)
        arr[50:190, 50:190] = 235        # face 0: bright
        arr[60:180, 250:370] = 20        # face 1: dark
        self.img = os.path.join(self.dir, 'in.png')
        Image.fromarray(arr).save(self.img)
        self.faces = [Face(0, 50, 50, 140, 140), Face(1, 250, 60, 120, 120)]

    def run_pipeline(self, faces=None, reg=None):
        return run_ai_analysis(self.img, os.path.join(self.media, 'faces'), os.path.join(self.media, 'heatmaps'),
                               os.path.join(self.media, 'processed'), reg or self.registry(),
                               StubDetector(self.faces if faces is None else faces), ens.EnsembleConfig())

    def test_every_face_is_analysed_independently(self):
        ai = self.run_pipeline()
        self.assertEqual(ai['status'], 'completed')
        self.assertEqual(ai['face_count'], 2)
        f0, f1 = ai['faces']
        self.assertEqual(f0['ensemble']['label'], ens.POTENTIALLY_MANIPULATED)
        self.assertEqual(f1['ensemble']['label'], ens.AUTHENTIC)
        self.assertEqual(len(f0['model_outputs']), 5)       # all model slots, including SDXL, are reported
        self.assertEqual(sum(o['status'] == NOT_CONFIGURED for o in f0['model_outputs']), 2)
        self.assertEqual(ai['image_level']['label'], ens.POTENTIALLY_MANIPULATED)
        self.assertEqual(ai['image_level']['representative_face'], 0)
        for f in ai['faces']:
            self.assertTrue(os.path.isfile(os.path.join(self.media, f['crop_path'])))
            self.assertTrue(f['crop_path'].endswith('.png'))        # lossless crops
        self.assertNotEqual(f0['face_id'], f1['face_id'])

    def test_disabled_models_are_not_counted_as_expected(self):
        self.cfg['sdxl_detector'] = {'enabled': False}
        ai = self.run_pipeline()
        face = ai['faces'][0]
        self.assertEqual(face['ensemble']['models_expected'], 4)
        self.assertEqual(face['ensemble']['models_available'], 3)
        self.assertEqual(len(face['model_outputs']), 5)

    def test_timings_and_versions_recorded(self):
        ai = self.run_pipeline()
        for k in ('decode_ms', 'detect_ms', 'ensemble_ms', 'total_ms', 'models_ms'):
            self.assertIn(k, ai['timings_ms'])
        self.assertIn('vit_b16', ai['timings_ms']['models_ms'])
        out = ai['faces'][0]['model_outputs'][0]
        self.assertEqual((out['version'], len(out['weights_sha256'])), ('9.9.9', 64))
        self.assertFalse(ai['ensemble_config']['validated'])

    def test_no_face_means_no_model_runs(self):
        ai = self.run_pipeline(faces=[])
        self.assertEqual((ai['status'], ai['image_level']['label']), ('no_face', ens.NO_FACE))
        self.assertEqual(ai['faces'], [])

    def test_undecodable_image_fails_safely(self):
        bad = os.path.join(self.dir, 'bad.png'); open(bad, 'wb').write(b'nope')
        ai = run_ai_analysis(bad, self.media, self.media, registry=self.registry(), detector=StubDetector([]))
        self.assertEqual(ai['status'], 'failed')
        self.assertNotIn(str(self.dir), json.dumps(ai))

    def test_one_failing_model_does_not_stop_others(self):
        reg = self.registry(); reg.load_all()
        def boom(img): raise RuntimeError('/secret/path exploded')
        reg.detectors['xception']._forward = lambda b: boom(b)
        ai = self.run_pipeline(reg=reg)
        outs = {o['key']: o for o in ai['faces'][0]['model_outputs']}
        self.assertEqual(outs['xception']['status'], FAILED)
        self.assertNotIn('secret', outs['xception']['message'])
        self.assertEqual(outs['vit_b16']['status'], AVAILABLE)
        self.assertEqual(ai['faces'][0]['ensemble']['models_available'], 2)
        self.assertEqual(ai['faces'][0]['ensemble']['label'], ens.UNCERTAIN)   # 2 < min_models

    @override_settings(AI_MODEL_TIMEOUT=0.2)
    def test_slow_model_times_out(self):
        reg = self.registry(); reg.load_all()
        real = reg.detectors['vit_b16']._forward
        reg.detectors['vit_b16']._forward = lambda b: (time.sleep(1.0), real(b))[1]
        ai = self.run_pipeline(faces=self.faces[:1], reg=reg)
        outs = {o['key']: o for o in ai['faces'][0]['model_outputs']}
        self.assertEqual(outs['vit_b16']['status'], TIMEOUT)
        self.assertEqual(outs['xception']['status'], AVAILABLE)

    def test_results_are_persisted_with_face_index_and_versions(self):
        user = User.objects.create_user('p', 'p@e.com', 'Correct-Horse-9')
        up = UploadedImage.objects.create(user=user, image=SimpleUploadedFile('a.png', open(self.img, 'rb').read()),
                                          original_filename='a.png', file_size=1)
        analysis = persist_ai_results(up, self.run_pipeline())
        self.assertEqual(analysis.model_results.count(), 10)                # 2 faces x 5 model slots
        self.assertEqual(analysis.final_prediction, 'LIKELY_MANIPULATED')
        self.assertEqual(analysis.executed_models, 3)
        self.assertEqual(analysis.unavailable_models, 2)
        primary = list(analysis.primary_model_results())
        self.assertEqual(len(primary), 5)
        self.assertTrue(all(m.face_index == 0 for m in primary))
        ok = [m for m in primary if m.status == 'success']
        self.assertTrue(all(m.details['version'] == '9.9.9' and m.details['weights_sha256'] for m in ok))
        nc = [m for m in primary if m.status == 'not_configured']
        self.assertTrue(all(m.fake_probability == 0 and m.prediction == 'NOT_CONFIGURED' for m in nc))
        self.assertEqual(len(analysis.pipeline_info['faces']), 2)
        self.assertIn('timings_ms', analysis.pipeline_info)

    def test_no_models_configured_pipeline_is_honest(self):
        (self.dir / 'models.json').write_text('{}')
        for k in ('vit_b16', 'xception', 'efficientnet_b4'):
            shutil.rmtree(self.dir / k)
        reg = ModelRegistry(build_specs())
        reg.load_all()
        ai = self.run_pipeline(reg=reg)
        f = ai['faces'][0]['ensemble']
        self.assertEqual((f['label'], f['score'], f['models_available']), (ens.UNCERTAIN, None, 0))


# ============================================================== evaluation
class EvaluationTests(TestCase):
    def test_metrics_known_values(self):
        m = ev.compute_metrics([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1], 0.5)
        self.assertEqual(m['confusion_matrix'], {'tn': 1, 'fp': 1, 'fn': 1, 'tp': 1})
        self.assertEqual((m['accuracy'], m['precision'], m['recall'], m['specificity']), (.5, .5, .5, .5))
        self.assertEqual((m['false_positive_rate'], m['false_negative_rate']), (.5, .5))
        self.assertEqual(m['roc_auc'], .75)

    def test_single_class_does_not_crash(self):
        m = ev.compute_metrics([1, 1, 1], [.9, .8, .7])
        self.assertIsNone(m['roc_auc'])
        self.assertIsNone(m['specificity'])

    def test_leakage_audit_detects_duplicates_and_group_overlap(self):
        rows = [{'sha256': 'a' * 64, 'split': 'train', 'group_id': 'v1'},
                {'sha256': 'a' * 64, 'split': 'test', 'group_id': 'v2'},
                {'sha256': 'b' * 64, 'split': 'validation', 'group_id': 'v3'},
                {'sha256': 'c' * 64, 'split': 'external', 'group_id': 'v3'},
                {'sha256': 'd' * 64, 'split': 'test', 'group_id': ''}]
        a = ev.leakage_audit(rows)
        types = {p['type'] for p in a['problems']}
        self.assertFalse(a['ok'])
        self.assertEqual(types, {'duplicate_file_across_splits', 'group_across_splits'})
        self.assertTrue(a['warnings'])
        clean = ev.leakage_audit([{'sha256': 'a', 'split': 'train', 'group_id': 'x'},
                                  {'sha256': 'b', 'split': 'test', 'group_id': 'y'}])
        self.assertTrue(clean['ok'])

    def test_threshold_uses_validation_data_only(self):
        yv, sv = [0] * 50 + [1] * 50, [.1] * 50 + [.9] * 50
        t = ev.select_threshold(yv, sv)
        self.assertTrue(.1 < t <= .9)
        self.assertEqual(ev.select_threshold([1, 1], [.5, .6]), .5)       # single class fallback

    def test_run_evaluation_tunes_on_validation_and_reports_gap(self):
        rng = np.random.default_rng(0)
        rows = []
        for sp, n, noise in (('validation', 80, .15), ('test', 80, .2), ('external', 80, .45)):
            for i in range(n):
                y = i % 2
                rows.append({'split': sp, 'label': y, 'sha256': f'{sp}{i}', 'group_id': f'{sp}{i}',
                             'sig': float(np.clip(.5 + (y - .5) * (1 - 2 * noise) + rng.normal(0, noise), 0, 1))})
        rep = ev.run_evaluation(rows, lambda r: {'m': r['sig']}, ['m'])
        m = rep['models']['m']
        self.assertEqual(m['threshold_tuned_on'], 'validation')
        self.assertEqual(set(m['splits']), {'validation', 'test', 'external'})
        self.assertGreater(m['generalisation_gap_auc'], 0)            # external is worse: reported, not hidden
        self.assertTrue(rep['leakage_audit']['ok'])

    def test_eer_and_tpr_at_fixed_fpr(self):
        y = [0] * 100 + [1] * 100
        perfect = ev.eer_and_tpr(y, [.1] * 100 + [.9] * 100)
        self.assertEqual((perfect['eer'], perfect['tpr_at_fpr']), (0.0, 1.0))
        chance = ev.eer_and_tpr(y, list(np.random.default_rng(0).random(200)))
        self.assertTrue(.35 < chance['eer'] < .65)
        self.assertIsNone(ev.eer_and_tpr([1, 1, 1], [.2, .5, .9])['eer'])      # single class: not computable

    def test_apply_ensemble_mean_and_stacking(self):
        sc = {'a': [.2, .8], 'b': [.4, .6]}
        np.testing.assert_allclose(ev.apply_ensemble({'method': 'mean'}, ['a', 'b'], sc), [.3, .7])
        cfg = {'method': 'stacking', 'coefficients': {'a': 1.0, 'b': 0.0}, 'intercept': 0.0}
        np.testing.assert_allclose(ev.apply_ensemble(cfg, ['a', 'b'], sc), [.2, .8], atol=1e-6)   # sigmoid(logit(p)) == p

    def test_fit_ensemble_is_reproducible_and_validated(self):
        rng = np.random.default_rng(1)
        y = np.array([0, 1] * 60)
        s = {k: np.clip(.5 + (y - .5) * .5 + rng.normal(0, .2, len(y)), .01, .99).tolist() for k in 'abc'}
        c1, c2 = ev.fit_ensemble(s, y, seed=7), ev.fit_ensemble(s, y, seed=7)
        self.assertEqual((c1['method'], c1['decision_threshold']), (c2['method'], c2['decision_threshold']))
        self.assertTrue(c1['validated'])
        self.assertFalse(c1['calibrated'])                            # no test split => never claimed
        with self.assertRaises(ValueError):
            ev.fit_ensemble({'a': [.1] * 5}, [0] * 5)

    def test_manifest_labels(self):
        self.assertEqual((ev.parse_label('Fake'), ev.parse_label('real'), ev.parse_label('1')), (1, 0, 1))
        with self.assertRaises(ValueError):
            ev.parse_label('maybe')


class EvidenceNoFabricatedAIScoreTests(TestCase):
    """Regression: with no AI verdict the old code injected a made-up 50/100 AI score."""

    def _call(self, pred):
        from forensics.evidence import calculate_evidence_scores
        return calculate_evidence_scores(0.0, pred, 'UNAVAILABLE', 'UNAVAILABLE', 'UNAVAILABLE',
                                         False, True, 'PNG')

    def test_no_verdict_contributes_nothing(self):
        for pred in ('UNCERTAIN', 'NO_FACE'):
            r = self._call(pred)
            self.assertEqual(r['evidence_ai'], 0.0, pred)
            self.assertEqual(r['overall_risk_score'], 0.0, pred)
            self.assertEqual(r['overall_risk'], 'LOW', pred)
