"""
Phase 5 tests: forensic modules (ELA, frequency, resampling, copy-move, metadata, C2PA),
the transparent evidence report, and view integration. Fixtures are synthetic textures so the
suite needs no external data; real-photo validation lives in `manage.py benchmark_forensics`.
"""

import json
import os
import shutil
import sys
import tempfile
import types
from unittest import mock

import cv2
import numpy as np
from django.test import TestCase
from django.urls import reverse
from PIL import Image

from detector.models import ForensicAnalysis, UploadedImage
from detector.tests_security import TempMediaTestCase, make_image_bytes
from forensics import evidence as ev
from forensics.c2pa_check import classify_c2pa_error, run_c2pa_check
from forensics.catalog import CATALOG
from forensics.copy_move import run_copy_move_analysis
from forensics.ela import run_ela_analysis
from forensics.frequency import run_frequency_analysis
from forensics.metadata import match_editing_software
from forensics.resampling import run_resampling_analysis


def texture(seed, n=320):
    r = np.random.default_rng(seed)
    a = cv2.GaussianBlur(r.normal(128, 50, (n, n)).astype(np.float32), (0, 0), 1.6)
    a += cv2.GaussianBlur(r.normal(0, 40, (n, n)).astype(np.float32), (0, 0), 6)
    return np.stack([np.clip(a + c * 8, 0, 255) for c in range(3)], -1).astype(np.uint8)


class FileTestCase(TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix='ds_forensic_')
        self.addCleanup(shutil.rmtree, self.d, True)

    def save(self, arr, name, fmt='PNG', **kw):
        p = os.path.join(self.d, name)
        Image.fromarray(arr).save(p, fmt, **kw)
        return p


# ------------------------------------------------------------------------- ELA
class ELATests(FileTestCase):
    def test_not_applicable_to_png_instead_of_a_meaningless_score(self):
        r = run_ela_analysis(self.save(texture(1), 'a.png'), self.d)
        self.assertTrue(r['success'])
        self.assertEqual(r['ela_level'], 'UNAVAILABLE')
        self.assertFalse(r['details']['applicable'])
        self.assertIn('not applicable', r['ela_interpretation'])
        self.assertIsNone(r['ela_image_path'])

    def test_jpeg_reports_raw_statistics_not_double_amplified(self):
        r = run_ela_analysis(self.save(texture(1), 'a.jpg', 'JPEG', quality=92), self.d)
        self.assertEqual(r['ela_level'], 'LOW')
        d = r['details']
        self.assertTrue(d['applicable'])
        self.assertLess(d['mean_error'], 5)               # raw 0-255 error, not an amplified value
        self.assertTrue(r['ela_image_path'].endswith('.png'))
        self.assertTrue(0 <= r['ela_score'] <= 100)

    def test_parameters_are_configurable_and_recorded(self):
        p = self.save(texture(2), 'a.jpg', 'JPEG', quality=92)
        r1 = run_ela_analysis(p, self.d, quality=75, scale=10, mode='gray')
        self.assertEqual(r1['details']['quality'], 75)
        self.assertEqual(r1['details']['display_gain'], 10.0)
        self.assertGreater(run_ela_analysis(p, self.d, quality=70)['details']['mean_error'],
                           run_ela_analysis(p, self.d, quality=95)['details']['mean_error'])

    def test_never_reports_high_and_always_states_limits(self):
        r = run_ela_analysis(self.save(texture(3), 'a.jpg', 'JPEG', quality=80), self.d)
        self.assertIn(r['ela_level'], ('LOW', 'MEDIUM'))
        self.assertIn('not proof of manipulation', r['ela_interpretation'])
        self.assertEqual(r['details']['reliability'], 'low')

    def test_no_temp_file_leak(self):
        before = set(os.listdir(tempfile.gettempdir()))
        run_ela_analysis(self.save(texture(4), 'a.jpg', 'JPEG', quality=90), self.d)
        leaked = [f for f in set(os.listdir(tempfile.gettempdir())) - before if f.endswith('.jpg')]
        self.assertEqual(leaked, [])

    def test_corrupt_input_fails_safely(self):
        p = os.path.join(self.d, 'bad.jpg'); open(p, 'wb').write(b'not an image')
        r = run_ela_analysis(p, self.d)
        self.assertFalse(r['success'])
        self.assertNotIn(self.d, r['error'])


# ------------------------------------------------------------------- frequency
class FrequencyTests(FileTestCase):
    def _gray(self, seed=4, n=384):
        return cv2.cvtColor(texture(seed, n), cv2.COLOR_RGB2GRAY)

    def test_detects_injected_periodic_artifact_in_lossless_image(self):
        g = self._gray()
        h, w = g.shape
        pat = ((np.indices((h, w)).sum(0)) % 2 * 2 - 1) * 3
        clean = run_frequency_analysis(self.save(np.stack([g] * 3, -1), 'c.png'), self.d)
        art = run_frequency_analysis(self.save(np.stack([np.clip(g + pat, 0, 255).astype(np.uint8)] * 3, -1), 'a.png'), self.d)
        self.assertEqual(clean['freq_level'], 'LOW')
        self.assertEqual(clean['details']['peak_count'], 0)
        self.assertEqual(art['freq_level'], 'MEDIUM')
        self.assertGreaterEqual(art['details']['peak_count'], 1)

    def test_no_resize_native_window_used(self):
        r = run_frequency_analysis(self.save(texture(1, 300), 'a.png'), self.d)
        self.assertEqual(r['details']['window'], [300, 300])     # old code forced 512x512 Lanczos

    def test_white_noise_and_jpeg_do_not_false_alarm(self):
        wn = np.random.default_rng(0).integers(0, 255, (384, 384, 3), dtype=np.uint8)
        self.assertEqual(run_frequency_analysis(self.save(wn, 'w.png'), self.d)['freq_level'], 'LOW')
        j = run_frequency_analysis(self.save(texture(4, 384), 'j.jpg', 'JPEG', quality=60), self.d)
        self.assertEqual(j['freq_level'], 'LOW')
        self.assertTrue(j['details']['jpeg_grid_excluded'])

    def test_spectral_leakage_from_gradients_and_edges_is_windowed_out(self):
        g = cv2.cvtColor(texture(4, 384), cv2.COLOR_RGB2GRAY).astype(np.float32)
        ramp = np.linspace(0, 120, 384, dtype=np.float32)[None, :] + np.linspace(0, 60, 384, dtype=np.float32)[:, None]
        b = np.clip(g * 0.3 + ramp, 0, 255).astype(np.uint8)
        b[100:300, 150:151] = 255                         # sharp, non-periodic edge
        r = run_frequency_analysis(self.save(np.stack([b] * 3, -1), 'leak.png'), self.d)
        self.assertEqual(r['details']['peak_count'], 0)    # without a Hann window this gives ~38 false peaks

    def test_small_image_reported_unavailable_not_scored(self):
        r = run_frequency_analysis(self.save(texture(1, 100), 's.png'), self.d)
        self.assertTrue(r['success'])
        self.assertEqual(r['freq_level'], 'UNAVAILABLE')

    def test_limitations_stated(self):
        r = run_frequency_analysis(self.save(texture(1), 'a.png'), self.d)
        self.assertIn('absence of peaks proves nothing', r['freq_interpretation'])


# ------------------------------------------------------------------ resampling
class ResamplingTests(FileTestCase):
    def test_resized_image_scores_far_above_native(self):
        a = texture(1)
        native = run_resampling_analysis(self.save(a, 'n.png'), self.d)
        up = cv2.resize(a, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)[:320, :320]
        res = run_resampling_analysis(self.save(up, 'r.png'), self.d)
        self.assertLess(native['resampling_peak_ratio'], 8)
        self.assertGreater(res['resampling_peak_ratio'], 2 * native['resampling_peak_ratio'])
        self.assertEqual((native['resampling_level'], res['resampling_level']), ('LOW', 'MEDIUM'))

    def test_native_resolution_is_analysed_without_prior_resize(self):
        r = run_resampling_analysis(self.save(texture(2, 300), 'a.png'), self.d)
        self.assertEqual(r['details']['window'], [300, 300])

    def test_level_is_capped_and_caveated(self):
        up = cv2.resize(texture(3), None, fx=1.37, fy=1.37)[:320, :320]
        r = run_resampling_analysis(self.save(up, 'r.png'), self.d)
        self.assertNotEqual(r['resampling_level'], 'HIGH')
        self.assertIn('Ordinary editing', r['resampling_interpretation'])


# ------------------------------------------------------------------- copy-move
class CopyMoveTests(FileTestCase):
    def _img(self, seed):
        a = texture(seed, 400)
        return np.clip(a + np.random.default_rng(seed).normal(0, 3, a.shape), 0, 255).astype(np.uint8)

    def test_untouched_image_is_not_flagged(self):
        for s in (1, 2, 3):
            r = run_copy_move_analysis(self.save(self._img(s), f'u{s}.png'), self.d)
            self.assertEqual(r['copy_move_level'], 'LOW', s)
            self.assertEqual(r['details']['regions'], [])

    def test_clone_is_geometrically_verified_and_localised(self):
        a = self._img(1)
        b = a.copy()
        b[250:350, 260:360] = a[40:140, 40:140]                 # source (40,40,100x100) -> target (260,250)
        r = run_copy_move_analysis(self.save(b, 'c.png'), self.d)
        self.assertIn(r['copy_move_level'], ('MEDIUM', 'HIGH'))
        reg = r['details']['regions'][0]
        self.assertGreaterEqual(reg['inliers'], 10)
        cx = lambda bb: (bb[0] + bb[2] / 2, bb[1] + bb[3] / 2)
        inside = lambda c, x0, y0, x1, y1: x0 <= c[0] <= x1 and y0 <= c[1] <= y1
        boxes = [reg['source_bbox'], reg['target_bbox']]
        self.assertTrue(any(inside(cx(bb), 40, 40, 140, 140) for bb in boxes))      # source region
        self.assertTrue(any(inside(cx(bb), 260, 250, 360, 350) for bb in boxes))    # target region
        self.assertTrue(r['copy_move_image_path'].endswith('.png'))

    def test_unrelated_duplicates_are_not_merged_into_one_region(self):
        # 40 small patches each cloned ONCE with different offsets: dozens of ratio-test pairs exist,
        # but no single geometric transform explains them, so nothing may be reported.
        r = np.random.default_rng(7)
        base = texture(17, 900)
        img = base.copy()
        k = 0
        while k < 40:
            sx, sy, dx, dy = r.integers(0, 830, 4)
            if abs(sx - dx) + abs(sy - dy) < 120:
                continue
            img[dy:dy + 64, dx:dx + 64] = base[sy:sy + 64, sx:sx + 64]
            k += 1
        res = run_copy_move_analysis(self.save(img, 'scatter.png'), self.d)
        self.assertGreaterEqual(res['details']['candidate_pairs'], 20)
        self.assertEqual((res['copy_move_level'], res['copy_move_region_count']), ('LOW', 0))

    def test_clone_of_flat_region_is_a_documented_miss(self):
        # ORB needs texture: a cloned flat patch cannot be matched. This is stated in the catalog
        # (false-negative risk) rather than hidden; rotated/scaled clones on real photos are
        # covered by `manage.py benchmark_forensics`.
        self.assertIn('smooth', CATALOG['copy_move']['false_negative'].lower())

    def test_smooth_image_reports_no_evidence_not_an_error(self):
        flat = np.full((200, 200, 3), 120, np.uint8)
        r = run_copy_move_analysis(self.save(flat, 'f.png'), self.d)
        self.assertTrue(r['success'])
        self.assertEqual(r['copy_move_level'], 'LOW')

    def test_limitation_about_natural_repetition_is_stated(self):
        r = run_copy_move_analysis(self.save(self._img(1), 'u.png'), self.d)
        self.assertIn('repetition', r['copy_move_interpretation'])


# ---------------------------------------------------------------- metadata / c2pa
class MetadataMatchingTests(TestCase):
    def test_editing_software_matches_whole_words_only(self):
        for sw in ('Adobe Photoshop 25.1', 'GIMP 2.10.34', 'Midjourney', 'AI generated image', 'Capture One 23'):
            self.assertTrue(match_editing_software(sw), sw)
        for sw in ('Microsoft Paint', 'Fujifilm MainBoard', 'Gaia GPS', 'Canon PowerShot S95',
                   'HP Scanjet', 'Windows Photo Gallery', 'Gimpelsoft Camera', 'Photosynthetic Scanner', '', None):
            self.assertFalse(match_editing_software(sw), sw)


class FakeC2paModule(types.ModuleType):
    def __init__(self, state='Valid', valid=True, raises=None):
        super().__init__('c2pa')
        class Reader:
            def __init__(self, path):
                if raises:
                    raise raises

            def __enter__(self): return self
            def __exit__(self, *a): return False
            def json(self): return json.dumps({'active_manifest': 'm', 'manifests': {'m': {
                'claim_generator': 'TestTool/1', 'title': 't', 'assertions': [{'label': 'c2pa.actions'}]}}})
            def get_validation_state(self): return state
            def is_valid(self): return valid
        self.Reader = Reader


class C2PATests(FileTestCase):
    def _run(self, module):
        p = self.save(texture(1, 64), 'a.png')
        with mock.patch.dict(sys.modules, {'c2pa': module}):
            return run_c2pa_check(p)

    def test_library_missing_is_reported_not_faked(self):
        with mock.patch.dict(sys.modules, {'c2pa': None}):
            r = run_c2pa_check(self.save(texture(1, 64), 'a.png'))
        self.assertEqual(r['c2pa_status'], 'LIBRARY_MISSING')

    def test_no_manifest_means_not_present_with_neutral_wording(self):
        class ManifestNotFound(Exception): pass
        r = self._run(FakeC2paModule(raises=ManifestNotFound('no manifest')))
        self.assertEqual(r['c2pa_status'], 'NOT_PRESENT')
        self.assertIn('No valid Content Credentials were found', r['c2pa_interpretation'])
        self.assertNotIn('fake', r['c2pa_interpretation'].lower().replace('nothing about whether', ''))

    def test_corrupt_manifest_is_invalid_not_absent(self):
        r = self._run(FakeC2paModule(raises=RuntimeError('claim decoding failed: bad CBOR')))
        self.assertEqual(r['c2pa_status'], 'INVALID')
        self.assertTrue(r['c2pa_has_manifest'])

    def test_unknown_error_is_error_with_no_leak(self):
        r = self._run(FakeC2paModule(raises=RuntimeError('disk exploded at /srv/x')))
        self.assertEqual(r['c2pa_status'], 'ERROR')
        self.assertNotIn('/srv', r['error'] + r['c2pa_interpretation'])

    def test_valid_vs_trusted_are_distinguished(self):
        v = self._run(FakeC2paModule(state='Valid'))
        t = self._run(FakeC2paModule(state='Trusted'))
        self.assertEqual((v['c2pa_status'], v['c2pa_signer_trusted']), ('VALID', False))
        self.assertEqual((t['c2pa_status'], t['c2pa_signer_trusted']), ('VALID', True))
        self.assertIn('NOT confirmed', v['c2pa_interpretation'])
        self.assertIn('does not by itself show', v['c2pa_interpretation'])

    def test_failed_validation_is_invalid(self):
        r = self._run(FakeC2paModule(state='Invalid', valid=False))
        self.assertEqual(r['c2pa_status'], 'INVALID')

    def test_error_text_mentioning_manifest_is_not_automatically_absent(self):
        # the old code mapped ANY error containing "manifest" to NOT_PRESENT
        self.assertEqual(classify_c2pa_error(RuntimeError('manifest store is invalid'))[0], 'INVALID')
        r = self._run(FakeC2paModule(raises=RuntimeError('manifest store is invalid')))
        self.assertEqual(r['c2pa_status'], 'INVALID')

    def test_error_classifier(self):
        self.assertEqual(classify_c2pa_error(Exception('ManifestNotFound'))[0], 'NOT_PRESENT')
        self.assertEqual(classify_c2pa_error(Exception('signature mismatch'))[0], 'INVALID')
        self.assertEqual(classify_c2pa_error(Exception('timeout'))[0], 'ERROR')


# ----------------------------------------------------------------- evidence
def lvl(**kw):
    base = {'ela': 'UNAVAILABLE', 'frequency': 'UNAVAILABLE', 'noise': 'UNAVAILABLE', 'copy_move': 'UNAVAILABLE',
            'resampling': 'UNAVAILABLE', 'face_boundary': 'UNAVAILABLE'}
    base.update(kw)
    return {'ela': {'ela_level': base['ela'], 'ela_interpretation': 'x', 'details': {'applicable': base['ela'] != 'UNAVAILABLE'}},
            'frequency': {'freq_level': base['frequency'], 'freq_interpretation': 'x'},
            'noise': {'noise_level': base['noise'], 'noise_interpretation': 'x'},
            'copy_move': {'copy_move_level': base['copy_move'], 'copy_move_interpretation': 'x'},
            'resampling': {'resampling_level': base['resampling'], 'resampling_interpretation': 'x'},
            'face_boundary': {'face_boundary_level': base['face_boundary'], 'face_boundary_interpretation': 'x'},
            'metadata': {'meta_has_exif': False, 'meta_note': 'n'}, 'c2pa': {'c2pa_status': 'NOT_PRESENT'}}


class EvidenceReportTests(TestCase):
    def report(self, ai_label='UNCERTAIN', models=0, **levels):
        return ev.build_evidence_report({'label': ai_label, 'score': None, 'models_available': models},
                                        lvl(**levels), sha256='ab' * 32, file_size=123)

    def test_five_categories_are_kept_separate(self):
        r = self.report()
        for k in ('ai_detection', 'forensic_indicators', 'provenance', 'integrity', 'explainability', 'summary'):
            self.assertIn(k, r)
        self.assertEqual(r['integrity']['algorithm'], 'SHA-256')
        self.assertEqual(r['integrity']['sha256'], 'ab' * 32)
        self.assertNotIn('score', r['integrity'])

    def test_each_indicator_has_required_fields_and_no_fake_confidence(self):
        r = self.report(ela='MEDIUM', noise='HIGH', copy_move='HIGH')
        for key, i in r['forensic_indicators'].items():
            for f in ('status', 'severity', 'description', 'limitations', 'reliability', 'artifact',
                      'false_positive_risk', 'false_negative_risk'):
                self.assertIn(f, i, key)
            self.assertIsNone(i['confidence'], key)

    def test_severity_is_capped_by_measured_reliability(self):
        r = self.report(ela='MEDIUM', noise='HIGH', copy_move='HIGH', frequency='MEDIUM')['forensic_indicators']
        self.assertEqual(r['ela']['severity'], 'info')
        self.assertEqual(r['noise']['severity'], 'low')
        self.assertEqual(r['frequency']['severity'], 'low')
        self.assertEqual(r['copy_move']['severity'], 'moderate')
        self.assertNotIn('high', {i['severity'] for i in r.values()})

    def test_png_ela_is_not_applicable_not_no_indication(self):
        r = ev.build_evidence_report({'label': 'UNCERTAIN'}, {**lvl(), 'ela': {
            'ela_level': 'UNAVAILABLE', 'ela_interpretation': 'n/a', 'details': {'applicable': False}}})
        self.assertEqual(r['forensic_indicators']['ela']['status'], 'not_applicable')

    def test_assessment_rules(self):
        self.assertEqual(self.report('NO_FACE')['summary']['assessment'], 'NO_FACE')
        self.assertEqual(self.report('ANALYSIS_FAILED')['summary']['assessment'], 'FAILED')
        pot = self.report('POTENTIALLY_MANIPULATED', 4, ela='MEDIUM')['summary']
        self.assertEqual(pot['assessment'], 'POTENTIALLY_MANIPULATED')
        self.assertIn('Error Level', ' '.join(pot['reasoning']))
        self.assertEqual(self.report('AUTHENTIC', 4)['summary']['assessment'], 'AUTHENTIC')
        self.assertEqual(self.report('AUTHENTIC', 4, copy_move='MEDIUM')['summary']['assessment'], 'UNCERTAIN')
        self.assertEqual(self.report('AUTHENTIC', 4, ela='MEDIUM', frequency='MEDIUM')['summary']['assessment'], 'UNCERTAIN')
        self.assertEqual(self.report('AUTHENTIC', 4, ela='MEDIUM')['summary']['assessment'], 'AUTHENTIC')

    def test_forensic_indicators_alone_never_establish_manipulation(self):
        r = self.report('UNCERTAIN', 0, copy_move='HIGH', ela='MEDIUM', frequency='MEDIUM', resampling='MEDIUM')
        self.assertEqual(r['summary']['assessment'], 'UNCERTAIN')
        self.assertIn('No AI model produced a result', ' '.join(r['summary']['reasoning']))

    def test_labels_and_limitations_use_careful_wording(self):
        s = self.report('POTENTIALLY_MANIPULATED', 4)['summary']
        self.assertEqual(s['label'], 'POTENTIALLY MANIPULATED')
        blob = json.dumps(s).lower()
        for banned in ('100% fake', 'proof of manipulation.', 'guaranteed', 'definitive'):
            self.assertNotIn(banned, blob.replace('not proof of manipulation', '').replace('never proof', ''))
        self.assertTrue(any('never proof' in l for l in s['limitations']))

    def test_provenance_states_are_distinct_and_neutral(self):
        mk = lambda c2: ev.build_evidence_report({'label': 'UNCERTAIN'}, {**lvl(), 'c2pa': c2})['provenance']['c2pa']
        self.assertEqual(mk({'c2pa_status': 'NOT_PRESENT'})['status'], 'no_credential')
        self.assertEqual(mk({'c2pa_status': 'NOT_PRESENT'})['description'], 'No valid Content Credentials were found.')
        self.assertEqual(mk({'c2pa_status': 'INVALID'})['status'], 'credential_invalid')
        self.assertEqual(mk({'c2pa_status': 'VALID'})['status'], 'credential_valid')
        self.assertEqual(mk({'c2pa_status': 'VALID', 'c2pa_signer_trusted': True})['status'], 'credential_verified')
        self.assertEqual(mk({'c2pa_status': 'LIBRARY_MISSING'})['status'], 'unavailable')

    def test_every_catalog_entry_documents_limits(self):
        for k, c in CATALOG.items():
            for f in ('measures', 'cannot_prove', 'inputs', 'failure', 'false_positive', 'false_negative', 'cost'):
                self.assertTrue(c[f], f'{k}.{f}')


class LegacyEvidenceColumnTests(TestCase):
    def call(self, pred='UNCERTAIN', conf=0.0, exif=False, fmt='JPEG', sw=False, **lv):
        d = dict(ela_level='UNAVAILABLE', freq_level='UNAVAILABLE', noise_level='UNAVAILABLE')
        d.update(lv)
        return ev.calculate_evidence_scores(conf, pred, d['ela_level'], d['freq_level'], d['noise_level'],
                                            sw, exif, fmt, **{k: v for k, v in lv.items() if k.endswith('_level') and k not in d or k in ('copy_move_level', 'face_boundary_level', 'resampling_level')})

    def test_missing_jpeg_exif_and_editing_tag_add_nothing(self):
        r = ev.calculate_evidence_scores(0.0, 'UNCERTAIN', 'LOW', 'LOW', 'LOW', True, False, 'JPEG')
        self.assertEqual((r['overall_risk'], r['overall_risk_score'], r['evidence_meta']), ('LOW', 0.0, 0.0))

    def test_risk_follows_readable_rules_and_score_is_a_count(self):
        r = ev.calculate_evidence_scores(80, 'LIKELY_MANIPULATED', 'MEDIUM', 'LOW', 'LOW', False, True, 'JPEG')
        self.assertEqual(r['overall_risk'], 'HIGH')
        self.assertEqual(r['overall_risk_score'], 33.33)           # 1 of 3 applicable indicators flagged
        self.assertIn('not a probability', r['risk_interpretation'])
        self.assertEqual(ev.calculate_evidence_scores(80, 'LIKELY_MANIPULATED', 'LOW', 'LOW', 'LOW', False, True, 'JPEG')['overall_risk'], 'MEDIUM')
        self.assertEqual(ev.calculate_evidence_scores(70, 'LIKELY_AUTHENTIC', 'LOW', 'LOW', 'LOW', False, True, 'JPEG')['overall_risk'], 'LOW')
        self.assertEqual(ev.calculate_evidence_scores(0, 'UNCERTAIN', 'HIGH', 'LOW', 'LOW', False, True, 'JPEG')['overall_risk'], 'LOW')


# -------------------------------------------------------------- integration
class ForensicIntegrationTests(TempMediaTestCase):
    def test_upload_stores_structured_report_and_serves_images_privately(self):
        self.client.login(username='alice', password='Correct-Horse-9')
        from django.core.files.uploadedfile import SimpleUploadedFile
        up = self.client.post(reverse('upload'), {'image': SimpleUploadedFile(
            'p.jpg', make_image_bytes('JPEG', (320, 320)), content_type='image/jpeg'), 'consent': 'on'})
        self.assertEqual(up.status_code, 302)
        rec = UploadedImage.objects.get(user=self.alice)
        f = ForensicAnalysis.objects.get(analysis__uploaded_image=rec)
        rep = f.evidence_report
        self.assertEqual(rep['version'], 'evidence-2')
        self.assertEqual(rep['summary']['assessment'], 'NO_FACE')           # noise image has no face
        self.assertEqual(rep['integrity']['sha256'], rec.sha256_hash)
        self.assertIn(rep['provenance']['c2pa']['status'], ('no_credential', 'unavailable'))
        self.assertIn('ela', rep['forensic_indicators'])
        self.assertEqual(f.overall_risk, 'LOW')
        # visualisations are lossless PNGs, served only to the owner
        self.assertTrue(f.ela_image.name.endswith('.png') or not f.ela_image)
        if f.freq_image:
            self.assertEqual(self.client.get(f.freq_image.url).status_code, 200)
            self.client.logout(); self.client.login(username='bob', password='Correct-Horse-9')
            self.assertEqual(self.client.get(f.freq_image.url).status_code, 404)


class VisualisationCapTests(TestCase):
    def test_saved_map_is_capped_but_small_maps_are_untouched(self):
        import tempfile
        from PIL import Image
        from forensics.common import save_png
        folder = tempfile.mkdtemp()
        big = np.random.default_rng(0).integers(0, 255, (3000, 4000, 3), dtype=np.uint8)
        with Image.open(save_png(big, folder, 'big')) as im:
            self.assertEqual(max(im.size), 2048)
            self.assertAlmostEqual(im.size[0] / im.size[1], 4000 / 3000, places=2)
        small = big[:300, :400]
        with Image.open(save_png(small, folder, 'small')) as im:
            self.assertEqual(im.size, (400, 300))
        with self.settings(FORENSIC_VIS_MAX_SIDE=0), Image.open(save_png(big[:2500, :2600], folder, 'nocap')) as im:
            self.assertEqual(im.size, (2600, 2500))
