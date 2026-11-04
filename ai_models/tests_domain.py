from django.test import SimpleTestCase

from ai_models import ensemble
from ai_models.types import AVAILABLE, ModelOutput


def mo(key, score, scope='both', thr=0.5):
    return ModelOutput(key=key, name=key, status=AVAILABLE, score=score, threshold=thr, scope=scope)


class DomainEnsembleTests(SimpleTestCase):
    def cfg(self):
        return ensemble.EnsembleConfig()

    def test_ai_specialist_overrides_faceswap_majority(self):
        outs = [mo('a', .05, 'faceswap'), mo('b', .1, 'faceswap'), mo('c', .1, 'faceswap'),
                mo('d', .2, 'faceswap'), mo('v', .3, 'both'), mo('s', .99, 'ai_generated')]
        r = ensemble.combine(outs, self.cfg())
        self.assertEqual(r['label'], ensemble.POTENTIALLY_MANIPULATED)
        self.assertIn('specialist_flag', r['reasons'])
        self.assertFalse(r['disagreement'])
        self.assertEqual(r['not_applicable_votes'], 4)
        self.assertEqual(r['authentic_votes'], 1)
        self.assertEqual(r['domains']['ai_generated']['state'], 'FLAGGED')

    def test_faceswap_specialists_flag_while_ai_model_clear(self):
        outs = [mo('a', .95, 'faceswap'), mo('b', .9, 'faceswap'), mo('c', .92, 'faceswap'),
                mo('s', .01, 'ai_generated')]
        r = ensemble.combine(outs, self.cfg())
        self.assertEqual(r['label'], ensemble.POTENTIALLY_MANIPULATED)
        self.assertEqual(r['not_applicable_votes'], 1)

    def test_all_clear_stays_authentic(self):
        outs = [mo('a', .05, 'faceswap'), mo('b', .1, 'faceswap'), mo('c', .1, 'both')]
        r = ensemble.combine(outs, self.cfg())
        self.assertEqual(r['label'], ensemble.AUTHENTIC)
        self.assertEqual(r['not_applicable_votes'], 0)

    def test_default_scope_behaves_as_before(self):
        outs = [mo('a', .9), mo('b', .1), mo('c', .1)]
        r = ensemble.combine(outs, self.cfg())
        self.assertNotIn('specialist_flag', r['reasons'])
        self.assertEqual(r['specialist_flags'], [])


class FrequencyFeatureTests(SimpleTestCase):
    def test_combined_extractor_is_concatenation(self):
        import numpy as np
        from ai_models.frequency_features import extract
        face = (np.random.default_rng(0).random((120, 120, 3)) * 255).astype('uint8')
        a = extract(face, 'azimuthal_fft', 256); d = extract(face, 'dct_block_stats', 256)
        c = extract(face, 'fft_dct_combined', 256)
        self.assertEqual(c.shape, (1, 128))
        np.testing.assert_allclose(c, np.hstack([a, d]))
