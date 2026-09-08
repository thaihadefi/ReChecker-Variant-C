import unittest

import numpy as np

from config.config import ExperimentConfig
from representation.variants import VARIANTS

try:
    import tensorflow as tf

    from models.models import build_model
except ImportError:
    tf = None


@unittest.skipIf(tf is None, "TensorFlow is not installed")
class ModelSmokeTests(unittest.TestCase):
    def setUp(self):
        self.config = ExperimentConfig(
            max_len=6, vector_dim=8, hidden_units=4, dense_units=4
        )

    def test_baseline_forward_pass(self):
        model = build_model(self.config, VARIANTS["b0"])
        vectors = np.zeros((2, 6, 8), dtype=np.float32)
        mask = np.zeros((2, 6), dtype=bool)
        vectors[:, :2] = 1
        mask[:, :2] = True
        output = model.predict([vectors, mask], verbose=0)
        self.assertEqual(output.shape, (2, 2))
        np.testing.assert_allclose(output.sum(axis=1), 1.0, atol=1e-6)

    def test_variant_forward_pass(self):
        model = build_model(self.config, VARIANTS["b3"])
        vectors = np.zeros((2, 6, 8), dtype=np.float32)
        segments = np.full((2, 6), 3, dtype=np.int32)
        segments[:, :3] = [0, 2, 1]
        output = model.predict([vectors, segments], verbose=0)
        self.assertEqual(output.shape, (2, 2))
        np.testing.assert_allclose(output.sum(axis=1), 1.0, atol=1e-6)

    def test_baseline_padding_values_do_not_change_prediction(self):
        model = build_model(self.config, VARIANTS["b0"])
        vectors = np.zeros((2, 6, 8), dtype=np.float32)
        vectors[:, :2] = 1
        vectors[1, 2:] = 100
        mask = np.zeros((2, 6), dtype=bool)
        mask[:, :2] = True
        output = model.predict([vectors, mask], verbose=0)
        np.testing.assert_allclose(output[0], output[1], atol=1e-6)

    def test_variant_padding_values_do_not_change_prediction(self):
        model = build_model(self.config, VARIANTS["b3"])
        vectors = np.zeros((2, 6, 8), dtype=np.float32)
        vectors[:, :3] = 1
        vectors[1, 3:] = 100
        segments = np.full((2, 6), 3, dtype=np.int32)
        segments[:, :3] = [0, 2, 1]
        output = model.predict([vectors, segments], verbose=0)
        np.testing.assert_allclose(output[0], output[1], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
