import unittest

from config.config import ExperimentConfig


class ConfigTests(unittest.TestCase):
    def test_documented_ratio_bounds_are_accepted(self):
        self.assertEqual(ExperimentConfig(w_ratio=0.5).w_budget, 249)
        self.assertEqual(ExperimentConfig(w_ratio=0.8).c_budget, 100)

    def test_ratio_outside_documented_bounds_is_rejected(self):
        for ratio in (0.49, 0.81):
            with self.subTest(ratio=ratio), self.assertRaisesRegex(ValueError, "documented"):
                ExperimentConfig(w_ratio=ratio)

    def test_config_round_trip(self):
        original = ExperimentConfig(max_len=100, batch_size=8)
        self.assertEqual(ExperimentConfig.from_dict(original.to_dict()), original)


if __name__ == "__main__":
    unittest.main()
