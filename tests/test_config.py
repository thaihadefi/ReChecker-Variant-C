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

    def test_invalid_threshold_strategy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "threshold_strategy"):
            ExperimentConfig(threshold_strategy="max_f1")

    def test_invalid_early_stop_monitor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "early_stop_monitor"):
            ExperimentConfig(early_stop_monitor="val_f1")


if __name__ == "__main__":
    unittest.main()
