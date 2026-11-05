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

    def test_invalid_token_dropout_is_rejected(self):
        for value in (-0.1, 1.0):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "token_dropout"):
                ExperimentConfig(token_dropout=value)

    def test_invalid_threshold_strategy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "threshold_strategy"):
            ExperimentConfig(threshold_strategy="max_f1")

    def test_invalid_early_stop_monitor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "early_stop_monitor"):
            ExperimentConfig(early_stop_monitor="val_f1")

    def test_invalid_extraction_strategy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "extraction_strategy"):
            ExperimentConfig(extraction_strategy="random")

    def test_non_default_extraction_and_dedup_round_trip(self):
        original = ExperimentConfig(
            extraction_strategy="call_flow",
            dedup=True,
            abstract_ids=True,
            min_token_groups=5,
        )
        serialized = original.to_dict()
        self.assertEqual(serialized["extraction_strategy"], "call_flow")
        self.assertTrue(serialized["dedup"])
        self.assertEqual(ExperimentConfig.from_dict(serialized), original)
        cli = original.to_cli_args()
        self.assertIn("--extraction-strategy", cli)
        self.assertIn("call_flow", cli)
        self.assertIn("--dedup", cli)


if __name__ == "__main__":
    unittest.main()
