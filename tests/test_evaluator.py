import unittest

import numpy as np

from experiments.metrics import (
    evaluate_predictions,
    recompute_metrics,
    select_eer_threshold,
    select_recall_threshold,
)


class EvaluatorTests(unittest.TestCase):
    def test_threshold_meets_recall_and_is_highest_candidate(self):
        labels = np.array([1, 1, 1, 1, 1, 0, 0])
        probabilities = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.65, 0.1])
        threshold = select_recall_threshold(labels, probabilities, min_recall=0.8)
        self.assertEqual(threshold, 0.6)
        self.assertGreaterEqual(np.mean(probabilities[:5] >= threshold), 0.8)

    def test_eer_threshold_balances_false_positive_and_negative_rates(self):
        labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        probabilities = np.array([0.9, 0.8, 0.6, 0.3, 0.7, 0.4, 0.2, 0.1])
        threshold = select_eer_threshold(labels, probabilities)
        self.assertEqual(threshold, 0.6)
        metrics = evaluate_predictions(labels, probabilities, threshold)
        self.assertEqual(metrics["false_positive_rate"], metrics["false_negative_rate"])

    def test_eer_threshold_falls_back_to_half_for_single_class(self):
        threshold = select_eer_threshold(np.array([1, 1, 1]), np.array([0.9, 0.5, 0.2]))
        self.assertEqual(threshold, 0.5)

    def test_recompute_metrics_rescoring_matches_direct_call(self):
        result = {
            "validation_predictions": [
                {"label": 1, "probability": 0.9},
                {"label": 1, "probability": 0.8},
                {"label": 1, "probability": 0.6},
                {"label": 1, "probability": 0.3},
                {"label": 0, "probability": 0.7},
                {"label": 0, "probability": 0.4},
                {"label": 0, "probability": 0.2},
                {"label": 0, "probability": 0.1},
            ],
            "predictions": [
                {"label": 1, "probability": 0.85},
                {"label": 0, "probability": 0.15},
            ],
        }
        recomputed = recompute_metrics(result, "eer")
        expected_threshold = select_eer_threshold(
            [item["label"] for item in result["validation_predictions"]],
            [item["probability"] for item in result["validation_predictions"]],
        )
        expected = evaluate_predictions(
            [item["label"] for item in result["predictions"]],
            [item["probability"] for item in result["predictions"]],
            expected_threshold,
        )
        self.assertEqual(recomputed, expected)

    def test_confusion_matrix_is_stable_when_a_class_is_not_predicted(self):
        metrics = evaluate_predictions(
            np.array([0, 0, 1]), np.array([0.1, 0.2, 0.3]), threshold=0.9
        )
        self.assertEqual(metrics["confusion_matrix"], {"tn": 2, "fp": 0, "fn": 1, "tp": 0})
        self.assertEqual(metrics["recall_vulnerable"], 0.0)

    def test_non_finite_probabilities_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate_predictions(np.array([0, 1]), np.array([0.1, np.nan]), 0.5)

    def test_mismatched_shapes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "equal-length"):
            select_recall_threshold(np.array([0, 1]), np.array([0.1]), 0.95)

    def test_out_of_range_probabilities_are_rejected(self):
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            evaluate_predictions(np.array([0, 1]), np.array([0.1, 1.1]), 0.5)


if __name__ == "__main__":
    unittest.main()
