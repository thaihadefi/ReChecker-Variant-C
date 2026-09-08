import unittest

import numpy as np

from experiments.metrics import evaluate_predictions, select_recall_threshold


class EvaluatorTests(unittest.TestCase):
    def test_threshold_meets_recall_and_is_highest_candidate(self):
        labels = np.array([1, 1, 1, 1, 1, 0, 0])
        probabilities = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.65, 0.1])
        threshold = select_recall_threshold(labels, probabilities, min_recall=0.8)
        self.assertEqual(threshold, 0.6)
        self.assertGreaterEqual(np.mean(probabilities[:5] >= threshold), 0.8)

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
