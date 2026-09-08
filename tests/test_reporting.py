import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rechecker.config import ExperimentConfig
from rechecker.experiments.artifacts import write_json
from rechecker.experiments.reporting import summarize_run
from rechecker.representation.variants import VARIANTS


class ReportingTests(unittest.TestCase):
    def _run_dir(self, root: str) -> Path:
        run_dir = Path(root) / "run"
        write_json(run_dir / "config.json", ExperimentConfig(folds=2).to_dict())
        return run_dir

    def _result(self, run_dir: Path, variant: str, fold: int, accuracy: float):
        write_json(
            run_dir / "results" / f"fold_{fold + 1:02d}_{variant}.json",
            {
                "variant": variant,
                "fold": fold,
                "metrics": {"accuracy": accuracy, "threshold": 0.5},
                "outer_test_size": 1,
                "predictions": [{"sample_id": f"sample-{fold}"}],
            },
        )

    def test_incomplete_run_is_rejected_by_default(self):
        with TemporaryDirectory() as directory:
            run_dir = self._run_dir(directory)
            self._result(run_dir, "b0", 0, 0.8)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                summarize_run(run_dir, (VARIANTS["b0"],))

    def test_partial_summary_is_explicit(self):
        with TemporaryDirectory() as directory:
            run_dir = self._run_dir(directory)
            self._result(run_dir, "b0", 0, 0.8)
            summary = summarize_run(
                run_dir, (VARIANTS["b0"],), allow_partial=True
            )
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["missing_folds"], {"b0": [1]})

    def test_complete_paired_summary(self):
        with TemporaryDirectory() as directory:
            run_dir = self._run_dir(directory)
            for fold in range(2):
                self._result(run_dir, "b0", fold, 0.7 + fold * 0.1)
                self._result(run_dir, "b3", fold, 0.8 + fold * 0.1)
            summary = summarize_run(run_dir, (VARIANTS["b0"], VARIANTS["b3"]))
            self.assertTrue(summary["complete"])
            self.assertAlmostEqual(
                summary["paired_b3_minus_b0"]["accuracy"]["mean"], 0.1
            )

    def test_paired_summary_rejects_different_test_samples(self):
        with TemporaryDirectory() as directory:
            run_dir = self._run_dir(directory)
            for fold in range(2):
                self._result(run_dir, "b0", fold, 0.7)
                self._result(run_dir, "b3", fold, 0.8)
            path = run_dir / "results" / "fold_01_b3.json"
            changed = {
                "variant": "b3",
                "fold": 0,
                "metrics": {"accuracy": 0.8, "threshold": 0.5},
                "outer_test_size": 1,
                "predictions": [{"sample_id": "different"}],
            }
            write_json(path, changed)
            with self.assertRaisesRegex(ValueError, "different samples"):
                summarize_run(run_dir, (VARIANTS["b0"], VARIANTS["b3"]))


if __name__ == "__main__":
    unittest.main()
