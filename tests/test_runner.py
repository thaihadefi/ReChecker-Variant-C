import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import gensim  # noqa: F401
    import tensorflow  # noqa: F401

    RUNTIME_AVAILABLE = True
except ImportError:
    RUNTIME_AVAILABLE = False

from config.config import ExperimentConfig
from data.types import GadgetRecord
from experiments.artifacts import FoldArtifactPaths, load_model_bundle
from representation.variants import VARIANTS


@unittest.skipUnless(RUNTIME_AVAILABLE, "TensorFlow and Gensim are required")
class TinyFoldIntegrationTests(unittest.TestCase):
    def test_b3_fold_writes_a_reloadable_bundle(self):
        # 12 records (not 8): with each record its own singleton group and
        # label alternating by index, StratifiedGroupKFold on only 8 groups
        # can place every label-1 group in one fold and every label-0 group
        # in the other (verified: happens deterministically with this seed),
        # leaving the fit partition with a single class. 12 groups gives the
        # splitter enough room to keep both classes on both sides.
        train = [
            GadgetRecord(f"train-{index}", index % 2, (f"w{index}",), ("c",), f"g{index}")
            for index in range(12)
        ]
        test = [
            GadgetRecord("test-0", 0, ("w0",), ("c",), "test-g0"),
            GadgetRecord("test-1", 1, ("w1",), ("c",), "test-g1"),
        ]
        config = ExperimentConfig(
            max_len=5,
            vector_dim=8,
            hidden_units=4,
            dense_units=4,
            batch_size=2,
            epochs=1,
            folds=2,
            validation_folds=2,
            verbose=0,
        )
        from experiments.runner import train_fold

        with TemporaryDirectory() as directory:
            paths = FoldArtifactPaths(Path(directory), 0, "b3")
            result = train_fold(VARIANTS["b3"], train, test, config, paths)
            result.update({"fold": 0})
            from experiments.artifacts import write_json

            write_json(paths.result, result)
            self.assertTrue(paths.is_complete())
            bundle = load_model_bundle(paths.manifest)
            self.assertEqual(bundle.spec.name, "b3")
            self.assertEqual(bundle.config.max_len, 5)


if __name__ == "__main__":
    unittest.main()
