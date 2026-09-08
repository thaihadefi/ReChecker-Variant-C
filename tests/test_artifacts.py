import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from config.config import ExperimentConfig
from experiments.artifacts import (
    FoldArtifactPaths,
    read_json,
    write_bundle_manifest,
    write_json,
)
from representation.variants import VARIANTS


class ArtifactTests(unittest.TestCase):
    def test_bundle_is_complete_only_when_files_match_fold_and_variant(self):
        with TemporaryDirectory() as directory:
            paths = FoldArtifactPaths(Path(directory), 1, "b3")
            paths.weights.parent.mkdir(parents=True)
            paths.embedding.parent.mkdir(parents=True)
            paths.weights.write_bytes(b"weights")
            paths.embedding.write_bytes(b"embedding")
            write_json(paths.result, {"fold": 1, "variant": "b3"})
            write_bundle_manifest(
                paths, ExperimentConfig(), VARIANTS["b3"], threshold=0.4
            )
            self.assertTrue(paths.is_complete())
            manifest = read_json(paths.manifest)
            self.assertEqual(manifest["embedding_kind"], "fasttext")
            self.assertEqual(manifest["threshold"], 0.4)

    def test_corrupt_manifest_is_not_complete(self):
        with TemporaryDirectory() as directory:
            paths = FoldArtifactPaths(Path(directory), 0, "b0")
            for path in (paths.result, paths.weights, paths.embedding, paths.manifest):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"not-json")
            self.assertFalse(paths.is_complete())


if __name__ == "__main__":
    unittest.main()
