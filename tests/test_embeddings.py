import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

try:
    import gensim  # noqa: F401

    from rechecker.representation.embeddings import (
        fit_embedding,
        load_embedding,
        save_embedding,
    )
except ImportError:
    gensim = None


@unittest.skipIf(gensim is None, "Gensim is not installed")
class EmbeddingTests(unittest.TestCase):
    def test_fasttext_bundle_preserves_vectors_and_supports_oov(self):
        model = fit_embedding(
            [("transfer", "value"), ("sender", "value")],
            vector_dim=8,
            seed=42,
            kind="fasttext",
        )
        oov = model.wv["transferValue"]
        self.assertEqual(oov.shape, (8,))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "embedding.model"
            save_embedding(model, path)
            loaded = load_embedding(path, "fasttext")
        np.testing.assert_allclose(loaded.wv["transfer"], model.wv["transfer"])


if __name__ == "__main__":
    unittest.main()
