"""Fold-local Word2Vec and FastText training."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from .variants import EmbeddingKind


def fit_embedding(
    corpus: Iterable[Sequence[str]],
    vector_dim: int,
    seed: int,
    kind: EmbeddingKind,
):
    from gensim.models import FastText, Word2Vec

    sentences = list(corpus)
    common_kwargs = dict(min_count=1, vector_size=vector_dim, sg=0, seed=seed, workers=1)
    if kind == "fasttext":
        # gensim defaults to a 2,000,000-bucket ngram hash table, allocating a
        # bucket*vector_size matrix (~2.4GB at vector_dim=300) regardless of
        # this fold's much smaller vocabulary. Cap it to a size that still
        # comfortably covers a few-thousand-token corpus without ballooning
        # per-fold disk usage.
        return FastText(sentences, bucket=100_000, **common_kwargs)
    return Word2Vec(sentences, **common_kwargs)


def save_embedding(model, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(destination))


def load_embedding(path: str | Path, kind: EmbeddingKind):
    from gensim.models import FastText, Word2Vec

    model_type = FastText if kind == "fasttext" else Word2Vec
    return model_type.load(str(path))
