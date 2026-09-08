"""Convert token sequences into padded embedding tensors."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .sequences import SequenceExample


def vectorize(
    tokens: Sequence[str], embeddings, max_len: int, vector_dim: int
) -> np.ndarray:
    vectors = np.zeros((max_len, vector_dim), dtype=np.float32)
    for index, token in enumerate(tokens[:max_len]):
        try:
            vectors[index] = embeddings[token]
        except KeyError:
            # Word2Vec has no OOV representation; FastText normally resolves it.
            continue
    return vectors


def vectorize_examples(
    examples: Sequence[SequenceExample], embeddings, max_len: int, vector_dim: int
) -> np.ndarray:
    if not examples:
        raise ValueError("cannot vectorize an empty example set")
    return np.stack([
        vectorize(item.tokens, embeddings, max_len, vector_dim)
        for item in examples
    ])
