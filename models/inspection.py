"""Stable model inspection helpers."""

from __future__ import annotations

import numpy as np


def count_parameters(model) -> dict[str, int]:
    trainable = sum(int(np.prod(variable.shape)) for variable in model.trainable_weights)
    non_trainable = sum(
        int(np.prod(variable.shape)) for variable in model.non_trainable_weights
    )
    return {
        "trainable_parameters": trainable,
        "non_trainable_parameters": non_trainable,
        "total_parameters": trainable + non_trainable,
    }
