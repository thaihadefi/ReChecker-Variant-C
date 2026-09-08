"""Group-disjoint outer and validation splits."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from rechecker.config import ExperimentConfig
from rechecker.data.types import GadgetRecord


def _split(records: Sequence[GadgetRecord], folds: int, seed: int):
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    groups = np.asarray([record.group_id for record in records])
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )
    return list(splitter.split(np.zeros(len(records)), labels, groups))


def make_outer_splits(
    records: Sequence[GadgetRecord], config: ExperimentConfig
):
    return _split(records, config.folds, config.seed)


def make_validation_split(
    records: Sequence[GadgetRecord], config: ExperimentConfig
):
    return _split(records, config.validation_folds, config.seed + 1)[0]


def assert_group_disjoint(
    left: Sequence[GadgetRecord], right: Sequence[GadgetRecord], context: str
) -> None:
    overlap = {record.group_id for record in left} & {
        record.group_id for record in right
    }
    if overlap:
        raise RuntimeError(f"group leakage detected in {context}: {len(overlap)} groups")
