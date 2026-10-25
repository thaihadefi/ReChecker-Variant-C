"""Training-set and feature pruning used to probe overfitting and generalization."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from data.grouping import exact_token_group_id
from data.types import GadgetRecord


UNK_TOKEN = "[UNK]"
_INDEXED_SYMBOL = re.compile(r"^(VAR|FUN)\d+$")
_NUMERIC_LITERAL = re.compile(r"^(0x[0-9a-fA-F]+|\d[\d_]*(\.\d+)?([eE]\d+)?)$")


def abstract_token(token: str) -> str:
    """Drop dataset-specific symbol indices and literal values.

    The normalizer numbers identifiers by first appearance (VAR1, VAR2, ...),
    so the same index means unrelated things in different contracts. Keeping
    only the symbol kind removes that spurious, source-specific signal.
    """
    match = _INDEXED_SYMBOL.match(token)
    if match:
        return match.group(1)
    if _NUMERIC_LITERAL.match(token):
        return "NUM"
    return token


def abstract_identifiers(records: Sequence[GadgetRecord]) -> list[GadgetRecord]:
    """Abstract every record and recompute groups, since abstraction can make
    previously distinct gadgets identical (they must then share a group)."""
    abstracted = []
    for record in records:
        w_tokens = tuple(abstract_token(token) for token in record.w_tokens)
        c_tokens = tuple(abstract_token(token) for token in record.c_tokens)
        abstracted.append(replace(
            record,
            w_tokens=w_tokens,
            c_tokens=c_tokens,
            group_id=exact_token_group_id(w_tokens + c_tokens),
        ))
    return abstracted


def _validate_label_pure_groups(records: Sequence[GadgetRecord]) -> None:
    labels: dict[str, set[int]] = {}
    for record in records:
        labels.setdefault(record.group_id, set()).add(record.label)
    mixed = [group for group, values in labels.items() if len(values) > 1]
    if mixed:
        raise ValueError(f"{len(mixed)} duplicate groups carry conflicting labels")


def deduplicate_groups(records: Sequence[GadgetRecord]) -> list[GadgetRecord]:
    """Keep one record per exact-duplicate group (first occurrence).

    Groups are label-pure in the source data; a mixed group would make the
    kept label arbitrary, so it is rejected instead of silently resolved.
    """
    _validate_label_pure_groups(records)
    seen: set[str] = set()
    kept = []
    for record in records:
        if record.group_id not in seen:
            seen.add(record.group_id)
            kept.append(record)
    return kept


def sample_group_fraction(
    records: Sequence[GadgetRecord], fraction: float, seed: int
) -> list[GadgetRecord]:
    """Keep a label-stratified random fraction of groups (whole groups only)."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    _validate_label_pure_groups(records)
    if fraction == 1.0:
        return list(records)
    group_label = {record.group_id: record.label for record in records}
    rng = np.random.default_rng(seed)
    kept_groups: set[str] = set()
    for label in (0, 1):
        groups = sorted(group for group, value in group_label.items() if value == label)
        if not groups:
            continue
        count = max(1, round(fraction * len(groups)))
        kept_groups.update(rng.choice(groups, size=count, replace=False).tolist())
    return [record for record in records if record.group_id in kept_groups]


def fit_group_vocabulary(
    records: Sequence[GadgetRecord], min_groups: int
) -> frozenset[str]:
    """Tokens that occur in at least `min_groups` distinct duplicate groups.

    Counting groups rather than occurrences stops one heavily duplicated
    contract from keeping its private identifiers alive.
    """
    if min_groups < 1:
        raise ValueError("min_groups must be at least 1")
    seen_groups: set[str] = set()
    document_frequency: Counter[str] = Counter()
    for record in records:
        if record.group_id in seen_groups:
            continue
        seen_groups.add(record.group_id)
        document_frequency.update(set(record.baseline_tokens))
    return frozenset(
        token for token, count in document_frequency.items() if count >= min_groups
    )


def prune_vocabulary(
    records: Sequence[GadgetRecord], vocabulary: frozenset[str]
) -> list[GadgetRecord]:
    """Replace out-of-vocabulary tokens with a shared [UNK] token.

    group_id is left unchanged on purpose: the vocabulary is fitted on the
    training partition only, and split membership must not depend on it.
    """
    def keep(tokens: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(token if token in vocabulary else UNK_TOKEN for token in tokens)

    return [
        replace(record, w_tokens=keep(record.w_tokens), c_tokens=keep(record.c_tokens))
        for record in records
    ]
