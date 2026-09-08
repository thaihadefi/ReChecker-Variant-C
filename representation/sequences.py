"""Flat and segment-aware sequence construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from data.types import GadgetRecord

from .variants import VariantSpec


W_SEGMENT = 0
C_SEGMENT = 1
SEP_SEGMENT = 2
PAD_SEGMENT = 3
SEP_TOKEN = "[SEP]"


@dataclass(frozen=True)
class SequenceExample:
    tokens: tuple[str, ...]
    segment_ids: np.ndarray
    retained_w: int
    retained_c: int

    @property
    def valid_mask(self) -> np.ndarray:
        return self.segment_ids != PAD_SEGMENT


def _take(tokens: Sequence[str], budget: int, head_tail: bool) -> tuple[str, ...]:
    if len(tokens) <= budget:
        return tuple(tokens)
    if not head_tail:
        return tuple(tokens[:budget])
    head = budget // 2
    return tuple(tokens[:head]) + tuple(tokens[-(budget - head):])


def _flat_prefix(record: GadgetRecord, max_len: int) -> SequenceExample:
    tokens = record.baseline_tokens[:max_len]
    retained_w = min(len(record.w_tokens), len(tokens))
    retained_c = max(0, len(tokens) - retained_w)
    segments = np.full(max_len, PAD_SEGMENT, dtype=np.int32)
    segments[:retained_w] = W_SEGMENT
    segments[retained_w:len(tokens)] = C_SEGMENT
    return SequenceExample(tokens, segments, retained_w, retained_c)


def _region_sequence(
    record: GadgetRecord,
    max_len: int,
    w_ratio: float,
    head_tail: bool,
) -> SequenceExample:
    w_budget = int((max_len - 1) * w_ratio)
    c_budget = max_len - 1 - w_budget
    if w_budget < 1 or c_budget < 1:
        raise ValueError("max_len and w_ratio must allocate at least one token to W and C")
    w_tokens = _take(record.w_tokens, w_budget, head_tail)
    c_tokens = _take(record.c_tokens, c_budget, head_tail)
    tokens = w_tokens + (SEP_TOKEN,) + c_tokens
    segments = np.full(max_len, PAD_SEGMENT, dtype=np.int32)
    segments[:len(w_tokens)] = W_SEGMENT
    segments[len(w_tokens)] = SEP_SEGMENT
    segments[len(w_tokens) + 1:len(tokens)] = C_SEGMENT
    return SequenceExample(tokens, segments, len(w_tokens), len(c_tokens))


def build_sequence(
    record: GadgetRecord,
    spec: VariantSpec,
    max_len: int,
    w_ratio: float,
) -> SequenceExample:
    if spec.sequence == "flat-prefix":
        return _flat_prefix(record, max_len)
    return _region_sequence(
        record,
        max_len,
        w_ratio,
        head_tail=spec.sequence == "region-head-tail",
    )


def embedding_tokens(record: GadgetRecord, spec: VariantSpec) -> tuple[str, ...]:
    if spec.use_separator:
        return record.w_tokens + (SEP_TOKEN,) + record.c_tokens
    return record.baseline_tokens


def coverage(
    records: Sequence[GadgetRecord],
    examples: Sequence[SequenceExample],
) -> dict[str, float]:
    if not records:
        return {}
    if len(records) != len(examples):
        raise ValueError("records and examples must have equal lengths")

    def ratio(retained: int, total: int) -> float:
        return retained / total if total else 1.0

    return {
        "sep_coverage": float(np.mean([SEP_TOKEN in item.tokens for item in examples])),
        "c_coverage": float(np.mean([item.retained_c > 0 for item in examples])),
        "token_retention": float(np.mean([
            len(item.tokens) / (len(record.w_tokens) + 1 + len(record.c_tokens))
            for record, item in zip(records, examples)
        ])),
        "w_retention": float(np.mean([
            ratio(item.retained_w, len(record.w_tokens))
            for record, item in zip(records, examples)
        ])),
        "c_retention": float(np.mean([
            ratio(item.retained_c, len(record.c_tokens))
            for record, item in zip(records, examples)
        ])),
    }
