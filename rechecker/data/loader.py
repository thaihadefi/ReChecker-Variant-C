"""Orchestrate parsing, W/C extraction, normalization, and token grouping."""

from __future__ import annotations

from pathlib import Path

from .extraction import split_first_function_regions
from .grouping import exact_token_group_id
from .normalization import clean_fragment
from .parser import parse_gadgets
from .tokenizer import tokenize_lines
from .types import GadgetRecord


def load_records(path: str | Path) -> list[GadgetRecord]:
    records: list[GadgetRecord] = []
    for ordinal, parsed in enumerate(parse_gadgets(path)):
        w_raw, c_raw, diagnostics = split_first_function_regions(parsed.lines)

        # One pass preserves identifier mappings shared between W and C.
        cleaned = clean_fragment(w_raw + c_raw)
        if len(cleaned) != len(w_raw) + len(c_raw):
            raise ValueError(
                f"normalization changed line count for {parsed.sample_id or ordinal}; "
                "cannot preserve the W/C boundary"
            )
        w_tokens = tuple(tokenize_lines(cleaned[:len(w_raw)]))
        c_tokens = tuple(tokenize_lines(cleaned[len(w_raw):]))
        if not w_tokens:
            raise ValueError(f"empty W token region for {parsed.sample_id or ordinal}")
        records.append(
            GadgetRecord(
                sample_id=parsed.sample_id or f"sample-{ordinal:05d}",
                label=parsed.label,
                w_tokens=w_tokens,
                c_tokens=c_tokens,
                group_id=exact_token_group_id(w_tokens + c_tokens),
                diagnostics=diagnostics,
            )
        )
    if not records:
        raise ValueError(f"no gadgets found in {path}")
    return records
