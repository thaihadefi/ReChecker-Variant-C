"""Orchestrate parsing, W/C extraction, normalization, and token grouping."""

from __future__ import annotations

from pathlib import Path

from .extraction import split_first_function_regions, split_regions
from .grouping import exact_token_group_id
from .normalization import clean_fragment
from .parser import parse_gadgets
from .tokenizer import tokenize_lines
from .types import ExtractionDiagnostics, GadgetRecord


def load_records(
    path: str | Path,
    extraction_strategy: str = "first",
    w_budget: int = 299,
    c_budget: int = 200,
) -> list[GadgetRecord]:
    records: list[GadgetRecord] = []
    cache: dict[
        tuple[str, ...],
        tuple[tuple[str, ...], tuple[str, ...], str, ExtractionDiagnostics],
    ] = {}

    for ordinal, parsed in enumerate(parse_gadgets(path)):
        key = tuple(parsed.lines)
        cached = cache.get(key)
        if cached is None:
            w_raw, c_raw, diagnostics = split_regions(
                parsed.lines,
                strategy=extraction_strategy,
                w_budget=w_budget,
                c_budget=c_budget,
            )

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

            if extraction_strategy == "first":
                group_id = exact_token_group_id(w_tokens + c_tokens)
            else:
                canon_w, canon_c, _ = split_first_function_regions(parsed.lines)
                canon_tokens = tuple(tokenize_lines(clean_fragment(canon_w + canon_c)))
                group_id = exact_token_group_id(canon_tokens)

            cached = (w_tokens, c_tokens, group_id, diagnostics)
            cache[key] = cached
        else:
            w_tokens, c_tokens, group_id, diagnostics = cached

        records.append(
            GadgetRecord(
                sample_id=parsed.sample_id or f"sample-{ordinal:05d}",
                label=parsed.label,
                w_tokens=w_tokens,
                c_tokens=c_tokens,
                group_id=group_id,
                diagnostics=diagnostics,
            )
        )
    if not records:
        raise ValueError(f"no gadgets found in {path}")
    return records
