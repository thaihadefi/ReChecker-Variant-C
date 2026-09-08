"""W/C region extraction and diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

from .types import ExtractionDiagnostics


FUNCTION_MARKERS = {"function", "constructor"}


def split_first_function_regions(
    lines: Sequence[str],
) -> tuple[list[str], list[str], ExtractionDiagnostics]:
    """Preserve the historical first-function heuristic and expose its limitations.

    The source dataset does not carry explicit candidate metadata. A region begins at
    the first function/constructor declaration and ends at the next such declaration.
    Intervening modifier or contract declarations therefore remain visible in diagnostics.
    """
    starts = [
        index
        for index, line in enumerate(lines)
        if line.split() and line.split()[0] in FUNCTION_MARKERS
    ]
    if not starts:
        return list(lines), [], ExtractionDiagnostics()

    boundaries = starts + [len(lines)]
    blocks = [
        (start, end)
        for start, end in zip(boundaries, boundaries[1:])
        if start < end
    ]
    w_start, w_end = blocks[0]
    w_lines = list(lines[w_start:w_end])
    c_lines = [line for start, end in blocks[1:] for line in lines[start:end]]
    first_tokens = [line.split()[0] for line in w_lines if line.split()]
    diagnostics = ExtractionDiagnostics(
        dropped_preamble_lines=w_start,
        w_contains_modifier_declaration="modifier" in first_tokens,
        w_contains_contract_declaration="contract" in first_tokens,
    )
    return w_lines, c_lines, diagnostics
