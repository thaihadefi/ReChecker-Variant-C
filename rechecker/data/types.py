"""Domain records shared by preprocessing and experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedGadget:
    sample_id: str
    label: int
    lines: tuple[str, ...]


@dataclass(frozen=True)
class ExtractionDiagnostics:
    dropped_preamble_lines: int = 0
    w_contains_modifier_declaration: bool = False
    w_contains_contract_declaration: bool = False


@dataclass(frozen=True)
class GadgetRecord:
    sample_id: str
    label: int
    w_tokens: tuple[str, ...]
    c_tokens: tuple[str, ...]
    group_id: str
    diagnostics: ExtractionDiagnostics = ExtractionDiagnostics()

    @property
    def baseline_tokens(self) -> tuple[str, ...]:
        return self.w_tokens + self.c_tokens
