"""Declarative B0-B4 ablation definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


EmbeddingKind = Literal["word2vec", "fasttext"]
SequenceKind = Literal["flat-prefix", "region-prefix", "region-head-tail"]


@dataclass(frozen=True)
class VariantSpec:
    name: str
    description: str
    embedding: EmbeddingKind
    sequence: SequenceKind
    use_separator: bool
    use_segment_embedding: bool


VARIANTS: dict[str, VariantSpec] = {
    "b0": VariantSpec(
        "b0", "Word2Vec baseline", "word2vec", "flat-prefix", False, False
    ),
    "b1": VariantSpec(
        "b1", "FastText-only ablation", "fasttext", "flat-prefix", False, False
    ),
    "b2": VariantSpec(
        "b2", "FastText plus [SEP] without segment embedding",
        "fasttext", "region-prefix", True, False,
    ),
    "b3": VariantSpec(
        "b3", "full Variant C", "fasttext", "region-prefix", True, True
    ),
    "b4": VariantSpec(
        "b4", "Variant C with head-tail region policy",
        "fasttext", "region-head-tail", True, True,
    ),
}

ALIASES = {"baseline": "b0", "variant_c": "b3", "variant-c": "b3"}


def resolve_variants(selection: str) -> tuple[VariantSpec, ...]:
    if selection == "all":
        return tuple(VARIANTS.values())
    canonical = ALIASES.get(selection, selection)
    try:
        return (VARIANTS[canonical],)
    except KeyError as error:
        choices = ", ".join((*VARIANTS, *ALIASES, "all"))
        raise ValueError(f"unknown variant {selection!r}; choose one of {choices}") from error
