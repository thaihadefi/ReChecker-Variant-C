"""Input representation policies and embedding adapters."""

from .sequences import SequenceExample, build_sequence, coverage
from .variants import VARIANTS, VariantSpec, resolve_variants

__all__ = [
    "SequenceExample",
    "VARIANTS",
    "VariantSpec",
    "build_sequence",
    "coverage",
    "resolve_variants",
]
