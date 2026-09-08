"""Exact normalized-token duplicate grouping."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence


def exact_token_group_id(tokens: Sequence[str]) -> str:
    normalized = "\x1f".join(tokens).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()[:16]
