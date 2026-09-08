"""Parser for the original ReChecker separator/label dataset format."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from .types import ParsedGadget


SEPARATOR = "-" * 33


def parse_gadgets(path: str | Path) -> Iterator[ParsedGadget]:
    fragment: list[str] = []
    label: int | None = None
    sample_id = ""

    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if SEPARATOR in line:
                if fragment:
                    if label is None:
                        raise ValueError(
                            f"missing binary label before separator for {sample_id or line_number}"
                        )
                    yield ParsedGadget(sample_id, label, tuple(fragment))
                    fragment = []
                    label = None
                    sample_id = ""
                continue

            fields = stripped.split()
            if fields[0].isdigit():
                if not fragment:
                    sample_id = " ".join(fields)
                elif stripped in {"0", "1"}:
                    # Preserve the released parser: standalone numeric Solidity lines are
                    # indistinguishable from labels, and the final label wins.
                    label = int(stripped)
                else:
                    fragment.append(stripped)
            else:
                fragment.append(stripped)

    if fragment:
        if label is None:
            raise ValueError(f"missing binary label at end of file for {sample_id}")
        yield ParsedGadget(sample_id, label, tuple(fragment))
