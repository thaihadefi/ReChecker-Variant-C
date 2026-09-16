"""Convert DAppSCAN audit-labeled contracts into the ReChecker gadget format.

Walks a local clone of https://github.com/InPlusLab/DAppSCAN (DAppSCAN-source),
classifies each analyzed contract file as reentrancy-positive (has an
SWC-107 finding) or negative (audited, no SWC-107 finding), and writes them
out in the same "<index> <name>\\n<code>\\n<label>\\n---------------------------------"
format that data/parser.py already reads.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SEPARATOR = "-" * 33
SWC_REENTRANCY = "SWC-107"


def _strip_comments(source: str) -> str:
    """Blank out // and /* */ comment text while preserving line structure.

    data/normalization.py (the pinned, released pipeline) assumes its input
    already had comments stripped (it drops any whole line ending in "*/",
    Messi-Q's own preprocessing convention). Raw DAppSCAN source mixes real
    comments with code on the same line, so it must be pre-cleaned here
    without changing the line count downstream extraction depends on.
    """
    out: list[str] = []
    i, n = 0, len(source)
    in_line_comment = in_block_comment = False
    in_string: str | None = None
    while i < n:
        char = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if in_line_comment:
            if char == "\n":
                in_line_comment = False
                out.append(char)
            else:
                out.append(" ")
            i += 1
        elif in_block_comment:
            if char == "*" and nxt == "/":
                in_block_comment = False
                out.append("  ")
                i += 2
            else:
                out.append(char if char == "\n" else " ")
                i += 1
        elif in_string:
            out.append(char)
            if char == "\\" and i + 1 < n:
                out.append(nxt)
                i += 2
                continue
            if char == in_string:
                in_string = None
            i += 1
        elif char in ("'", '"'):
            in_string = char
            out.append(char)
            i += 1
        elif char == "/" and nxt == "/":
            in_line_comment = True
            out.append("  ")
            i += 2
        elif char == "/" and nxt == "*":
            in_block_comment = True
            out.append("  ")
            i += 2
        else:
            out.append(char)
            i += 1
    return "".join(out)


def _classify(dappscan_root: Path) -> tuple[list[Path], list[Path]]:
    positive, negative = [], []
    for report_path in (dappscan_root / "DAppSCAN-source" / "SWCsource").rglob("*.json"):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, json.JSONDecodeError):
            continue
        file_path = report.get("filePath")
        if not file_path:
            continue
        contract_path = dappscan_root / file_path
        if not contract_path.is_file():
            continue
        is_reentrant = any(
            SWC_REENTRANCY in swc.get("category", "") for swc in report.get("SWCs", [])
        )
        (positive if is_reentrant else negative).append(contract_path)
    return positive, negative


def _is_safe_fragment(lines: list[str]) -> bool:
    return not any(set(line.strip()) == {"-"} and len(line.strip()) >= 3 for line in lines)


def build_dataset(
    dappscan_root: Path, output_path: Path, neg_ratio: float, seed: int
) -> dict[str, int]:
    positive, negative = _classify(dappscan_root)
    sample_count = min(len(negative), round(neg_ratio * len(positive)))
    negative = random.Random(seed).sample(negative, sample_count)

    written, skipped = 0, 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        index = 1
        for label, paths in ((1, positive), (0, negative)):
            for path in paths:
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    skipped += 1
                    continue
                lines = _strip_comments(text).splitlines()
                if not lines or not _is_safe_fragment(lines):
                    skipped += 1
                    continue
                out.write(f"{index} {path.name}\n")
                out.write("\n".join(lines) + "\n")
                out.write(f"{label}\n")
                out.write(SEPARATOR + "\n")
                index += 1
                written += 1
    return {
        "positive_available": len(positive),
        "negative_available": len(negative) + skipped,
        "written": written,
        "skipped": skipped,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dappscan-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("Dataset/dappscan_reentrancy.txt"))
    parser.add_argument("--neg-ratio", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = _parser().parse_args()
    stats = build_dataset(args.dappscan_root, args.output, args.neg_ratio, args.seed)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
