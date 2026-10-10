"""Convert SmartBugs-curated contracts into the ReChecker gadget format.

Used as a small, manually annotated training expansion. A contract is
positive when any annotated vulnerability is in the "reentrancy" category,
negative when it is annotated only with other categories. SmartBugs tags a
file with its main weakness only, so "other category" does not prove the
absence of reentrancy: negatives exclude the unchecked-low-level-call
folder (call-before-update is common there, e.g. keepMyEther) and any file
that is also listed as reentrant elsewhere. Comments are
stripped before writing, which also removes the curators' inline
"// <yes> <report> REENTRANCY" markers so they cannot leak the label.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from data.build_dappscan_dataset import write_gadgets

REENTRANCY = "reentrancy"
UNSAFE_NEGATIVE_CATEGORIES = frozenset({"unchecked_low_level_calls"})


def classify(smartbugs_root: Path) -> tuple[list[Path], list[Path]]:
    entries = json.loads((smartbugs_root / "vulnerabilities.json").read_text(encoding="utf-8"))
    positive, negative = [], []
    for entry in entries:
        path = smartbugs_root / entry["path"]
        if not path.is_file():
            continue
        categories = {item["category"] for item in entry.get("vulnerabilities", [])}
        if REENTRANCY in categories:
            positive.append(path)
        elif not categories & UNSAFE_NEGATIVE_CATEGORIES:
            negative.append(path)
    positive_names = {path.name for path in positive}
    negative = [path for path in negative if path.name not in positive_names]
    return sorted(positive), sorted(negative)


def build_dataset(
    smartbugs_root: Path, output_path: Path, neg_ratio: float, seed: int
) -> dict[str, int]:
    positive, negative = classify(smartbugs_root)
    sample_count = min(len(negative), round(neg_ratio * len(positive)))
    negative = random.Random(seed).sample(negative, sample_count)
    written, skipped = write_gadgets([(1, positive), (0, negative)], output_path)
    return {
        "positive": len(positive),
        "negative": len(negative),
        "written": written,
        "skipped": skipped,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smartbugs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("Dataset/smartbugs_reentrancy.txt"))
    parser.add_argument("--neg-ratio", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = _parser().parse_args()
    stats = build_dataset(args.smartbugs_root, args.output, args.neg_ratio, args.seed)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
