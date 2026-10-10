"""Label-noise and input-window audit for the DAppSCAN test set.

1. Reconstructs which DAppSCAN source file and audit finding each sample in
   Dataset/dappscan_reentrancy.txt came from (byte-identical rebuild check).
2. For every positive, checks whether the audited SWC-107 location survives
   W/C extraction and truncation, i.e. whether the model can see it at all.
3. Draws a label-stratified random sample and writes, per sample, a review
   sheet (finding + the functions that make external calls) for manual
   relabeling into audit_sample.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from data.build_dappscan_dataset import _classify, _strip_comments, build_dataset
from data.extraction import FUNCTION_MARKERS
from data.normalization import clean_fragment
from data.tokenizer import tokenize_line

_LINE_RANGE = re.compile(r"L(\d+)(?:-L?(\d+))?")
_ANNOTATION = re.compile(r"^\s*//.*SWC-\d+")
_EXTERNAL_CALL = re.compile(
    r"\.(call|delegatecall|send|transfer|transferFrom|safeTransfer|safeTransferFrom|"
    r"safeMint|_safeMint|functionCall|functionCallWithValue|sendValue)\s*[({]"
    r"|\bI[A-Z]\w*\([^)]*\)\.\w+\("
)


@dataclass(frozen=True)
class Finding:
    function: str
    first_line: int
    last_line: int


def parse_line_range(value: str) -> tuple[int, int] | None:
    match = _LINE_RANGE.search(value or "")
    if not match:
        return None
    first = int(match.group(1))
    return first, int(match.group(2) or first)


def reentrancy_findings(dappscan_root: Path) -> dict[Path, list[Finding]]:
    findings: dict[Path, list[Finding]] = defaultdict(list)
    for report_path in (dappscan_root / "DAppSCAN-source" / "SWCsource").rglob("*.json"):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, json.JSONDecodeError):
            continue
        if not report.get("filePath"):
            continue
        for swc in report.get("SWCs", []):
            if "SWC-107" not in swc.get("category", ""):
                continue
            span = parse_line_range(swc.get("lineNumber", ""))
            if span:
                findings[dappscan_root / report["filePath"]].append(
                    Finding(swc.get("function", ""), *span)
                )
    return findings


def locate_findings(source: str, findings: list[Finding]) -> list[Finding]:
    """Shift report line numbers onto the shipped file.

    DAppSCAN inserted "// SWC-107-Reentrancy: Lx-Ly" annotation lines into the
    sources after the reports were written, so a reported line n is the n-th
    line not counting those annotations (verified: 131/137 named findings then
    start inside the named function, versus 119/137 without the shift).
    """
    file_line_of: dict[int, int] = {}
    original = 0
    for number, line in enumerate(source.splitlines(), start=1):
        if _ANNOTATION.match(line):
            continue
        original += 1
        file_line_of[original] = number
    return [
        Finding(
            finding.function,
            file_line_of.get(finding.first_line, finding.first_line),
            file_line_of.get(finding.last_line, finding.last_line),
        )
        for finding in findings
    ]


def sample_sources(dappscan_root: Path, dataset_file: Path, neg_ratio: float, seed: int):
    """(sample_id, path, label) in dataset order, verified against the dataset file."""
    with tempfile.TemporaryDirectory() as scratch:
        rebuilt = Path(scratch) / "rebuilt.txt"
        build_dataset(dappscan_root, rebuilt, neg_ratio, seed)
        if rebuilt.read_bytes() != dataset_file.read_bytes():
            raise RuntimeError(
                "rebuilt dataset differs from the committed file; wrong DAppSCAN "
                "revision, --neg-ratio or --seed"
            )
    positive, negative = _classify(dappscan_root)
    negative = random.Random(seed).sample(
        negative, min(len(negative), round(neg_ratio * len(positive)))
    )
    ordered = [(1, path) for path in positive] + [(0, path) for path in negative]
    return [
        (f"{index} {path.name}", path, label)
        for index, (label, path) in enumerate(ordered, start=1)
    ]


def token_positions(source: str) -> dict[str, dict[int, tuple[int, int]]]:
    """Map source line number -> (first, last) token offset in each model input.

    Mirrors parser (blank lines dropped), extraction (W = first function block,
    C = everything after it, preamble dropped) and normalization/tokenization.
    Returns offsets for "w", "c" (region-prefix variants) and "flat" (b0).
    """
    kept = [
        (number, line.strip())
        for number, line in enumerate(_strip_comments(source).splitlines(), start=1)
        if line.strip()
    ]
    starts = [
        index for index, (_, line) in enumerate(kept)
        if line.split() and line.split()[0] in FUNCTION_MARKERS
    ]
    positions: dict[str, dict[int, tuple[int, int]]] = {"w": {}, "c": {}, "flat": {}}
    if not starts:
        return positions
    w_end = starts[1] if len(starts) > 1 else len(kept)
    region_lines = kept[starts[0]:]
    cleaned = clean_fragment([line for _, line in region_lines])
    w_size = w_end - starts[0]
    counters = {"w": 0, "c": 0, "flat": 0}
    for offset, ((number, _), text) in enumerate(zip(region_lines, cleaned)):
        count = len(tokenize_line(text))
        region = "w" if offset < w_size else "c"
        for key in (region, "flat"):
            if count:
                positions[key][number] = (counters[key], counters[key] + count - 1)
            counters[key] += count
    return positions


def finding_visibility(
    source: str, findings: list[Finding], max_len: int = 500, w_ratio: float = 0.6
) -> dict[str, bool]:
    w_budget = int((max_len - 1) * w_ratio)
    c_budget = max_len - 1 - w_budget
    positions = token_positions(source)

    def visible(region: str, budget: int) -> bool:
        return any(
            positions[region][line][0] < budget
            for finding in findings
            for line in range(finding.first_line, finding.last_line + 1)
            if line in positions[region]
        )

    return {
        "in_first_function": any(
            line in positions["w"]
            for finding in findings
            for line in range(finding.first_line, finding.last_line + 1)
        ),
        "extracted_at_all": any(
            line in positions["flat"]
            for finding in findings
            for line in range(finding.first_line, finding.last_line + 1)
        ),
        "visible_region_prefix": visible("w", w_budget) or visible("c", c_budget),
        "visible_flat_prefix": visible("flat", max_len),
    }


def _functions(source_lines: list[str]) -> list[tuple[int, int]]:
    """Approximate (first, last) 1-based line spans of functions by brace matching."""
    spans = []
    for index, line in enumerate(source_lines):
        if not re.match(r"\s*(function|constructor|fallback|receive)\b", line):
            continue
        depth, opened = 0, False
        for end in range(index, len(source_lines)):
            depth += source_lines[end].count("{") - source_lines[end].count("}")
            opened = opened or "{" in source_lines[end]
            if opened and depth <= 0:
                spans.append((index + 1, end + 1))
                break
            if not opened and ";" in source_lines[end]:
                break
    return spans


def review_sheet(source: str, findings: list[Finding], limit: int = 12) -> str:
    """Finding spans plus every function containing an external call."""
    lines = _strip_comments(source).splitlines()
    spans = {(f.first_line, f.last_line) for f in findings}
    spans |= {
        span for span in _functions(lines)
        if any(_EXTERNAL_CALL.search(lines[n - 1]) for n in range(span[0], span[1] + 1))
    }
    parts = []
    for first, last in sorted(spans)[:limit]:
        body = "\n".join(
            f"{n:5d}  {lines[n - 1]}" for n in range(first, min(last, len(lines)) + 1)
        )
        parts.append(f"--- L{first}-L{last} ---\n{body}")
    return "\n\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dappscan-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("Dataset/dappscan_reentrancy.txt"))
    parser.add_argument("--neg-ratio", type=float, default=2.0)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--audit-seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path("EXPERIMENT/label-audit"))
    args = parser.parse_args()

    samples = sample_sources(args.dappscan_root, args.dataset, args.neg_ratio, args.dataset_seed)
    findings = reentrancy_findings(args.dappscan_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    visibility_rows = []
    for sample_id, path, label in samples:
        if label != 1:
            continue
        source = path.read_text(encoding="utf-8", errors="ignore")
        visibility_rows.append({
            "sample_id": sample_id,
            "findings": len(findings[path]),
            **finding_visibility(source, locate_findings(source, findings[path])),
        })
    totals = {
        key: sum(row[key] for row in visibility_rows)
        for key in ("in_first_function", "extracted_at_all", "visible_region_prefix", "visible_flat_prefix")
    }
    totals["positives"] = len(visibility_rows)
    totals["positives_with_line_findings"] = sum(row["findings"] > 0 for row in visibility_rows)
    (args.output_dir / "finding_visibility.json").write_text(
        json.dumps({"totals": totals, "rows": visibility_rows}, indent=2), encoding="utf-8"
    )

    rng = random.Random(args.audit_seed)
    chosen = []
    for label in (1, 0):
        pool = [item for item in samples if item[2] == label]
        chosen += rng.sample(pool, min(args.per_class, len(pool)))

    sheets = args.output_dir / "sheets"
    sheets.mkdir(exist_ok=True)
    with (args.output_dir / "audit_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "sample_id", "label", "source_path", "finding_functions", "finding_lines",
            "verdict", "reason",
        ])
        for sample_id, path, label in chosen:
            source = path.read_text(encoding="utf-8", errors="ignore")
            found = locate_findings(source, findings[path])
            writer.writerow([
                sample_id, label, str(path.relative_to(args.dappscan_root)),
                ";".join(f.function for f in found),
                ";".join(f"L{f.first_line}-L{f.last_line}" for f in found),
                "", "",
            ])
            safe_name = re.sub(r"[^\w.-]", "_", sample_id)
            (sheets / f"{label}_{safe_name}.txt").write_text(
                f"# {sample_id} label={label}\n# {path.relative_to(args.dappscan_root)}\n\n"
                + review_sheet(source, found),
                encoding="utf-8",
            )
    print(json.dumps(totals, indent=2))
    print(f"Wrote {len(chosen)} review sheets to {sheets}")


if __name__ == "__main__":
    main()
