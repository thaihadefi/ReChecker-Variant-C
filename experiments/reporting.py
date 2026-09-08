"""Strict fold-result validation and paired summaries."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from representation.variants import VariantSpec

from .artifacts import read_json, write_json


def _prediction_ids(result: dict) -> set[str]:
    predictions = result.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError(
            f"{result.get('variant')} fold {result.get('fold')} has no prediction list"
        )
    sample_ids = [item["sample_id"] for item in predictions]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(
            f"{result.get('variant')} fold {result.get('fold')} has duplicate predictions"
        )
    expected_size = result.get("outer_test_size")
    if expected_size is not None and len(sample_ids) != expected_size:
        raise ValueError(
            f"{result.get('variant')} fold {result.get('fold')} has incomplete predictions"
        )
    return set(sample_ids)


def _numeric_metrics(results: list[dict]) -> list[str]:
    if not results:
        return []
    names = set(results[0]["metrics"])
    for result in results[1:]:
        names &= set(result["metrics"])
    return sorted(
        name
        for name in names
        if name != "threshold"
        and all(isinstance(result["metrics"][name], (int, float)) for result in results)
    )


def summarize_run(
    run_dir: str | Path,
    expected_variants: tuple[VariantSpec, ...],
    allow_partial: bool = False,
) -> dict[str, object]:
    run_dir = Path(run_dir)
    config = read_json(run_dir / "config.json")
    expected_folds = set(range(int(config["folds"])))
    requested = {spec.name for spec in expected_variants}
    by_variant: dict[str, dict[int, dict]] = defaultdict(dict)

    for path in sorted((run_dir / "results").glob("fold_*_*.json")):
        result = read_json(path)
        variant = result["variant"]
        fold = int(result["fold"])
        if variant not in requested:
            continue
        if fold in by_variant[variant]:
            raise ValueError(f"duplicate result for {variant} fold {fold}")
        if fold not in expected_folds:
            raise ValueError(f"out-of-range fold {fold} for {variant}")
        _prediction_ids(result)
        by_variant[variant][fold] = result

    missing = {
        variant: sorted(expected_folds - set(by_variant.get(variant, {})))
        for variant in sorted(requested)
        if expected_folds - set(by_variant.get(variant, {}))
    }
    if missing and not allow_partial:
        raise ValueError(f"run is incomplete; missing folds: {missing}")

    summary: dict[str, object] = {
        "complete": not missing,
        "expected_folds": len(expected_folds),
        "folds_found": {
            variant: len(by_variant.get(variant, {})) for variant in sorted(requested)
        },
        "missing_folds": missing,
    }
    for variant in sorted(requested):
        results = list(by_variant.get(variant, {}).values())
        if not results:
            continue
        summary[variant] = {
            name: {
                "mean": float(np.mean([item["metrics"][name] for item in results])),
                "std": float(np.std([item["metrics"][name] for item in results])),
            }
            for name in _numeric_metrics(results)
        }

    baseline = by_variant.get("b0", {})
    for variant in sorted(requested - {"b0"}):
        compared = by_variant.get(variant, {})
        common = sorted(set(baseline) & set(compared))
        if not common:
            continue
        for index in common:
            if _prediction_ids(baseline[index]) != _prediction_ids(compared[index]):
                raise ValueError(
                    f"paired variants use different samples in outer fold {index}"
                )
        results = [baseline[index] for index in common] + [compared[index] for index in common]
        summary[f"paired_{variant}_minus_b0"] = {
            name: {
                "mean": float(np.mean([
                    compared[index]["metrics"][name] - baseline[index]["metrics"][name]
                    for index in common
                ])),
                "std": float(np.std([
                    compared[index]["metrics"][name] - baseline[index]["metrics"][name]
                    for index in common
                ])),
            }
            for name in _numeric_metrics(results)
        }

    write_json(run_dir / "summary.json", summary)
    return summary
