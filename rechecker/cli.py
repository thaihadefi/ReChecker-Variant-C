"""Command-line entry point for leakage-safe, group-aware experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from rechecker.config import ExperimentConfig
from rechecker.data import load_records
from rechecker.representation.variants import ALIASES, VARIANTS, resolve_variants

from .experiments.artifacts import (
    FoldArtifactPaths,
    environment_metadata,
    file_sha256,
    read_json,
    source_fingerprint,
    write_json,
)
from .experiments.reporting import summarize_run
from .experiments.splits import assert_group_disjoint, make_outer_splits


VARIANT_CHOICES = ("all", *VARIANTS, *ALIASES)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=Path("Dataset/reentrancy_1671.txt"))
    parser.add_argument("--output-root", type=Path, default=Path("EXPERIMENT"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--fold", type=int, help="zero-based outer fold to run")
    parser.add_argument("--variant", choices=VARIANT_CHOICES, default="all")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--allow-partial-summary", action="store_true")
    parser.add_argument("--max-len", type=int, default=500)
    parser.add_argument("--vector-dim", type=int, default=300)
    parser.add_argument("--hidden-units", type=int, default=300)
    parser.add_argument("--dense-units", type=int, default=300)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--validation-folds", type=int, default=5)
    parser.add_argument("--w-ratio", type=float, default=0.6)
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=2)
    return parser


def _config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        data_file=args.data_file,
        output_root=args.output_root,
        max_len=args.max_len,
        vector_dim=args.vector_dim,
        hidden_units=args.hidden_units,
        dense_units=args.dense_units,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        folds=args.folds,
        validation_folds=args.validation_folds,
        w_ratio=args.w_ratio,
        min_recall=args.min_recall,
        seed=args.seed,
        verbose=args.verbose,
    )


def _initialize_run(
    run_dir: Path, config: ExperimentConfig, records
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    if config_path.exists() and read_json(config_path) != config.to_dict():
        raise RuntimeError(f"{run_dir} contains a different experiment configuration")
    write_json(config_path, config.to_dict())
    write_json(run_dir / "dataset.json", {
        "samples": len(records),
        "vulnerable": sum(record.label for record in records),
        "groups": len({record.group_id for record in records}),
        "dataset_sha256": file_sha256(config.data_file),
        "source_sha256": source_fingerprint(),
        "w_c_extraction_strategy": "first-function-declaration-block",
        "w_region_diagnostics": {
            "with_dropped_preamble": sum(
                record.diagnostics.dropped_preamble_lines > 0 for record in records
            ),
            "with_modifier_declaration": sum(
                record.diagnostics.w_contains_modifier_declaration for record in records
            ),
            "with_contract_declaration": sum(
                record.diagnostics.w_contains_contract_declaration for record in records
            ),
        },
        "environment": environment_metadata(),
    })


def _run_one(args: argparse.Namespace, config: ExperimentConfig, run_dir: Path) -> None:
    from .experiments.runner import train_fold

    if args.fold is None:
        raise SystemExit("--fold is required unless --run-all or --summarize is used")
    specs = resolve_variants(args.variant)
    if len(specs) != 1:
        raise SystemExit("a single-fold process requires one explicit --variant")
    records = load_records(config.data_file)
    splits = make_outer_splits(records, config)
    if not 0 <= args.fold < len(splits):
        raise SystemExit(f"--fold must be between 0 and {len(splits) - 1}")
    _initialize_run(run_dir, config, records)
    train_indices, test_indices = splits[args.fold]
    train_records = [records[index] for index in train_indices]
    test_records = [records[index] for index in test_indices]
    assert_group_disjoint(train_records, test_records, "outer split")
    spec = specs[0]
    paths = FoldArtifactPaths(run_dir, args.fold, spec.name)
    result = train_fold(spec, train_records, test_records, config, paths)
    result.update({
        "fold": args.fold,
        "outer_train_size": len(train_records),
        "outer_test_size": len(test_records),
    })
    write_json(paths.result, result)
    print(json.dumps(result["metrics"], indent=2))


def _run_all(args: argparse.Namespace, config: ExperimentConfig, run_dir: Path) -> None:
    specs = resolve_variants(args.variant)
    records = load_records(config.data_file)
    _initialize_run(run_dir, config, records)
    for fold in range(config.folds):
        for spec in specs:
            paths = FoldArtifactPaths(run_dir, fold, spec.name)
            if paths.is_complete():
                print(f"skip complete {paths.stem}")
                continue
            command = [
                sys.executable,
                "-m",
                "rechecker.cli",
                *config.to_cli_args(),
                "--run-dir", str(run_dir),
                "--fold", str(fold),
                "--variant", spec.name,
            ]
            subprocess.run(command, check=True)
    summary = summarize_run(run_dir, specs)
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = _parser().parse_args()
    if args.run_all and args.summarize:
        raise SystemExit("--run-all and --summarize are mutually exclusive")
    config = _config(args)
    run_dir = args.run_dir or config.new_run_dir()
    specs = resolve_variants(args.variant)
    if args.summarize:
        if args.run_dir is None:
            raise SystemExit("--summarize requires --run-dir")
        summary = summarize_run(
            run_dir, specs, allow_partial=args.allow_partial_summary
        )
        print(json.dumps(summary, indent=2))
    elif args.run_all:
        _run_all(args, config, run_dir)
    else:
        _run_one(args, config, run_dir)


if __name__ == "__main__":
    main()
