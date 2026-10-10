"""Cross-dataset validation: train on 1671 (optionally pruned or expanded),
evaluate zero-shot on DAppSCAN.

Test-set labels and probabilities, and the source validation probabilities,
are persisted so threshold/distribution analyses can be redone without
retraining (see experiments/distribution_shift.py).
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from config.config import ExperimentConfig
from data.loader import load_records
from data.types import GadgetRecord
from experiments.metrics import (
    evaluate_predictions,
    select_eer_threshold,
    select_recall_threshold,
)
from experiments.reproducibility import set_seed
from experiments.splits import assert_group_disjoint, make_validation_split
from models.models import build_model
from representation.embeddings import fit_embedding
from representation.pruning import (
    abstract_identifiers,
    deduplicate_groups,
    fit_group_vocabulary,
    prune_vocabulary,
    sample_group_fraction,
)
from representation.sequences import build_sequence, embedding_tokens
from representation.variants import VARIANTS, VariantSpec
from representation.vectorization import vectorize_examples


DEFAULT_TEST_FILES = {
    "dappscan_ratio2": Path("Dataset/dappscan_reentrancy.txt"),
    "dappscan_ratio1": Path("Dataset/dappscan_reentrancy_ratio1.txt"),
}


@dataclass(frozen=True)
class TrainingSetup:
    """Training-data interventions; the defaults reproduce the original protocol."""

    dedup: bool = False
    abstract_ids: bool = False
    min_token_groups: int = 1
    train_fraction: float = 1.0
    extra_train_files: tuple[str, ...] = ()


def _examples(records: list[GadgetRecord], spec: VariantSpec, config: ExperimentConfig):
    return [
        build_sequence(record, spec, config.max_len, config.w_ratio)
        for record in records
    ]


def _model_inputs(examples, embeddings, spec: VariantSpec, config: ExperimentConfig):
    vectors = vectorize_examples(
        examples, embeddings, config.max_len, config.vector_dim
    )
    if spec.use_segment_embedding:
        auxiliary = np.stack([item.segment_ids for item in examples])
    else:
        auxiliary = np.stack([item.valid_mask for item in examples])
    return [vectors, auxiliary]


def _class_weights(labels: np.ndarray) -> dict[int, float]:
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def prepare_records(
    train_records: list[GadgetRecord],
    test_datasets: dict[str, list[GadgetRecord]],
    setup: TrainingSetup,
    config: ExperimentConfig,
    seed: int,
):
    """Apply the setup's interventions and return fit/validation/test records.

    Everything learned from data (vocabulary) is fitted on the fit partition
    only; test labels are never consulted.
    """
    if setup.abstract_ids:
        train_records = abstract_identifiers(train_records)
        test_datasets = {
            name: abstract_identifiers(records) for name, records in test_datasets.items()
        }
    if setup.dedup:
        train_records = deduplicate_groups(train_records)
    train_records = sample_group_fraction(train_records, setup.train_fraction, seed)

    fit_indices, val_indices = make_validation_split(train_records, config)
    fit_records = [train_records[i] for i in fit_indices]
    val_records = [train_records[i] for i in val_indices]
    assert_group_disjoint(fit_records, val_records, "train vs validation split")

    if setup.min_token_groups > 1:
        vocabulary = fit_group_vocabulary(fit_records, setup.min_token_groups)
        fit_records = prune_vocabulary(fit_records, vocabulary)
        val_records = prune_vocabulary(val_records, vocabulary)
        test_datasets = {
            name: prune_vocabulary(records, vocabulary)
            for name, records in test_datasets.items()
        }
    return fit_records, val_records, test_datasets


def run_cross_dataset_eval(
    spec: VariantSpec,
    train_file: Path,
    test_files: dict[str, Path],
    config: ExperimentConfig,
    seed: int,
    setup: TrainingSetup = TrainingSetup(),
) -> dict[str, object]:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.utils import to_categorical

    set_seed(seed)
    local_config = replace(config, seed=seed, verbose=0)

    print("\n=======================================================", flush=True)
    print(f"=== Running Variant {spec.name} (Seed {seed}) ===", flush=True)
    print("=======================================================", flush=True)
    train_records = load_records(train_file)
    for extra in setup.extra_train_files:
        train_records = train_records + load_records(extra)
    test_datasets = {name: load_records(path) for name, path in test_files.items()}

    fit_records, val_records, test_datasets = prepare_records(
        train_records, test_datasets, setup, local_config, seed
    )
    train_groups = {r.group_id for r in fit_records + val_records}
    overlap = {
        name: len(train_groups & {r.group_id for r in recs})
        for name, recs in test_datasets.items()
    }

    print(f"Train (Fit): {len(fit_records)} samples, Val: {len(val_records)} samples", flush=True)
    for name, recs in test_datasets.items():
        print(
            f"Test ({name}): {len(recs)} samples (vuln: {sum(r.label for r in recs)}, "
            f"groups shared with train: {overlap[name]})",
            flush=True,
        )

    t0 = time.perf_counter()
    corpus = [embedding_tokens(r, spec) for r in fit_records]
    embedding_model = fit_embedding(corpus, local_config.vector_dim, seed, spec.embedding)
    print(f"Fitted {spec.embedding} embedding in {time.perf_counter() - t0:.2f}s", flush=True)

    fit_x = _model_inputs(_examples(fit_records, spec, local_config), embedding_model.wv, spec, local_config)
    val_x = _model_inputs(_examples(val_records, spec, local_config), embedding_model.wv, spec, local_config)
    test_x_dict = {
        name: _model_inputs(_examples(recs, spec, local_config), embedding_model.wv, spec, local_config)
        for name, recs in test_datasets.items()
    }

    del embedding_model
    gc.collect()

    fit_y = np.asarray([r.label for r in fit_records], dtype=np.int64)
    val_y = np.asarray([r.label for r in val_records], dtype=np.int64)

    tf.keras.backend.clear_session()
    model = build_model(local_config, spec)
    print("Training BiLSTM model...", flush=True)
    t_train = time.perf_counter()
    history = model.fit(
        fit_x,
        to_categorical(fit_y, num_classes=2),
        validation_data=(val_x, to_categorical(val_y, num_classes=2)),
        class_weight=_class_weights(fit_y),
        batch_size=local_config.batch_size,
        epochs=local_config.epochs,
        callbacks=[EarlyStopping(
            monitor=local_config.early_stop_monitor,
            mode="min" if local_config.early_stop_monitor == "val_loss" else "max",
            patience=local_config.patience,
            restore_best_weights=True,
        )],
        verbose=0,
    )
    epochs_trained = len(history.history["loss"])
    training_seconds = time.perf_counter() - t_train
    print(f"Model trained in {epochs_trained} epochs ({training_seconds:.1f}s).", flush=True)

    val_probs = model.predict(val_x, batch_size=local_config.batch_size, verbose=0)[:, 1]
    recall_threshold = select_recall_threshold(val_y, val_probs, local_config.min_recall)
    eer_threshold = select_eer_threshold(val_y, val_probs)

    evaluations = {}
    for name, test_recs in test_datasets.items():
        test_y = np.asarray([r.label for r in test_recs], dtype=np.int64)
        test_probs = model.predict(test_x_dict[name], batch_size=local_config.batch_size, verbose=0)[:, 1]

        res_recall = evaluate_predictions(test_y, test_probs, recall_threshold)
        res_eer = evaluate_predictions(test_y, test_probs, eer_threshold)
        evaluations[name] = {
            "recall_threshold": recall_threshold,
            "eer_threshold": eer_threshold,
            "metrics_under_recall_threshold": res_recall,
            "metrics_under_eer_threshold": res_eer,
            "sample_ids": [r.sample_id for r in test_recs],
            "labels": test_y.tolist(),
            "probabilities": test_probs.astype(float).tolist(),
        }

        print(
            f"--- {name}: ROC-AUC {res_recall['roc_auc']*100:.2f}% | PR-AUC {res_recall['pr_auc']*100:.2f}% | "
            f"F1@recall-thr {res_recall['f1_vulnerable']*100:.2f}% | F1@EER-thr {res_eer['f1_vulnerable']*100:.2f}%",
            flush=True,
        )

    del model, fit_x, val_x, test_x_dict
    tf.keras.backend.clear_session()
    gc.collect()

    return {
        "variant": spec.name,
        "seed": seed,
        "setup": asdict(setup),
        "model_config": {
            "hidden_units": local_config.hidden_units,
            "dense_units": local_config.dense_units,
            "dropout": local_config.dropout,
            "token_dropout": local_config.token_dropout,
        },
        "sizes": {
            "fit": len(fit_records),
            "validation": len(val_records),
            "fit_positive": int(fit_y.sum()),
        },
        "groups_shared_with_test": overlap,
        "epochs_trained": epochs_trained,
        "training_seconds": training_seconds,
        "validation": {
            "labels": val_y.tolist(),
            "probabilities": val_probs.astype(float).tolist(),
        },
        "evaluations": evaluations,
    }


def _parse_test_files(value: str | None) -> dict[str, Path]:
    if not value:
        return dict(DEFAULT_TEST_FILES)
    files = {}
    for item in value.split(","):
        name, _, path = item.partition("=")
        if not name or not path:
            raise ValueError(f"--test-files entries must look like name=path, got {item!r}")
        files[name.strip()] = Path(path.strip())
    return files


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="b3,b0")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--output-dir", default="EXPERIMENT/cross-dataset")
    parser.add_argument("--train-file", type=Path, default=Path("Dataset/reentrancy_1671.txt"))
    parser.add_argument(
        "--extra-train", default="",
        help="comma-separated gadget files appended to the training source",
    )
    parser.add_argument(
        "--test-files",
        help="comma-separated name=path pairs (default: the two DAppSCAN ratio files)",
    )
    parser.add_argument("--dedup", action="store_true", help="keep one sample per exact-duplicate group")
    parser.add_argument("--abstract-ids", action="store_true", help="VARn/FUNn -> VAR/FUN, literals -> NUM")
    parser.add_argument(
        "--min-token-groups", type=int, default=1,
        help="replace tokens seen in fewer training groups than this with [UNK]",
    )
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--hidden-units", type=int, default=300)
    parser.add_argument("--dense-units", type=int, default=300)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--token-dropout", type=float, default=0.0)
    return parser


def main():
    args = _parser().parse_args()
    test_files = _parse_test_files(args.test_files)
    setup = TrainingSetup(
        dedup=args.dedup,
        abstract_ids=args.abstract_ids,
        min_token_groups=args.min_token_groups,
        train_fraction=args.train_fraction,
        extra_train_files=tuple(p.strip() for p in args.extra_train.split(",") if p.strip()),
    )

    variants = [VARIANTS[v.strip()] for v in args.variants.split(",") if v.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    config = ExperimentConfig(
        max_len=500,
        vector_dim=300,
        hidden_units=args.hidden_units,
        dense_units=args.dense_units,
        dropout=args.dropout,
        token_dropout=args.token_dropout,
        learning_rate=0.002,
        batch_size=32,
        epochs=40,
        patience=5,
        w_ratio=0.6,
        min_recall=0.95,
        early_stop_monitor="val_pr_auc",
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "cross_dataset_results.json"

    all_results = []
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            all_results = json.load(f)

    for spec in variants:
        for seed in seeds:
            if any(r["variant"] == spec.name and r["seed"] == seed for r in all_results):
                print(f"Skipping already completed Variant {spec.name} (Seed {seed})", flush=True)
                continue
            res = run_cross_dataset_eval(spec, args.train_file, test_files, config, seed, setup)
            all_results.append(res)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)

    print(f"\nSaved cross-dataset results to {out_file}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print(f"CROSS-DATASET VALIDATION SUMMARY ({out_dir})", flush=True)
    print("=" * 80, flush=True)
    for target in test_files:
        print(f"\n>>> Test Dataset: {target}", flush=True)
        for spec in variants:
            runs = [r for r in all_results if r["variant"] == spec.name and target in r.get("evaluations", {})]
            if not runs:
                continue
            metrics = [r["evaluations"][target]["metrics_under_recall_threshold"] for r in runs]

            def stat(key):
                values = [m[key] for m in metrics]
                return f"{np.mean(values)*100:5.2f}% ± {np.std(values)*100:4.2f}%"

            print(
                f"Variant {spec.name:4} | ROC-AUC {stat('roc_auc')} | PR-AUC {stat('pr_auc')} | "
                f"F1 {stat('f1_vulnerable')} | Recall {stat('recall_vulnerable')} | "
                f"Precision {stat('precision_vulnerable')}",
                flush=True,
            )


if __name__ == "__main__":
    main()
