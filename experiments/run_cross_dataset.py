"""Cross-dataset validation: Train on 1671, evaluate zero-shot on DAppSCAN."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
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
from representation.sequences import build_sequence, embedding_tokens
from representation.variants import VARIANTS, VariantSpec
from representation.vectorization import vectorize_examples


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


def run_cross_dataset_eval(
    spec: VariantSpec,
    train_file: Path,
    test_files: dict[str, Path],
    config: ExperimentConfig,
    seed: int,
) -> dict[str, object]:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.utils import to_categorical

    set_seed(seed)
    local_config = ExperimentConfig(
        max_len=config.max_len,
        vector_dim=config.vector_dim,
        hidden_units=config.hidden_units,
        dense_units=config.dense_units,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        epochs=config.epochs,
        patience=config.patience,
        folds=config.folds,
        validation_folds=config.validation_folds,
        w_ratio=config.w_ratio,
        min_recall=config.min_recall,
        threshold_strategy=config.threshold_strategy,
        early_stop_monitor=config.early_stop_monitor,
        seed=seed,
        verbose=0,
    )

    print(f"\n=======================================================", flush=True)
    print(f"=== Running Variant {spec.name} (Seed {seed}) ===", flush=True)
    print(f"=======================================================", flush=True)
    train_records = load_records(train_file)
    test_datasets = {name: load_records(path) for name, path in test_files.items()}

    fit_indices, val_indices = make_validation_split(train_records, local_config)
    fit_records = [train_records[i] for i in fit_indices]
    val_records = [train_records[i] for i in val_indices]
    assert_group_disjoint(fit_records, val_records, "train vs validation split")

    print(f"Train (Fit): {len(fit_records)} samples, Val: {len(val_records)} samples", flush=True)
    for name, recs in test_datasets.items():
        print(f"Test ({name}): {len(recs)} samples (vuln: {sum(r.label for r in recs)})", flush=True)

    t0 = time.perf_counter()
    corpus = [embedding_tokens(r, spec) for r in fit_records]
    embedding_model = fit_embedding(corpus, local_config.vector_dim, seed, spec.embedding)
    print(f"Fitted {spec.embedding} embedding in {time.perf_counter() - t0:.2f}s", flush=True)

    fit_ex = _examples(fit_records, spec, local_config)
    val_ex = _examples(val_records, spec, local_config)
    fit_x = _model_inputs(fit_ex, embedding_model.wv, spec, local_config)
    val_x = _model_inputs(val_ex, embedding_model.wv, spec, local_config)

    test_x_dict = {}
    for name, recs in test_datasets.items():
        exs = _examples(recs, spec, local_config)
        test_x_dict[name] = _model_inputs(exs, embedding_model.wv, spec, local_config)

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
    print(f"Model trained in {epochs_trained} epochs ({time.perf_counter() - t_train:.1f}s).", flush=True)

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
        }

        print(f"\n--- Result on {name} (Threshold Recall={recall_threshold:.4f}) ---", flush=True)
        print(f"  PR-AUC:    {res_recall['pr_auc']*100:.2f}%", flush=True)
        print(f"  F1:        {res_recall['f1_vulnerable']*100:.2f}%", flush=True)
        print(f"  Recall:    {res_recall['recall_vulnerable']*100:.2f}%", flush=True)
        print(f"  Precision: {res_recall['precision_vulnerable']*100:.2f}%", flush=True)
        print(f"  ROC-AUC:   {res_recall['roc_auc']*100:.2f}%", flush=True)
        print(f"  EER:       {res_recall['eer']*100:.2f}%", flush=True)

        print(f"--- Result on {name} (Threshold EER={eer_threshold:.4f}) ---", flush=True)
        print(f"  F1:        {res_eer['f1_vulnerable']*100:.2f}%", flush=True)
        print(f"  Recall:    {res_eer['recall_vulnerable']*100:.2f}%", flush=True)
        print(f"  Precision: {res_eer['precision_vulnerable']*100:.2f}%", flush=True)

    del model, fit_x, val_x, test_x_dict
    tf.keras.backend.clear_session()
    gc.collect()

    return {
        "variant": spec.name,
        "seed": seed,
        "epochs_trained": epochs_trained,
        "evaluations": evaluations,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="b3,b0")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--output-dir", default="EXPERIMENT/cross-dataset")
    args = parser.parse_args()

    train_file = Path("Dataset/reentrancy_1671.txt")
    test_files = {
        "dappscan_ratio2": Path("Dataset/dappscan_reentrancy.txt"),
        "dappscan_ratio1": Path("Dataset/dappscan_reentrancy_ratio1.txt"),
    }

    variants = [VARIANTS[v.strip()] for v in args.variants.split(",") if v.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    config = ExperimentConfig(
        max_len=500,
        vector_dim=300,
        hidden_units=300,
        dense_units=300,
        dropout=0.2,
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
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                all_results = json.load(f)
        except Exception:
            all_results = []

    for spec in variants:
        for seed in seeds:
            if any(r["variant"] == spec.name and r["seed"] == seed for r in all_results):
                print(f"Skipping already completed Variant {spec.name} (Seed {seed})", flush=True)
                continue
            res = run_cross_dataset_eval(spec, train_file, test_files, config, seed)
            all_results.append(res)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)

    print(f"\nSaved cross-dataset results to {out_file}", flush=True)

    # Print summary table
    print("\n" + "=" * 80, flush=True)
    print("CROSS-DATASET VALIDATION SUMMARY (Train: 1671, Test: DAppSCAN)", flush=True)
    print("=" * 80, flush=True)

    for target in ["dappscan_ratio2", "dappscan_ratio1"]:
        print(f"\n>>> Test Dataset: {target}", flush=True)
        for spec in variants:
            var_results = [r for r in all_results if r["variant"] == spec.name and target in r.get("evaluations", {})]
            if not var_results:
                continue
            praucs = [r["evaluations"][target]["metrics_under_recall_threshold"]["pr_auc"] for r in var_results]
            recalls = [r["evaluations"][target]["metrics_under_recall_threshold"]["recall_vulnerable"] for r in var_results]
            precisions = [r["evaluations"][target]["metrics_under_recall_threshold"]["precision_vulnerable"] for r in var_results]
            f1s = [r["evaluations"][target]["metrics_under_recall_threshold"]["f1_vulnerable"] for r in var_results]
            eers = [r["evaluations"][target]["metrics_under_recall_threshold"]["eer"] for r in var_results]

            print(
                f"Variant {spec.name:10} | "
                f"PR-AUC: {np.mean(praucs)*100:5.2f}% ± {np.std(praucs)*100:4.2f}% | "
                f"F1: {np.mean(f1s)*100:5.2f}% ± {np.std(f1s)*100:4.2f}% | "
                f"Recall: {np.mean(recalls)*100:5.2f}% ± {np.std(recalls)*100:4.2f}% | "
                f"Precision: {np.mean(precisions)*100:5.2f}% ± {np.std(precisions)*100:4.2f}% | "
                f"EER: {np.mean(eers)*100:5.2f}% ± {np.std(eers)*100:4.2f}%",
                flush=True,
            )


if __name__ == "__main__":
    main()
