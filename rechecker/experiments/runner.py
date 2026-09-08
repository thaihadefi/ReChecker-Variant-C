"""Train and evaluate one leakage-safe fold for any B0-B4 variant."""

from __future__ import annotations

import gc
import resource
import sys
import time
from collections.abc import Sequence

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from rechecker.config import ExperimentConfig
from rechecker.data.types import GadgetRecord
from rechecker.modeling.inspection import count_parameters
from rechecker.representation.embeddings import fit_embedding, save_embedding
from rechecker.representation.sequences import (
    build_sequence,
    coverage,
    embedding_tokens,
)
from rechecker.representation.variants import VariantSpec
from rechecker.representation.vectorization import vectorize_examples

from .artifacts import FoldArtifactPaths, artifact_bytes, write_bundle_manifest
from .metrics import benchmark_predict, evaluate_predictions, select_recall_threshold
from .reproducibility import set_seed
from .splits import assert_group_disjoint, make_validation_split


def _class_weights(labels: np.ndarray) -> dict[int, float]:
    classes = np.unique(labels)
    if len(classes) != 2:
        raise ValueError("fit partition must contain both classes")
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def _examples(records, spec, config):
    return [
        build_sequence(record, spec, config.max_len, config.w_ratio)
        for record in records
    ]


def _model_inputs(examples, embeddings, spec, config):
    vectors = vectorize_examples(
        examples, embeddings, config.max_len, config.vector_dim
    )
    if spec.use_segment_embedding:
        auxiliary = np.stack([item.segment_ids for item in examples])
    else:
        auxiliary = np.stack([item.valid_mask for item in examples])
    return [vectors, auxiliary]


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return value / divisor


def train_fold(
    spec: VariantSpec,
    train_records: Sequence[GadgetRecord],
    test_records: Sequence[GadgetRecord],
    config: ExperimentConfig,
    paths: FoldArtifactPaths,
) -> dict[str, object]:
    """Train with inner validation, then predict on the untouched outer test once."""
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.utils import to_categorical

    from rechecker.modeling.models import build_model

    set_seed(config.seed)
    fit_indices, validation_indices = make_validation_split(train_records, config)
    fit_records = [train_records[index] for index in fit_indices]
    validation_records = [train_records[index] for index in validation_indices]
    assert_group_disjoint(fit_records, validation_records, "validation split")

    embedding_started = time.perf_counter()
    corpus = [embedding_tokens(record, spec) for record in fit_records]
    embedding_model = fit_embedding(
        corpus, config.vector_dim, config.seed, spec.embedding
    )
    embedding_seconds = time.perf_counter() - embedding_started
    save_embedding(embedding_model, paths.embedding)

    fit_examples = _examples(fit_records, spec, config)
    validation_examples = _examples(validation_records, spec, config)
    test_examples = _examples(test_records, spec, config)
    fit_x = _model_inputs(fit_examples, embedding_model.wv, spec, config)
    validation_x = _model_inputs(
        validation_examples, embedding_model.wv, spec, config
    )
    test_x = _model_inputs(test_examples, embedding_model.wv, spec, config)
    del embedding_model
    gc.collect()

    fit_y = np.asarray([record.label for record in fit_records], dtype=np.int64)
    validation_y = np.asarray(
        [record.label for record in validation_records], dtype=np.int64
    )
    test_y = np.asarray([record.label for record in test_records], dtype=np.int64)

    tf.keras.backend.clear_session()
    model = build_model(config, spec)
    training_started = time.perf_counter()
    history = model.fit(
        fit_x,
        to_categorical(fit_y, num_classes=2),
        validation_data=(validation_x, to_categorical(validation_y, num_classes=2)),
        class_weight=_class_weights(fit_y),
        batch_size=config.batch_size,
        epochs=config.epochs,
        callbacks=[EarlyStopping(
            monitor="val_loss",
            patience=config.patience,
            restore_best_weights=True,
        )],
        verbose=config.verbose,
    )
    training_seconds = time.perf_counter() - training_started

    validation_probability = model.predict(
        validation_x, batch_size=config.batch_size, verbose=0
    )[:, 1]
    threshold = select_recall_threshold(
        validation_y, validation_probability, config.min_recall
    )
    test_probability, inference = benchmark_predict(
        lambda: model.predict(
            test_x, batch_size=config.batch_size, verbose=0
        )[:, 1],
        len(test_records),
    )
    metrics = evaluate_predictions(test_y, test_probability, threshold)

    paths.weights.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(paths.weights)
    write_bundle_manifest(paths, config, spec, threshold)

    result = {
        "schema_version": 1,
        "variant": spec.name,
        "variant_description": spec.description,
        "metrics": metrics,
        "coverage": coverage(test_records, test_examples) if spec.use_separator else None,
        "performance": {
            "embedding_fit_seconds": embedding_seconds,
            "model_training_seconds": training_seconds,
            "peak_rss_mb": _peak_rss_mb(),
            "inference": inference,
            "weights_bytes": artifact_bytes(paths.weights),
            "embedding_bytes": artifact_bytes(paths.embedding),
        },
        "model": count_parameters(model),
        "epochs_trained": len(history.history["loss"]),
        "best_validation_loss": float(min(history.history["val_loss"])),
        "sizes": {
            "fit": len(fit_records),
            "validation": len(validation_records),
            "test": len(test_records),
        },
        "predictions": [
            {
                "sample_id": record.sample_id,
                "group_id": record.group_id,
                "label": record.label,
                "probability": float(probability),
                "prediction": int(probability >= threshold),
            }
            for record, probability in zip(test_records, test_probability)
        ],
    }
    del model, fit_x, validation_x, test_x
    tf.keras.backend.clear_session()
    gc.collect()
    return result
