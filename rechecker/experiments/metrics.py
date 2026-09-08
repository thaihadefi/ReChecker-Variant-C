"""Classification metrics and validation-only threshold selection."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _validated_arrays(y_true, probabilities):
    labels = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("labels and probabilities must be equal-length 1-D arrays")
    if not len(labels):
        raise ValueError("labels and probabilities cannot be empty")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(scores).all():
        raise ValueError("probabilities must be finite")
    if ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("probabilities must be in [0, 1]")
    return labels, scores


def select_recall_threshold(y_true, probabilities, min_recall: float) -> float:
    labels, scores = _validated_arrays(y_true, probabilities)
    if not 0.0 < min_recall <= 1.0:
        raise ValueError("min_recall must be in (0, 1]")
    positives = scores[labels == 1]
    if not len(positives):
        return 0.5
    rank = max(0, len(positives) - int(np.ceil(min_recall * len(positives))))
    return float(np.sort(positives)[rank])


def _eer(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    index = int(
        np.nanargmin(np.abs((1.0 - true_positive_rate) - false_positive_rate))
    )
    return float(
        (false_positive_rate[index] + 1.0 - true_positive_rate[index]) / 2.0
    )


def evaluate_predictions(y_true, probabilities, threshold: float) -> dict[str, object]:
    labels, scores = _validated_arrays(y_true, probabilities)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    predicted = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    two_classes = len(np.unique(labels)) == 2
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision_vulnerable": float(
            precision_score(labels, predicted, zero_division=0)
        ),
        "recall_vulnerable": float(recall_score(labels, predicted, zero_division=0)),
        "f1_vulnerable": float(f1_score(labels, predicted, zero_division=0)),
        "f1_macro": float(
            f1_score(labels, predicted, average="macro", zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(labels, scores)) if two_classes else float("nan"),
        "pr_auc": float(average_precision_score(labels, scores)),
        "eer": _eer(labels, scores),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else float("nan"),
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else float("nan"),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def benchmark_predict(predict: Callable[[], np.ndarray], sample_count: int):
    started = time.perf_counter()
    probabilities = predict()
    elapsed = time.perf_counter() - started
    return probabilities, {
        "seconds": elapsed,
        "samples_per_second": sample_count / elapsed if elapsed else float("inf"),
        "milliseconds_per_sample": elapsed * 1000 / sample_count if sample_count else 0.0,
    }
