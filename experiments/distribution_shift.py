"""Robustness to class-distribution shift, computed from saved probabilities.

Two questions, no retraining needed:
1. Prevalence sweep: how do metrics move when the target positive rate
   changes (resampled from the DAppSCAN pool)?
2. Output adjustment: can the decision rule be re-fitted to the target
   distribution without target labels? Strategies:
   - source_recall / source_eer: thresholds chosen on source validation (status quo)
   - platt: logistic calibration fitted on source validation, decide at 0.5
   - quantile_match: keep the source validation predicted-positive rate on the target
   - em_prior: Saerens et al. (2002) EM estimate of the target prior, decide at 0.5
   - oracle_prior: predicted-positive rate = true target prevalence
     (reference only: needs the prevalence, never per-sample labels)
Ranking metrics (ROC-AUC) are invariant to all of these by construction.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.metrics import select_eer_threshold, select_recall_threshold


DEFAULT_PREVALENCES = (0.1, 0.2, 1 / 3, 0.5, 2 / 3)
STRATEGIES = (
    "source_recall", "source_eer", "platt", "quantile_match", "em_prior", "oracle_prior",
)
_EPS = 1e-7


def _logit(probabilities) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), _EPS, 1 - _EPS)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def fit_platt(labels, probabilities) -> LogisticRegression:
    return LogisticRegression(C=1e6).fit(_logit(probabilities), np.asarray(labels))


def apply_platt(model: LogisticRegression, probabilities) -> np.ndarray:
    return model.predict_proba(_logit(probabilities))[:, 1]


def em_prior_adjust(
    calibrated, source_prior: float, max_iter: int = 200, tol: float = 1e-6
) -> tuple[np.ndarray, float]:
    """EM re-estimation of the target positive prior (Saerens et al., 2002).

    `calibrated` must be posteriors calibrated to `source_prior`.
    """
    if not 0.0 < source_prior < 1.0:
        raise ValueError("source_prior must be in (0, 1)")
    posterior = np.clip(np.asarray(calibrated, dtype=np.float64), _EPS, 1 - _EPS)
    prior = source_prior
    adjusted = posterior
    for _ in range(max_iter):
        positive = posterior * prior / source_prior
        negative = (1 - posterior) * (1 - prior) / (1 - source_prior)
        adjusted = positive / (positive + negative)
        updated = float(np.clip(adjusted.mean(), _EPS, 1 - _EPS))
        if abs(updated - prior) < tol:
            prior = updated
            break
        prior = updated
    return adjusted, prior


def rate_matched_threshold(scores, positive_rate: float) -> float:
    """Threshold flagging (about) the top `positive_rate` share of `scores`."""
    values = np.sort(np.asarray(scores, dtype=np.float64))[::-1]
    count = int(round(np.clip(positive_rate, 0.0, 1.0) * len(values)))
    if count <= 0:
        return float(np.nextafter(values[0], np.inf))
    return float(values[min(count, len(values)) - 1])


def resample_to_prevalence(labels, prevalence: float, rng: np.random.Generator) -> np.ndarray:
    """Largest subset of the pool (without replacement) at the given positive rate."""
    labels = np.asarray(labels)
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    if not 0.0 < prevalence < 1.0:
        raise ValueError("prevalence must be in (0, 1)")
    n_pos = min(len(positives), int(prevalence / (1 - prevalence) * len(negatives)))
    n_neg = min(len(negatives), int(round(n_pos * (1 - prevalence) / prevalence)))
    if n_pos < 1 or n_neg < 1:
        raise ValueError("pool cannot reach the requested prevalence")
    chosen = np.concatenate([
        rng.choice(positives, n_pos, replace=False),
        rng.choice(negatives, n_neg, replace=False),
    ])
    return np.sort(chosen)


def _decision_metrics(labels: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    tp = int(((predicted == 1) & (labels == 1)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())
    tn = int(((predicted == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "balanced_accuracy": (recall + 1 - fpr) / 2,
        "predicted_positive_rate": float(predicted.mean()),
    }


def evaluate_strategies(
    val_labels, val_probabilities, target_labels, target_probabilities, min_recall: float = 0.95
) -> dict[str, dict[str, float]]:
    """Every strategy only reads target *scores*; target labels are used for scoring."""
    val_labels = np.asarray(val_labels)
    val_probabilities = np.asarray(val_probabilities, dtype=np.float64)
    labels = np.asarray(target_labels)
    scores = np.asarray(target_probabilities, dtype=np.float64)

    recall_threshold = select_recall_threshold(val_labels, val_probabilities, min_recall)
    eer_threshold = select_eer_threshold(val_labels, val_probabilities)
    platt = fit_platt(val_labels, val_probabilities)
    calibrated = apply_platt(platt, scores)
    source_prior = float(val_labels.mean())
    em_adjusted, em_prior = em_prior_adjust(calibrated, source_prior)
    source_rate = float((val_probabilities >= eer_threshold).mean())

    decisions = {
        "source_recall": scores >= recall_threshold,
        "source_eer": scores >= eer_threshold,
        "platt": calibrated >= 0.5,
        "quantile_match": scores >= rate_matched_threshold(scores, source_rate),
        "em_prior": em_adjusted >= 0.5,
        "oracle_prior": scores >= rate_matched_threshold(scores, float(labels.mean())),
    }
    results = {
        name: _decision_metrics(labels, predicted.astype(np.int64))
        for name, predicted in decisions.items()
    }
    shared = {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "true_prevalence": float(labels.mean()),
        "em_estimated_prevalence": em_prior,
        "brier_raw": float(np.mean((scores - labels) ** 2)),
        "brier_platt": float(np.mean((calibrated - labels) ** 2)),
        "brier_em": float(np.mean((em_adjusted - labels) ** 2)),
        "mean_score": float(scores.mean()),
    }
    for metrics in results.values():
        metrics.update(shared)
    return results


def prevalence_sweep(
    run: dict, test_name: str, prevalences=DEFAULT_PREVALENCES, draws: int = 20, seed: int = 0
) -> dict[str, dict[str, dict[str, float]]]:
    """Mean metrics per prevalence and strategy over `draws` random subsets."""
    evaluation = run["evaluations"][test_name]
    labels = np.asarray(evaluation["labels"])
    scores = np.asarray(evaluation["probabilities"], dtype=np.float64)
    val = run["validation"]
    rng = np.random.default_rng(seed)
    sweep = {}
    for prevalence in prevalences:
        accumulated: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for _ in range(draws):
            index = resample_to_prevalence(labels, prevalence, rng)
            for strategy, metrics in evaluate_strategies(
                val["labels"], val["probabilities"], labels[index], scores[index]
            ).items():
                for key, value in metrics.items():
                    accumulated[strategy][key].append(value)
        sweep[f"{prevalence:.3f}"] = {
            strategy: {key: float(np.mean(values)) for key, values in metrics.items()}
            for strategy, metrics in accumulated.items()
        }
    return sweep


def fold_result_as_run(result: dict, seed: int) -> dict:
    """Adapt an in-distribution fold result (experiments/runner.py) to the
    cross-dataset run shape, so the same sweep can be run on source-domain
    test folds as a contrast."""
    def block(items):
        return {
            "labels": [item["label"] for item in items],
            "probabilities": [item["probability"] for item in items],
        }

    return {
        "variant": result["variant"],
        "seed": seed,
        "validation": block(result["validation_predictions"]),
        "evaluations": {"fold_test": block(result["predictions"])},
    }


def summarize(results_file: Path | None, test_name: str, draws: int, runs: list[dict] | None = None) -> dict:
    if runs is None:
        if results_file is None:
            raise ValueError("either results_file or runs must be provided")
        runs = json.loads(results_file.read_text(encoding="utf-8"))
    per_variant: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        if "validation" not in run or test_name not in run.get("evaluations", {}):
            continue
        per_variant[run["variant"]].append(prevalence_sweep(run, test_name, draws=draws, seed=run["seed"]))
    if not per_variant:
        raise ValueError(f"no runs with validation predictions and test set {test_name!r}")

    summary = {}
    for variant, sweeps in per_variant.items():
        summary[variant] = {}
        for prevalence in sweeps[0]:
            summary[variant][prevalence] = {}
            for strategy in STRATEGIES:
                keys = sweeps[0][prevalence][strategy]
                summary[variant][prevalence][strategy] = {
                    key: {
                        "mean": float(np.mean([s[prevalence][strategy][key] for s in sweeps])),
                        "std": float(np.std([s[prevalence][strategy][key] for s in sweeps])),
                    }
                    for key in keys
                }
        summary[variant]["seeds"] = len(sweeps)
    return summary


def markdown_section(summary: dict, label: str, variant: str) -> str:
    by_prevalence = summary[variant]
    header_cols = " | ".join(STRATEGIES)
    lines = [
        f"## {label} — {variant} ({by_prevalence['seeds']} runs)",
        "",
        f"| Prevalence | ROC-AUC | EM-estimated prior | {header_cols} | all-positive F1 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for prevalence, strategies in by_prevalence.items():
        if prevalence == "seeds":
            continue
        p_val = float(prevalence)
        any_metrics = strategies["source_eer"]
        cells = [
            f"{m['f1']['mean']*100:.1f} / {m['fpr']['mean']*100:.1f} / {m['balanced_accuracy']['mean']*100:.1f}"
            for m in (strategies[s] for s in STRATEGIES)
        ]
        all_pos_f1 = 200.0 * p_val / (1.0 + p_val)
        lines.append(
            f"| {round(p_val * 100)}% | {any_metrics['roc_auc']['mean']*100:.1f} | "
            f"{any_metrics['em_estimated_prevalence']['mean']*100:.1f} | "
            + " | ".join(cells)
            + f" | {all_pos_f1:.1f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results", type=Path, help="cross_dataset_results.json")
    source.add_argument(
        "--fold-results", type=Path, nargs="+",
        help="in-distribution fold result JSONs (with validation_predictions)",
    )
    parser.add_argument("--test-name", default="dappscan_pool")
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.fold_results:
        runs = [
            fold_result_as_run(json.loads(path.read_text(encoding="utf-8")), seed=index)
            for index, path in enumerate(args.fold_results)
        ]
        summary = summarize(None, "fold_test", args.draws, runs)
        if not args.output:
            parser.error("--output is required with --fold-results")
        output = args.output
    else:
        summary = summarize(args.results, args.test_name, args.draws)
        output = args.output or args.results.with_name("distribution_shift.json")
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for variant, by_prevalence in summary.items():
        print(
            f"\n=== {variant} ({by_prevalence['seeds']} runs) — F1 / FPR / balanced accuracy; "
            "all-positive F1 = 2p/(1+p) is the no-skill reference ==="
        )
        print("prev   " + " | ".join(f"{s:>22}" for s in STRATEGIES))
        for prevalence, strategies in by_prevalence.items():
            if prevalence == "seeds":
                continue
            cells = [
                f"{m['f1']['mean']*100:5.1f}/{m['fpr']['mean']*100:5.1f}/{m['balanced_accuracy']['mean']*100:5.1f}"
                for m in (strategies[s] for s in STRATEGIES)
            ]
            any_metrics = strategies["source_eer"]
            print(
                f"{float(prevalence):.2f}   " + " | ".join(f"{c:>22}" for c in cells)
                + f" | all-positive F1 {200 * float(prevalence) / (1 + float(prevalence)):.1f}"
                + f" | ROC-AUC {any_metrics['roc_auc']['mean']*100:.1f} EM-prior {any_metrics['em_estimated_prevalence']['mean']*100:.1f}"
            )
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
