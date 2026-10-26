"""Aggregate EXPERIMENT/generalization/<setting>/ runs into one table.

Per setting x variant x test set: 5-seed mean/std of threshold-free
(ROC-AUC, PR-AUC) and thresholded metrics, plus paired (same seed)
Wilcoxon signed-rank tests of ROC-AUC against a reference setting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

METRICS = (
    ("roc_auc", "metrics_under_recall_threshold"),
    ("pr_auc", "metrics_under_recall_threshold"),
    ("f1_vulnerable", "metrics_under_recall_threshold"),
    ("false_positive_rate", "metrics_under_recall_threshold"),
    ("f1_vulnerable", "metrics_under_eer_threshold"),
    ("f1_macro", "metrics_under_eer_threshold"),
)


def _paired(reference: dict[int, float], candidate: dict[int, float]) -> dict[str, object]:
    seeds = sorted(set(reference) & set(candidate))
    diffs = np.array([candidate[s] - reference[s] for s in seeds])
    result = {
        "seeds": len(seeds),
        "mean_diff": float(diffs.mean()) if len(diffs) else float("nan"),
        "wins": int((diffs > 0).sum()),
    }
    if len(diffs) >= 2 and np.any(diffs != 0):
        result["wilcoxon_p"] = float(wilcoxon(diffs).pvalue)
    return result


def summarize(root: Path, reference: str) -> dict:
    runs = {}
    for path in sorted(root.glob("*/cross_dataset_results.json")):
        runs[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))

    def auc_by_seed(setting, variant, test):
        return {
            r["seed"]: r["evaluations"][test]["metrics_under_recall_threshold"]["roc_auc"]
            for r in runs.get(setting, []) if r["variant"] == variant and test in r["evaluations"]
        }

    summary: dict = {}
    for setting, results in runs.items():
        summary[setting] = {}
        for variant in sorted({r["variant"] for r in results}):
            variant_runs = [r for r in results if r["variant"] == variant]
            summary[setting][variant] = {"seeds": len(variant_runs), "fit_size": variant_runs[0]["sizes"]["fit"]}
            test_names = sorted({t for r in variant_runs for t in r.get("evaluations", {})})
            for test in test_names:
                test_runs = [r for r in variant_runs if test in r.get("evaluations", {})]
                stats = {}
                for metric, block in METRICS:
                    values = [r["evaluations"][test][block][metric] for r in test_runs]
                    key = f"{metric}@{block.split('_')[2]}" if metric not in ("roc_auc", "pr_auc") else metric
                    stats[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
                stats["vs_reference"] = _paired(
                    auc_by_seed(reference, variant, test), auc_by_seed(setting, variant, test)
                )
                if variant == "b3":
                    stats["b3_vs_b0"] = _paired(
                        auc_by_seed(setting, "b0", test), auc_by_seed(setting, "b3", test)
                    )
                summary[setting][variant][test] = stats
    return summary


def markdown(summary: dict, test: str) -> str:
    lines = [
        f"Test set: {test}",
        "",
        "| Setting | Var | n | Fit | ROC-AUC | PR-AUC | F1@recall-thr | FPR@recall-thr | F1@EER-thr | F1-macro@EER | dAUC vs ref (wins, p) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for setting, variants in summary.items():
        for variant, stats in variants.items():
            if test not in stats:
                continue
            s = stats[test]

            def cell(key):
                return f"{s[key]['mean']*100:.2f} ± {s[key]['std']*100:.2f}"

            ref = s["vs_reference"]
            ref_cell = (
                f"{ref['mean_diff']*100:+.2f} ({ref['wins']}/{ref['seeds']}, p={ref.get('wilcoxon_p', float('nan')):.3f})"
                if ref["seeds"] else "-"
            )
            lines.append(
                f"| {setting} | {variant} | {stats['seeds']} | {stats['fit_size']} | {cell('roc_auc')} | {cell('pr_auc')} | "
                f"{cell('f1_vulnerable@recall')} | {cell('false_positive_rate@recall')} | {cell('f1_vulnerable@eer')} | "
                f"{cell('f1_macro@eer')} | {ref_cell} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("EXPERIMENT/generalization"))
    parser.add_argument("--reference", default="baseline")
    args = parser.parse_args()
    summary = summarize(args.root, args.reference)
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    tests = sorted({t for v in summary.values() for s in v.values() for t in s if t.startswith("dappscan")})
    text = "\n\n".join(markdown(summary, test) for test in tests)
    (args.root / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
