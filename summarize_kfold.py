"""Aggregate a results_kfold*_partial.json (produced by running run_one_fold.py
or run_one_fold_group.py once per fold, see run_kfold_driver.sh /
run_kfold_group_driver.sh) into mean +/- std per metric -- the same format
run_kfold.py writes when it runs all 5 folds itself in one process.

Usage:
    python summarize_kfold.py results_kfold_partial.json results_kfold.json
    python summarize_kfold.py results_kfold_group_partial.json results_kfold_group.json
"""
import sys
import json
import numpy as np


def summarize(partial_path, out_path):
    with open(partial_path) as f:
        fold_results = json.load(f)

    n_folds = {k: len(v) for k, v in fold_results.items()}
    if len(set(n_folds.values())) != 1:
        raise ValueError(f"variants have unequal fold counts, refusing to summarize: {n_folds}")

    summary = {}
    for variant, folds in fold_results.items():
        keys = folds[0].keys()
        summary[variant] = {
            k: {"mean": float(np.mean([f[k] for f in folds])), "std": float(np.std([f[k] for f in folds]))}
            for k in keys
        }

    with open(out_path, "w") as f:
        json.dump({"folds": fold_results, "summary": summary}, f, indent=2, default=float)

    print(f"summarized {list(n_folds.values())[0]} folds -> {out_path}\n")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    partial_path = sys.argv[1] if len(sys.argv) > 1 else "results_kfold_partial.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "results_kfold.json"
    summarize(partial_path, out_path)
