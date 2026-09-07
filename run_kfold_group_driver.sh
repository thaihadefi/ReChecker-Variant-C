#!/usr/bin/env bash
# Runs the StratifiedGroupKFold generalization check (Report Section 6) one
# fold per subprocess for the same OOM-avoidance reason as
# run_kfold_driver.sh -- see run_one_fold_group.py's docstring. Safe to
# re-run: run_one_fold_group.py appends to results_kfold_group_partial.json,
# so an interrupted run resumes from the next fold instead of restarting.
set -e

for i in 0 1 2 3 4; do
    python run_one_fold_group.py "$i"
done

python summarize_kfold.py results_kfold_group_partial.json results_kfold_group.json
