#!/usr/bin/env bash
# Runs the 5-fold CV (Report Section 4) one fold per subprocess so the OS
# fully reclaims memory between folds -- see run_one_fold.py's docstring for
# why (clear_session()+gc.collect() alone weren't enough to avoid an
# OOM-kill around fold 3 in the original run). Safe to re-run: run_one_fold.py
# appends to results_kfold_partial.json, so an interrupted run resumes from
# the next fold instead of restarting.
set -e

for i in 0 1 2 3 4; do
    python run_one_fold.py "$i"
done

python summarize_kfold.py results_kfold_partial.json results_kfold.json
