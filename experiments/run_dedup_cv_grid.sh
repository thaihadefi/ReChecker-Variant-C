#!/usr/bin/env bash
# 5-Fold Cross-Validation on the 185 Unique Contracts (--dedup) across W/C extraction strategies.
# Usage: experiments/run_dedup_cv_grid.sh <setting> [<setting> ...]
#        experiments/run_dedup_cv_grid.sh --list
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
VARIANT="${VARIANT:-all}"
SEEDS="${SEEDS:-}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-3}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"

REG="--hidden-units 64 --dense-units 64 --dropout 0.5 --token-dropout 0.2"
ALL_INTERVENTIONS="--abstract-ids --min-token-groups 5 $REG"
NAMES="dedup_first dedup_call_anchored dedup_call_flow dedup_all_first dedup_all_call_anchored dedup_all_call_flow"

setting_args() {
  case "$1" in
    dedup_first) echo "--dedup --extraction-strategy first" ;;
    dedup_call_anchored) echo "--dedup --extraction-strategy call_anchored" ;;
    dedup_call_flow) echo "--dedup --extraction-strategy call_flow" ;;
    dedup_all_first) echo "--dedup $ALL_INTERVENTIONS --extraction-strategy first" ;;
    dedup_all_call_anchored) echo "--dedup $ALL_INTERVENTIONS --extraction-strategy call_anchored" ;;
    dedup_all_call_flow) echo "--dedup $ALL_INTERVENTIONS --extraction-strategy call_flow" ;;
    *) return 1 ;;
  esac
}

if [[ "${1:-}" == "--list" ]]; then
  for name in $NAMES; do echo "$name: $(setting_args "$name")"; done
  exit 0
fi

for setting in "$@"; do
  if ! args="$(setting_args "$setting")"; then
    echo "unknown setting: $setting (see --list)" >&2
    exit 2
  fi
  out="EXPERIMENT/dedup-cv/$setting"
  mkdir -p "$out"
  seed_Args=""
  if [[ -n "$SEEDS" ]]; then
    seed_Args="--seeds $SEEDS"
  fi
  # shellcheck disable=SC2086
  PYTHONPATH=. "$PYTHON" -u main.py \
    --run-all --variant "$VARIANT" --run-dir "$out" \
    $seed_Args $args 2>&1 | grep --line-buffered -v "NodeDef mentions" | tee -a "$out/run.log"
done
