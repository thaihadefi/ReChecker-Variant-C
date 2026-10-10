#!/usr/bin/env bash
# Cross-dataset generalization grid (train 1671 [+SmartBugs], test DAppSCAN zero-shot).
# Usage: experiments/run_generalization_grid.sh <setting> [<setting> ...]
#        experiments/run_generalization_grid.sh --list
# Each setting writes EXPERIMENT/generalization/<setting>/cross_dataset_results.json
# and is resumable (finished variant/seed pairs are skipped).
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
SEEDS="${SEEDS:-42,1,2,3,4}"
VARIANTS="${VARIANTS:-b3,b0}"
# Cap TensorFlow thread pools so several grid processes can share the CPU
# without oversubscription (default pools are sized to every core).
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-3}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
TESTS="dappscan_ratio2=Dataset/dappscan_reentrancy.txt,dappscan_ratio1=Dataset/dappscan_reentrancy_ratio1.txt,dappscan_pool=Dataset/dappscan_reentrancy_pool.txt"
REG="--hidden-units 64 --dense-units 64 --dropout 0.5 --token-dropout 0.2"
SMARTBUGS="--extra-train Dataset/smartbugs_reentrancy.txt"

ALL="--dedup --abstract-ids --min-token-groups 5 $REG"
NAMES="baseline dedup dedup_abstract dedup_vocab dedup_reg dedup_all dedup_frac50 dedup_frac75 dedup_sb dedup_all_sb dedup_all_frac50 dedup_all_frac75"

# bash 3.2 (macOS default) has no associative arrays.
setting_args() {
  case "$1" in
    baseline) echo "" ;;
    dedup) echo "--dedup" ;;
    dedup_abstract) echo "--dedup --abstract-ids" ;;
    dedup_vocab) echo "--dedup --min-token-groups 5" ;;
    dedup_reg) echo "--dedup $REG" ;;
    dedup_all) echo "$ALL" ;;
    dedup_frac50) echo "--dedup --train-fraction 0.5" ;;
    dedup_frac75) echo "--dedup --train-fraction 0.75" ;;
    dedup_sb) echo "--dedup $SMARTBUGS" ;;
    dedup_all_sb) echo "$ALL $SMARTBUGS" ;;
    dedup_all_frac50) echo "$ALL --train-fraction 0.5" ;;
    dedup_all_frac75) echo "$ALL --train-fraction 0.75" ;;
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
  out="EXPERIMENT/generalization/$setting"
  mkdir -p "$out"
  # shellcheck disable=SC2086
  PYTHONPATH=. "$PYTHON" -u experiments/run_cross_dataset.py \
    --variants "$VARIANTS" --seeds "$SEEDS" --output-dir "$out" \
    --test-files "$TESTS" $args 2>&1 | grep --line-buffered -v "NodeDef mentions" | tee -a "$out/run.log"
done
