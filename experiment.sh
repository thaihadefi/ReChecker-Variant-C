#!/usr/bin/env bash
set -euo pipefail

# Preserve the historical positional run-directory argument while forwarding
# every other experiment option to the Python runner.
if [[ $# -gt 0 && "${1}" != -* ]]; then
    run_dir="${1}"
    shift
    exec python3 main.py --run-all --run-dir "${run_dir}" "$@"
fi

exec python3 main.py --run-all "$@"
