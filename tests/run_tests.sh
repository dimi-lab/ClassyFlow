#!/usr/bin/env bash
# Simple test runner for ClassyFlow.
#
#   ./run_tests.sh          # set up venv (if needed) and run the fast unit suite
#   ./run_tests.sh e2e      # also run the end-to-end Nextflow smoke test
#   ./run_tests.sh all      # run everything (unit + e2e)
#
# The venv is built from requirements.txt (production deps + pytest).
set -euo pipefail

# Script lives in tests/ but operates from the repo root (requirements.txt,
# .venv, and pytest.ini are there).
cd "$(dirname "$0")/.."
VENV=".venv"
# Interpreter used to build the venv. Override for a specific version, e.g.:
#   PY=python3.11 ./run_tests.sh
PY="${PY:-python3}"

if [[ ! -d "$VENV" ]]; then
    # scimap/umap-learn -> numba, which supports only Python <3.12. Warn early.
    pyver="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    case "$pyver" in
        3.8|3.9|3.10|3.11) : ;;
        *)
            echo "[run_tests] WARNING: $PY is Python $pyver. The full requirements.txt"
            echo "            (scimap/umap-learn -> numba) only installs on Python 3.8-3.11."
            echo "            Use the container, or: PY=python3.11 ./run_tests.sh"
            ;;
    esac
    echo "[run_tests] creating virtualenv in $VENV (using $PY) ..."
    "$PY" -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install -r requirements.txt
fi

# Activate so bin/ scripts run under the venv interpreter (needed by the E2E run).
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Config lives in tests/, so point pytest at it explicitly (we run from root).
CFG="-c tests/pytest.ini"

case "${1:-unit}" in
    unit|"")
        echo "[run_tests] running unit tests ..."
        pytest $CFG -m "not e2e"
        ;;
    e2e)
        echo "[run_tests] running E2E smoke test ..."
        pytest $CFG -m e2e
        ;;
    all)
        echo "[run_tests] running full suite (unit + e2e) ..."
        # Override the default `-m "not e2e"` from the ini to run everything.
        pytest $CFG -o addopts="-ra"
        ;;
    *)
        echo "usage: $0 [unit|e2e|all]" >&2
        exit 2
        ;;
esac
