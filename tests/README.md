# ClassyFlow tests

A small, two-layer test suite:

- **`unit/`** — fast pytest tests on the pure/deterministic helper functions in
  `bin/`. No Nextflow needed. Run in seconds.
- **`e2e/`** — one end-to-end smoke test that runs the whole pipeline on the
  bundled `data/` via Nextflow and asserts the key outputs exist. Opt-in.

## Running

```bash
./tests/run_tests.sh          # set up .venv (if needed) + run unit tests
./tests/run_tests.sh e2e      # run only the end-to-end smoke test
./tests/run_tests.sh all      # run everything
```

(The script can be run from anywhere; it operates from the repo root.)

Or directly, inside an activated venv. The pytest config lives in
`tests/pytest.ini`, so point pytest at it (or run from inside `tests/`):

```bash
pytest -c tests/pytest.ini            # unit only (default)
pytest -c tests/pytest.ini -m e2e     # end-to-end smoke test
# equivalently:  (cd tests && pytest)
```

## Environment

The venv is built from `requirements.txt` (a single file: pipeline deps + pytest).

**Python version matters:** the full dependency set (`scimap`, `umap-learn` →
`numba`) only installs on **Python 3.8–3.11** — the same 3.11 used by the
container (`container/Dockerfile`). On Python 3.12 the install fails.

- Build with a compatible interpreter: `PY=python3.11 ./tests/run_tests.sh`
- Or run the suite inside the `classyflow` container.

The **unit tests** avoid the numba stack (the two `scimap` tests skip
automatically if `scanpy` is missing), so they can run under a lighter env if
needed. The **E2E test** requires the full environment and is skipped if
`nextflow` is not on PATH.

## The E2E micro dataset

The E2E smoke test does **not** run on the full bundled `data/`. It runs on a
tiny subsample under `tests/e2e/micro_data/` (a few hundred cells across the two
batches) so the whole Nextflow pipeline finishes in a couple of minutes.
`conf/test.config` points `input_dirs` at it.

Regenerate it deterministically from `data/` with:

```bash
python tests/e2e/make_micro_dataset.py
```

The generator keeps the original file layout, headers and column order, and
enough labeled cells per cell type to satisfy the training-split thresholds
(`minimum_label_count`, holdout fraction). See the script's docstring for the
sizing rationale.
