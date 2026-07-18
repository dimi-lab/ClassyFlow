"""End-to-end smoke test: run the whole pipeline on the bundled sample data.

This is the release-freeze regression check. It runs Nextflow with the small
`conf/test.config` against `data/` and asserts the pipeline completes and produces
its key published outputs.

Requirements to actually execute:
  * `nextflow` on PATH (test is skipped otherwise).
  * The full requirements.txt environment on PATH (Python 3.10/3.11 — the
    container). The default report stage imports umap (numba), which does not
    install on Python 3.12, so this test cannot run under a 3.12 interpreter.

Run it explicitly with:  pytest -m e2e     (or ./run_tests.sh e2e)
"""
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow not installed")
def test_pipeline_runs_end_to_end(tmp_path):
    work_dir = tmp_path / "work"
    output_dir = tmp_path / "classyflow_test_output"

    cmd = [
        "nextflow", "run", str(REPO_ROOT / "main.nf"),
        "-profile", "local",
        "-c", str(REPO_ROOT / "conf" / "test.config"),
        "-work-dir", str(work_dir),
        "--output_dir", str(output_dir),
    ]
    # Run from tmp so .nextflow logs/caches don't pollute the repo.
    result = subprocess.run(
        cmd, cwd=tmp_path, capture_output=True, text=True, timeout=60 * 60,
    )

    assert result.returncode == 0, (
        f"Pipeline failed (exit {result.returncode}).\n"
        f"--- STDOUT ---\n{result.stdout[-4000:]}\n"
        f"--- STDERR ---\n{result.stderr[-4000:]}"
    )

    # Key published outputs must exist and be non-empty.
    def _nonempty(pattern: str):
        matches = [p for p in output_dir.rglob(pattern) if p.stat().st_size > 0]
        assert matches, f"expected a non-empty file matching '{pattern}' under {output_dir}"
        return matches

    _nonempty("classyflow_report.html")     # final assembled report
    _nonempty("celltypes.csv")              # train/holdout label list
    _nonempty("classes.npy")                # trained label encoder
    _nonempty("*.pkl")                      # a serialized model / dataframe
    _nonempty("*_qPRED.tsv")                # per-slide QC'd predictions
