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
import json
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

    _nonempty("classyflow_report.html")       # final assembled (full) report
    light = _nonempty("classyflow_report_light.html")[0]
    _nonempty("celltypes.csv")                # train/holdout label list
    _nonempty("classes.npy")                # trained label encoder
    _nonempty("*.pkl")                      # a serialized model / dataframe
    _nonempty("*_qPRED.tsv")                # per-slide QC'd predictions

    # feature_importance flows from feature selection into the light report's
    # selected-feature bar chart. With a celltype_profile set, the definition
    # column is present too.
    light_html = light.read_text()
    assert 'class="mk-col"' in light_html, "selected-feature bars missing from light report"
    assert ">Best Matching Definition<" in light_html

    # Contract on every feature-selection results JSON: one record per selected
    # feature, importance-ordered, marker token consistent, and NO feature
    # silently dropped (zero-coefficient features are kept, with no direction).
    for jf in _nonempty("feature_selection_*_results.json"):
        data = json.loads(jf.read_text())
        feats = data.get("feature_importance")
        assert feats, f"{jf.name} has no feature_importance"
        assert [r["feature"] for r in feats] == data["selected_features"], (
            f"{jf.name}: feature_importance must cover exactly selected_features, in order")
        for rec in feats:
            assert rec["marker"] == rec["feature"].split(":")[0].strip()
            assert rec["direction"] in ("positive", "negative", None)
            if rec["importance"] == 0:
                assert rec["direction"] is None, (
                    f"{jf.name}: zero-coefficient feature claims a direction")
        imps = [r["importance"] for r in feats]
        assert imps == sorted(imps, reverse=True), f"{jf.name}: not importance-ordered"

        # marker_importance is the rollup the concordance scorer consumes.
        markers = data.get("marker_importance")
        assert markers, f"{jf.name} has no marker_importance"
        assert {r["marker"] for r in markers} == {r["marker"] for r in feats}


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow not installed")
def test_pipeline_runs_without_optional_inputs(tmp_path):
    """Both `marker_vocabulary` and `celltype_profile` are optional.

    Runs the micro pipeline with harmonization off (so `marker_vocabulary` is
    never consumed) and no `celltype_profile` (so concordance scoring is
    skipped), asserting the run still completes and produces its core outputs
    while the profile-only artifacts are absent.
    """
    work_dir = tmp_path / "work"
    output_dir = tmp_path / "classyflow_test_output"

    # Groovy config (not a --flag) so the boolean is a real `false`, not the
    # truthy string "false" that `--batch_correct_column_names false` would give.
    override = tmp_path / "no_optional_inputs.config"
    override.write_text(
        "params {\n"
        "    batch_correct_column_names = false\n"
        "    celltype_profile = null\n"
        "}\n"
    )

    cmd = [
        "nextflow", "run", str(REPO_ROOT / "main.nf"),
        "-profile", "local",
        "-c", str(REPO_ROOT / "conf" / "test.config"),
        "-c", str(override),
        "-work-dir", str(work_dir),
        "--output_dir", str(output_dir),
    ]
    result = subprocess.run(
        cmd, cwd=tmp_path, capture_output=True, text=True, timeout=60 * 60,
    )

    assert result.returncode == 0, (
        f"Pipeline failed with no optional inputs (exit {result.returncode}).\n"
        f"--- STDOUT ---\n{result.stdout[-4000:]}\n"
        f"--- STDERR ---\n{result.stderr[-4000:]}"
    )

    def _nonempty(pattern: str):
        matches = [p for p in output_dir.rglob(pattern) if p.stat().st_size > 0]
        assert matches, f"expected a non-empty file matching '{pattern}' under {output_dir}"
        return matches

    # Core outputs are still produced without either optional input.
    light = _nonempty("classyflow_report_light.html")[0]
    _nonempty("celltypes.csv")
    _nonempty("*_qPRED.tsv")

    # Concordance artifacts require a celltype_profile, so must be absent here.
    assert not list(output_dir.rglob("feature_concordance.*")), (
        "feature_concordance.* must not be produced when celltype_profile is unset"
    )

    # The selected-feature bar chart reads feature_importance straight from the
    # feature-selection JSON, so it must render with no profile at all — while
    # the definition column and profile tree stay absent.
    light_html = light.read_text()
    assert 'class="mk-col"' in light_html, (
        "selected-feature bars must render without a celltype_profile")
    assert 'class="mk-bar mk-bar-up"' in light_html
    assert ">Best Matching Definition<" not in light_html
    assert '<ul class="profile-tree">' not in light_html

