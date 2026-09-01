"""Unit tests for bin/generate_light_report.py — the light report assembler.

These exercise the data collectors (pure-ish functions) and a full end-to-end
render against the real Jinja2 templates in ``assets/html_templates/``, using
small fixture artifacts written to a temp dir.
"""
import csv
import json
from pathlib import Path

import pandas as pd
import pytest

import generate_light_report as glr

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = REPO_ROOT / "assets" / "html_templates"


# --- Fixture builders ------------------------------------------------------

def _write_input_metrics(d: Path) -> Path:
    metrics = {
        "total_batches": 2, "total_rois": 4, "total_cells": 12000,
        "total_labeled_cells": 8000, "num_unique_labels": 3,
        "label_table": {
            "headers": ["Label", "Label Count", "Label Percent", "Class Balance", "Status"],
            "rows": [
                {"Label": "Tumor", "Label Count": 5000, "Label Percent": 62.5,
                 "Class Balance": "Dominant", "Status": "Included"},
                {"Label": "Bcell", "Label Count": 1000, "Label Percent": 12.5,
                 "Class Balance": "Underrepresented", "Status": "Included"},
            ],
        },
        "batches": [
            {"batch_id": "B1", "percent_synthetic": 0},
            {"batch_id": "B2", "percent_synthetic": 30},
        ],
    }
    p = d / "input_batch_metrics.json"
    p.write_text(json.dumps(metrics))
    return p


def _write_fs(d: Path, n_features=None):
    """Write feature_selection_<ct>_results.json for two cell types.

    ``feature_importance`` mirrors what rank_selected_features() emits upstream:
    one record per SELECTED FEATURE, descending importance, signed direction.
    Each marker contributes two features (Mean + StdDev) so the fixture also
    covers the many-features-per-marker case. Pass ``n_features`` to override
    the feature count (for truncation tests).
    """
    for ct, opt in [("Tumor", 6), ("Bcell", 4)]:
        n = n_features if n_features is not None else opt
        feats = []
        for i in range(n):
            marker = f"CD{i // 2}"            # two statistics per marker
            stat = "Mean" if i % 2 == 0 else "StdDev"
            feats.append({
                "feature": f"{marker}: Cell: {stat}",
                "marker": marker,
                "importance": float(n - i),
                "direction": "negative" if i % 3 == 1 else "positive",
            })
        fs = {
            "celltype": ct,
            "selected_features": [f["feature"] for f in feats],
            "selected_features_count": n,
            "feature_importance": feats,
            "marker_importance": [
                {"marker": m, "importance": max(f["importance"] for f in feats if f["marker"] == m),
                 "direction": next(f["direction"] for f in feats if f["marker"] == m)}
                for m in dict.fromkeys(f["marker"] for f in feats)
            ],
            "feature_selection_summary": {
                "original_features": 20, "non_variant_removed": 2,
                "reduction_percentage": round((1 - n / 20) * 100, 2),
            },
            "rfe_warning": False, "min_features_threshold": 3,
        }
        (d / f"feature_selection_{ct}_results.json").write_text(json.dumps(fs))


def _write_profile(d: Path) -> Path:
    profile = {
        "cell_types": [
            {"name": "Tumor", "markers": {"CD0": "positive", "CD1": "positive"}},
            {"name": "Immune", "markers": {"CD9": "positive"}},
            {"name": "Bcell", "parent": "Immune",
             "markers": {"CD9": "positive", "CD5": "positive"}},
            {"name": "Myeloid", "parent": "Immune",
             "markers": {"CD9": "positive", "CD0": "positive"}},
        ]
    }
    p = d / "celltype_profile.yaml"
    import yaml as _yaml
    p.write_text(_yaml.safe_dump(profile))
    return p


def _write_concordance(d: Path) -> Path:
    p = d / "feature_concordance.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "cf_class", "self_cell_type", "self_score", "best_match", "best_score",
            "relation", "shared_markers", "conflicting_markers", "note"])
        w.writeheader()
        w.writerow({"cf_class": "Tumor", "self_cell_type": "Tumor", "self_score": 0.9,
                    "best_match": "Tumor", "best_score": 0.9, "relation": "self",
                    "shared_markers": 4, "conflicting_markers": "", "note": ""})
        w.writerow({"cf_class": "Bcell", "self_cell_type": "Bcell", "self_score": 0.3,
                    "best_match": "Myeloid", "best_score": 0.5, "relation": "sibling",
                    "shared_markers": 2, "conflicting_markers": "CD3|CD8", "note": ""})
    return p


def _write_holdout(d: Path):
    for mdl, acc in [("First", 0.92), ("Second", 0.88)]:
        cm = f"holdoutEval_XGBoost_Model_{mdl}_confusion_matrix.html"
        roc = f"holdoutEval_XGBoost_Model_{mdl}_roc_curves.html"
        pr = f"holdoutEval_XGBoost_Model_{mdl}_pr_curves.html"
        for name, label in [(cm, "CONFUSION"), (roc, "ROC"), (pr, "PR")]:
            (d / name).write_text(f'<div class="plotly-graph-div">{label} {mdl}</div>')
        he = {
            "accuracy": acc, "f1_score": acc - 0.02, "class_imbalance_detected": True,
            "n_classes": 3, "total_samples": 800,
            "class_names": ["Tumor", "Bcell", "Fibroblast"], "class_counts": [500, 100, 200],
            "max_auc": {"class_name": "Tumor", "auc": 0.97},
            "min_auc": {"class_name": "Bcell", "auc": 0.72},
            "auc_scores": [{"class_name": "Tumor", "auc": 0.97},
                           {"class_name": "Fibroblast", "auc": 0.85},
                           {"class_name": "Bcell", "auc": 0.72}],
            "ap_scores": [{"class_name": "Tumor", "ap": 0.95},
                          {"class_name": "Fibroblast", "ap": 0.80},
                          {"class_name": "Bcell", "ap": 0.60}],
            "confusion_matrix_html_path": cm,
            "roc_curves_plot_path": roc,
            "pr_curves_plot_path": pr,
        }
        (d / f"holdoutEval_XGBoost_Model_{mdl}_results.json").write_text(json.dumps(he))


def _write_counts(d: Path) -> Path:
    rows = []
    for s, batch in [("C-3", "B1"), ("D-16", "B1"), ("D-22", "B2"), ("H-17", "B2")]:
        for ct, c in [("Tumor", 1800), ("Fibroblast", 800), ("Bcell", 400)]:
            rows.append({"sample_name": s, "batch": batch, "cell_type": ct, "count": c,
                         "total_cells": 3000, "low_density_cells": 50,
                         "roi_report": f"{s}_prediction_report.html"})
    p = d / "all_cell_counts.tsv"
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    return p


# --- Collector-level tests -------------------------------------------------

def test_build_feature_section_no_profile_still_charts_features(tmp_path):
    # Regression: the bar chart is built from the FS JSON's feature_importance,
    # so it must render with no cell-type profile and no concordance file.
    _write_fs(tmp_path)
    fs = glr.build_feature_section(str(tmp_path), None, None, None)["feature_selection"]
    assert fs["has_definitions"] is False
    assert fs["tree"] is None
    assert fs["overview"] is None
    names = {c["celltype"] for c in fs["celltypes"]}
    assert names == {"Tumor", "Bcell"}

    tumor = next(c for c in fs["celltypes"] if c["celltype"] == "Tumor")
    assert len(tumor["feature_bars"]) == 6
    assert tumor["n_features_total"] == 6 and tumor["n_features_shown"] == 6
    # Direction survives without a profile — that was the bug.
    assert {b["direction"] for b in tumor["feature_bars"]} == {"positive", "negative"}
    # No definition columns when no definitions were supplied.
    assert "relation_class" not in tumor and "verdict" not in tumor


def test_feature_bars_are_one_per_feature_not_per_marker(tmp_path):
    # The fixture gives each marker two statistics; every one gets its own bar.
    _write_fs(tmp_path)
    fs = glr.build_feature_section(str(tmp_path), None, None, None)["feature_selection"]
    tumor = next(c for c in fs["celltypes"] if c["celltype"] == "Tumor")
    bars = tumor["feature_bars"]
    assert [b["feature"] for b in bars] == [
        "CD0: Cell: Mean", "CD0: Cell: StdDev",
        "CD1: Cell: Mean", "CD1: Cell: StdDev",
        "CD2: Cell: Mean", "CD2: Cell: StdDev",
    ]
    # 6 bars spanning only 3 markers — collapsing would have shown 3.
    assert len(bars) == 6
    assert len({b["marker"] for b in bars}) == 3


def test_feature_bars_reuse_upstream_order_and_scale(tmp_path):
    _write_fs(tmp_path)
    fs = glr.build_feature_section(str(tmp_path), None, None, None)["feature_selection"]
    tumor = next(c for c in fs["celltypes"] if c["celltype"] == "Tumor")
    bars = tumor["feature_bars"]
    # The most important feature is full height/saturation.
    assert bars[0]["height_pct"] == 100.0 and bars[0]["intensity"] == 1.0
    # Bar heights fall off monotonically with importance.
    assert [b["height_pct"] for b in bars] == sorted(
        (b["height_pct"] for b in bars), reverse=True)
    # Weak bars keep a visible floor rather than collapsing to zero.
    assert bars[-1]["height_pct"] >= 8.0
    # Hue follows direction: index 1 is negative in the fixture.
    assert bars[1]["direction"] == "negative"
    assert bars[1]["bg"] != bars[0]["bg"]


def test_feature_bars_truncates_at_limit(tmp_path):
    _write_fs(tmp_path, n_features=20)
    fs = glr.build_feature_section(str(tmp_path), None, None, None)["feature_selection"]
    tumor = next(c for c in fs["celltypes"] if c["celltype"] == "Tumor")
    assert len(tumor["feature_bars"]) == glr.MAX_FEATURES_SHOWN == 15
    assert tumor["n_features_total"] == 20
    assert tumor["n_features_shown"] == 15
    # Truncation keeps the most important features.
    assert [b["feature"] for b in tumor["feature_bars"]][:2] == [
        "CD0: Cell: Mean", "CD0: Cell: StdDev"]


def test_feature_bars_empty_is_safe():
    assert glr._feature_bars([]) == []


def test_feature_bars_zero_importance_has_no_direction():
    # A selected feature with a zero coefficient is charted, greyed, and makes
    # no directional claim.
    bars = glr._feature_bars([
        {"feature": "CD3: Cell: Mean", "marker": "CD3", "importance": 0.8,
         "direction": "positive"},
        {"feature": "CD9: Cell: Mean", "marker": "CD9", "importance": 0.0,
         "direction": None},
    ])
    assert len(bars) == 2
    assert bars[1]["direction"] is None
    assert bars[1]["bg"] == glr._bar_color(0.0, None)
    assert bars[1]["height_pct"] == 8.0


def test_build_feature_section_with_definitions(tmp_path):
    _write_fs(tmp_path)
    conc = _write_concordance(tmp_path)
    prof = _write_profile(tmp_path)
    fs = glr.build_feature_section(str(tmp_path), None, str(conc), str(prof))["feature_selection"]
    assert fs["has_definitions"] is True
    assert fs["tree"] is not None

    tumor = next(c for c in fs["celltypes"] if c["celltype"] == "Tumor")
    # relation self -> green "match"; own def CD0+,CD1+ both selected.
    assert tumor["best_match"] == "Tumor"
    assert tumor["relation_class"] == "match"
    assert tumor["n_matched"] == 2 and tumor["n_expected"] == 2
    assert {m["marker"] for m in tumor["marker_matches"]} == {"CD0", "CD1"}
    assert all(m["status"] == "matched" for m in tumor["marker_matches"])
    assert tumor["n_conflict"] == 0 and tumor["n_missing"] == 0
    assert tumor["hierarchy_breadcrumb"] == ["Tumor"]
    assert "match the expected Tumor definition" in tumor["verdict"]
    # Chart is present in definition mode too.
    assert len(tumor["feature_bars"]) == 6

    bcell = next(c for c in fs["celltypes"] if c["celltype"] == "Bcell")
    # relation sibling -> yellow "related"; own def CD9+,CD5+ neither selected.
    assert bcell["best_match"] == "Myeloid"
    assert bcell["relation_class"] == "related"
    assert bcell["n_matched"] == 0 and bcell["n_missing"] == 2
    assert {m["status"] for m in bcell["marker_matches"]} == {"missing"}
    assert bcell["hierarchy_breadcrumb"] == ["Immune", "Myeloid"]
    assert "closely related cell type" in bcell["verdict"]

    assert fs["overview"] == {"match": 1, "related": 1, "mismatch": 0,
                              "undefined": 0, "total": 2}


def test_relation_class_undefined_is_not_mismatch():
    # A class with no definition in the profile is not a finding: it must not
    # be coloured the same as a genuine marker disagreement.
    assert glr._relation_class("no_self_def") == "undefined"
    assert glr._relation_class("no_match") == "undefined"
    assert glr._relation_class("") == "undefined"
    assert glr._relation_class("unrelated") == "mismatch"
    assert glr._relation_class("self") == "match"
    assert glr._relation_class("sibling") == "related"


def test_celltype_without_definition_reads_as_undefined(tmp_path):
    _write_fs(tmp_path)
    prof = _write_profile(tmp_path)
    # Concordance covers Tumor only; Bcell has no row at all.
    p = tmp_path / "feature_concordance.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "cf_class", "self_cell_type", "self_score", "best_match", "best_score",
            "relation", "shared_markers", "conflicting_markers", "note"])
        w.writeheader()
        w.writerow({"cf_class": "Tumor", "self_cell_type": "Tumor", "self_score": 0.9,
                    "best_match": "Tumor", "best_score": 0.9, "relation": "self",
                    "shared_markers": 2, "conflicting_markers": "", "note": ""})

    fs = glr.build_feature_section(str(tmp_path), None, str(p), str(prof))["feature_selection"]
    bcell = next(c for c in fs["celltypes"] if c["celltype"] == "Bcell")
    assert bcell["relation_class"] == "undefined"
    assert "No definition was provided" in bcell["verdict"]
    # Still charted, and counted as undefined rather than as a mismatch.
    assert len(bcell["feature_bars"]) == 4
    assert fs["overview"]["undefined"] == 1
    assert fs["overview"]["mismatch"] == 0


def test_bar_color_distinguishes_direction():
    # Same importance, opposite direction -> different hue (blue vs warm).
    pos = glr._bar_color(1.0, "positive")
    neg = glr._bar_color(1.0, "negative")
    assert pos == "#1e3a8a"      # blue ramp high end
    assert neg == "#9a3412"      # warm ramp high end
    assert pos != neg
    # No direction (zero coefficient) gets a neutral grey ramp rather than
    # borrowing the positive hue and implying a positive association.
    none = glr._bar_color(1.0, None)
    assert none == "#6b7280"
    assert none != pos and none != neg


def test_load_celltype_profile_tree(tmp_path):
    prof = _write_profile(tmp_path)
    profile = glr.load_celltype_profile(str(prof))
    root_names = {n["name"] for n in profile["roots"]}
    assert root_names == {"Tumor", "Immune"}
    immune = next(n for n in profile["roots"] if n["name"] == "Immune")
    child_names = {c["name"] for c in immune["children"]}
    assert child_names == {"Bcell", "Myeloid"}


def test_load_celltype_profile_optional(tmp_path):
    # celltype_profile is optional: a missing/None path degrades to an empty
    # profile rather than raising, so the report can render without one.
    assert glr.load_celltype_profile(None) == {}
    assert glr.load_celltype_profile("") == {}
    assert glr.load_celltype_profile(str(tmp_path / "does_not_exist.yaml")) == {}


def test_collect_model_performance_picks_best_model(tmp_path):
    _write_holdout(tmp_path)
    perf = glr.collect_model_performance(str(tmp_path), None)["model_performance"]
    assert perf is not None
    assert perf["model_name"] == "First"          # best model chosen, not Second
    assert perf["accuracy"] == 0.92
    # Class rows sorted by AUC descending.
    assert [c["name"] for c in perf["class_rows"]] == ["Tumor", "Fibroblast", "Bcell"]
    assert perf["class_rows"][0]["auc_class"] == "score-excellent"
    assert "CONFUSION First" in perf["confusion_html"]
    assert "ROC First" in perf["roc_curves_html"]


# --- End-to-end render test ------------------------------------------------

def _render(tmp_path, with_concordance: bool, n_features=None) -> str:
    im = _write_input_metrics(tmp_path)
    _write_fs(tmp_path, n_features=n_features)
    _write_holdout(tmp_path)
    counts = _write_counts(tmp_path)
    concordance_csv = str(_write_concordance(tmp_path)) if with_concordance else None
    profile_path = str(_write_profile(tmp_path)) if with_concordance else None

    all_data = {}
    all_data.update(glr.collect_input_metrics(str(im)))
    all_data.update(glr.build_feature_section(str(tmp_path), None, concordance_csv,
                                              profile_path))
    all_data.update(glr.collect_model_performance(str(tmp_path), None))
    all_data.update(glr.collect_predictions(str(counts), str(tmp_path)))

    class _Args:
        version = "9.9"
        input_dirs = "/data/A,/data/B"
        normalization = "boxcox"
        holdout_fraction = "0.1"
        min_label_count = "18"
        exclude_markers = "Bcl2|Ki67"
        config_file = "nextflow.config"
    all_data.update(glr.build_provenance(_Args()))

    env = glr.setup_jinja_environment(str(TEMPLATE_DIR))
    out = tmp_path / "classyflow_report_light.html"
    glr.generate_report(all_data, str(out), env, letterhead=None, pipeline_version="9.9")
    return out.read_text()


def test_full_render_has_all_sections_and_provenance(tmp_path):
    html = _render(tmp_path, with_concordance=True)

    # All four sections (+ provenance) present.
    for section_id in ("light-summary", "light-features", "light-modeling",
                       "light-predictions", "light-provenance"):
        assert f'id="{section_id}"' in html

    # Provenance strings rendered.
    assert "boxcox" in html
    assert "/data/A" in html
    assert "Bcl2" in html

    # Feature-selection profile mode: collapsible definitions + flat summary table.
    assert '<details class="fs-definitions"' in html
    assert '<ul class="profile-tree">' in html
    assert 'class="fs-summary-table"' in html
    assert '<div class="fs-overview">' in html
    assert 'class="mk-chart"' in html          # vertical selected-marker bars
    assert ">Selected Features<" in html
    # Diverging bars: positive markers point up, negative markers point down.
    assert 'class="mk-bar mk-bar-up"' in html
    assert 'class="mk-bar mk-bar-down"' in html
    # Combined best-match cell: relation label + expected-marker count + verdict.
    assert "def-cell def-match" in html
    assert "def-related" in html
    assert "Related lineage" in html
    assert "expected markers" in html
    assert 'class="def-verdict"' in html
    assert "match the expected Tumor definition" in html
    # Per-cell tabs and the embedded ranking PNG are gone.
    assert "showFeatureTab" not in html
    assert '<div class="fs-ranking-plot">' not in html
    # Provenance must not emit a link to a pages/ dir the light report never has.
    assert "pages/nextflow.config" not in html

    # Best model plots (First, not Second).
    assert "CONFUSION First" in html
    assert "CONFUSION Second" not in html
    assert "score-excellent" in html


def test_full_render_no_profile_mode_still_charts_features(tmp_path):
    import re
    html = _render(tmp_path, with_concordance=False)
    body = html.split("</style>", 1)[1]   # exclude the stylesheet from checks

    # The summary TABLE is still built, with one row per cell type and a
    # two-column layout (no definition column).
    table = re.search(r'<table class="fs-summary-table">.*?</table>', body, re.S)
    assert table, "feature-selection table missing without a profile"
    assert re.findall(r"<th>(.*?)</th>", table.group(0)) == [
        "Cell Type", "Selected Features"]
    tbody = re.search(r"<tbody>.*?</tbody>", table.group(0), re.S).group(0)
    assert len(re.findall(r"<tr[ >]", tbody)) == 2      # Tumor + Bcell
    assert 'class="col-mk-wide"' in body                # 2-col colgroup variant

    # Every row carries bars, and importance is encoded in bar height.
    assert body.count('class="mk-chart"') == 2
    assert body.count('class="mk-col"') == 10           # 6 Tumor + 4 Bcell
    assert 'class="mk-bar mk-bar-up"' in body
    assert 'class="mk-bar mk-bar-down"' in body
    heights = [float(h) for h in re.findall(r'mk-bar mk-bar-\w+" style="height:([\d.]+)%', body)]
    assert max(heights) == 100.0 and len(set(heights)) > 1, "bar heights must vary with importance"
    assert "relative importance" in body                # per-bar tooltip
    assert 'class="mk-scale"' in body                   # legend
    assert ">Selected Features<" in body
    # ...but no definition column, tree, or overview cards.
    assert '<ul class="profile-tree">' not in html
    assert "def-cell def-match" not in html
    assert ">Best Matching Definition<" not in html
    assert '<div class="fs-overview">' not in html
    # Retired chip rendering is gone for good.
    assert 'class="feature-chip"' not in html
    assert "9.9" in html
    # Model plots still embedded regardless of concordance.
    assert "CONFUSION First" in html


def test_feature_count_caption_only_when_truncated(tmp_path):
    # 6 features, limit 15 -> nothing truncated -> no caption.
    assert 'class="mk-caption"' not in _render(tmp_path, with_concordance=False)


def test_feature_count_caption_when_truncated(tmp_path):
    html = _render(tmp_path, with_concordance=False, n_features=20)
    assert 'class="mk-caption"' in html
    assert "15 of 20 selected features shown" in html
