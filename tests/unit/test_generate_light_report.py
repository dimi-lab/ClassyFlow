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


def _write_fs(d: Path):
    for ct, opt in [("Tumor", 5), ("Bcell", 4)]:
        fs = {
            "celltype": ct, "cv_folds": 12, "optimal_n_features": opt,
            "selected_features": [f"CD{i}: Mean" for i in range(opt)],
            "selected_features_count": opt,
            "feature_selection_summary": {
                "original_features": 20, "non_variant_removed": 2,
                "reduction_percentage": round((1 - opt / 20) * 100, 2),
            },
            "rfe_warning": False, "min_features_threshold": 3,
        }
        (d / f"feature_selection_{ct}_results.json").write_text(json.dumps(fs))


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

def test_collect_feature_selection_summary(tmp_path):
    _write_fs(tmp_path)
    data = glr.collect_feature_selection(str(tmp_path), None)["feature_selection"]
    assert data["summary"]["total_celltypes"] == 2
    assert data["summary"]["unique_original_features"] == 20
    # Tumor(5) + Bcell(4) share CD0..CD3, so 5 unique selected markers.
    assert data["summary"]["unique_selected_features"] == 5
    assert data["summary"]["cv_folds_summary"] == "12"


def test_collect_concordance_parses_relations(tmp_path):
    _write_concordance(tmp_path)
    conc = glr.collect_concordance(str(tmp_path / "feature_concordance.csv"))["concordance"]
    assert conc["available"] is True
    relations = {r["cf_class"]: r["relation"] for r in conc["rows"]}
    assert relations == {"Tumor": "self", "Bcell": "sibling"}
    bcell = next(r for r in conc["rows"] if r["cf_class"] == "Bcell")
    assert bcell["conflicting_markers"] == ["CD3", "CD8"]


def test_collect_concordance_absent():
    conc = glr.collect_concordance(None)["concordance"]
    assert conc["available"] is False
    assert conc["rows"] == []


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

def _render(tmp_path, with_concordance: bool) -> str:
    im = _write_input_metrics(tmp_path)
    _write_fs(tmp_path)
    _write_holdout(tmp_path)
    counts = _write_counts(tmp_path)
    concordance_csv = str(_write_concordance(tmp_path)) if with_concordance else None

    all_data = {}
    all_data.update(glr.collect_input_metrics(str(im)))
    all_data.update(glr.collect_feature_selection(str(tmp_path), None))
    all_data.update(glr.collect_concordance(concordance_csv))
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
    assert "9.9" in html

    # Concordance relation badges rendered.
    assert 'badge-success">\n' in html or "badge-success" in html
    assert "self" in html
    assert "sibling" in html

    # Best model plots embedded (First, not Second).
    assert "CONFUSION First" in html
    assert "CONFUSION Second" not in html

    # AUC score coloring class applied.
    assert "score-excellent" in html


def test_render_without_concordance_shows_note(tmp_path):
    html = _render(tmp_path, with_concordance=False)
    assert "Concordance scoring was not run" in html
