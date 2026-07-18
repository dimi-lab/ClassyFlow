"""Unit tests for bin/compare_panel_designs.py marker/metrics logic."""
import json

import pandas as pd
import pytest

import compare_panel_designs as cpd

pytestmark = pytest.mark.unit


def _batch_df(batch_id, marker_cols, images, labels):
    data = {m: [1.0] * len(images) for m in marker_cols}
    data["Image"] = images
    data["Classification"] = labels
    data["original_batchID"] = [batch_id] * len(images)
    return pd.DataFrame(data)


def test_compare_headers_presence_matrix_and_metrics(tmp_path):
    b1 = _batch_df("b1", ["CD4: Mean", "CD8: Mean"],
                   images=["img1", "img1", "img1"],
                   labels=["Tumor", "Tumor", "Tumor"])
    b2 = _batch_df("b2", ["CD4: Mean", "PanCK: Mean"],
                   images=["img2", "img2", "img2"],
                   labels=["Tumor", "Fibroblast", "Fibroblast"])

    p1, p2 = tmp_path / "b1.pkl", tmp_path / "b2.pkl"
    b1.to_pickle(p1)
    b2.to_pickle(p2)

    panel_csv = tmp_path / "panel_design.csv"
    metrics_json = tmp_path / "metrics.json"
    cpd.compare_headers([str(p1), str(p2)], str(panel_csv), str(metrics_json),
                        threshold=3, class_column="Classification")

    # --- presence matrix ---
    panel = pd.read_csv(panel_csv, index_col=0)
    assert sorted(panel.index) == ["CD4", "CD8", "PanCK"]
    assert panel.loc["CD4", "b1"] == 1 and panel.loc["CD4", "b2"] == 1
    assert panel.loc["CD8", "b1"] == 1 and panel.loc["CD8", "b2"] == 0
    assert panel.loc["PanCK", "b1"] == 0 and panel.loc["PanCK", "b2"] == 1

    # --- metrics JSON ---
    metrics = json.loads(metrics_json.read_text())
    assert metrics["total_batches"] == 2
    assert metrics["all_markers"] == ["CD4", "CD8", "PanCK"]
    assert metrics["marker_status_matrix"]["CD8"]["b2"] == "synthetic"
    assert metrics["marker_status_matrix"]["CD4"]["b1"] == "real"

    rows = {r["Label"]: r for r in metrics["label_table"]["rows"]}
    # Tumor = 4/6 (>40%) -> Dominant, count>=threshold -> Included.
    assert rows["Tumor"]["Class Balance"] == "Dominant"
    assert rows["Tumor"]["Status"] == "Included"
    # Fibroblast = 2/6 (33%) -> Balanced, count<threshold -> Excluded.
    assert rows["Fibroblast"]["Class Balance"] == "Balanced"
    assert rows["Fibroblast"]["Status"] == "Excluded"
