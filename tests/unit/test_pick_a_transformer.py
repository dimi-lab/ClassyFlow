"""Unit tests for the pure helpers in bin/pick_a_transformer.py.

We test the deterministic, side-effect-free helpers. The full ``apply_*_transform``
functions also emit HTML/plots and are covered by the end-to-end smoke test.
"""
import numpy as np
import pandas as pd
import pytest

import pick_a_transformer as pt

pytestmark = pytest.mark.unit


def test_get_target_columns_matches_substring_and_dedups():
    df = pd.DataFrame(columns=["CD4: Mean", "CD8: Mean", "CD4: Max", "Image"])
    cols = pt.get_target_columns(df, "Mean")
    assert cols == ["CD4: Mean", "CD8: Mean"]


def test_get_target_columns_multiple_targets():
    df = pd.DataFrame(columns=["CD4: Mean", "CD4: Max", "Area"])
    cols = pt.get_target_columns(df, "Mean, Max")
    assert set(cols) == {"CD4: Mean", "CD4: Max"}


def test_organize_columns_by_groups():
    groups = pt.organize_columns_by_groups(["CD4: Mean", "CD8: Mean", "CD4: Max", "Area"])
    assert groups["Mean"] == ["CD4: Mean", "CD8: Mean"]
    assert groups["Max"] == ["CD4: Max"]
    assert groups["Other"] == ["Area"]


def test_clean_image_names_strips_extension():
    df = pd.DataFrame({"Image": ["S1.ome.tiff", "S2.ome.tiff"]})
    out = pt.clean_image_names(df)
    assert list(out["Image"]) == ["S1", "S2"]


def test_calculate_skewness_metrics_reports_reduction():
    rng = np.random.default_rng(0)
    skewed = np.concatenate([rng.exponential(1.0, 500), [50, 60, 70]])  # right-skewed
    df_orig = pd.DataFrame({"CD4: Mean": skewed})
    df_trans = pd.DataFrame({"CD4: Mean": np.log1p(skewed)})  # log reduces skew

    metrics = pt.calculate_skewness_metrics(df_orig, df_trans, ["CD4: Mean"])
    row = metrics.iloc[0]
    assert row["marker"] == "CD4: Mean"
    assert abs(row["skewness_after"]) < abs(row["skewness_before"])
    assert row["reduction_ratio"] > 1.0


def test_calculate_outlier_metrics_counts_3sigma():
    # 100 tight values + 1 extreme outlier before; none after (constant).
    vals = np.concatenate([np.zeros(100), [1000.0]])
    df_orig = pd.DataFrame({"CD4: Mean": vals})
    df_trans = pd.DataFrame({"CD4: Mean": np.ones(101)})  # constant -> std 0 -> 0 outliers

    metrics = pt.calculate_outlier_metrics(df_orig, df_trans, ["CD4: Mean"])
    row = metrics.iloc[0]
    assert row["outliers_before"] >= 1
    assert row["outliers_after"] == 0
