"""Small pure-function guards across a few scripts."""
import pytest

pytestmark = pytest.mark.unit


# --- bin/search_all_alphas.py : format_floats -----------------------------
def test_format_floats_formats_range():
    import search_all_alphas as saa
    assert saa.format_floats([0.1, 0.3, 1.0]) == "0.1-1.0"
    assert saa.format_floats([0.123456, 9.87654]) == "0.1235-9.8765"


def test_format_floats_requires_two_elements():
    import search_all_alphas as saa
    with pytest.raises(ValueError, match="at least two"):
        saa.format_floats([0.5])


# --- bin/get_holdout_evaluation.py : detect_class_imbalance ----------------
def test_detect_class_imbalance_balanced_is_false():
    import get_holdout_evaluation as ghe
    assert ghe.detect_class_imbalance([100, 100, 100], threshold=0.1) is False


def test_detect_class_imbalance_small_minority_is_true():
    import get_holdout_evaluation as ghe
    assert ghe.detect_class_imbalance([100, 5], threshold=0.1) is True


def test_detect_class_imbalance_ratio_over_10_is_true():
    import get_holdout_evaluation as ghe
    # min ratio 8/200=0.04 > 0.01 threshold, but 100/8 = 12.5 : 1 > 10 -> imbalanced.
    assert ghe.detect_class_imbalance([100, 8, 92], threshold=0.01) is True


# --- bin/generate_cell_type_selection.py : feature/marker ranking -----------
# rank_selected_features() is the single source of truth for feature-level
# importance + direction (light report bar chart); collapse_to_markers() rolls
# that same list up per marker for concordance scoring.
def _dfF(rows):
    import pandas as pd
    return pd.DataFrame(rows, columns=["Name", "Feature_Importance", "Coefficient"])


def test_rank_selected_features_keeps_every_selected_feature():
    import generate_cell_type_selection as gcts
    out = gcts.rank_selected_features(_dfF([
        ("CD45: Cell: Mean",   0.9,  0.9),
        ("CD45: Cell: StdDev", 0.6, -0.6),   # same marker, own bar
        ("CD3: Cell: Mean",    0.5, -0.5),
    ]), 3)
    assert [r["feature"] for r in out] == [
        "CD45: Cell: Mean", "CD45: Cell: StdDev", "CD3: Cell: Mean"]
    # Marker token is carried alongside, but features are NOT collapsed.
    assert [r["marker"] for r in out] == ["CD45", "CD45", "CD3"]
    assert [r["direction"] for r in out] == ["positive", "negative", "negative"]


def test_rank_selected_features_respects_cutoff():
    import generate_cell_type_selection as gcts
    rows = [(f"CD{i}: Cell: Mean", 1.0 - i / 10, 1.0 - i / 10) for i in range(6)]
    out = gcts.rank_selected_features(_dfF(rows), 3)
    assert [r["feature"] for r in out] == [
        "CD0: Cell: Mean", "CD1: Cell: Mean", "CD2: Cell: Mean"]
    assert gcts.rank_selected_features(_dfF(rows), 0) == []


def test_rank_selected_features_keeps_zero_coefficients_without_direction():
    # Regression: zero-coefficient features were being dropped (first by an
    # accidental `0.0 > 0.0` comparison, then by an explicit filter). They WERE
    # selected -- featureCutoff comes from the RFE curve, independent of the
    # Lasso -- so they must appear, with direction None since a zero coefficient
    # has no sign to report.
    import generate_cell_type_selection as gcts
    out = gcts.rank_selected_features(_dfF([
        ("CD45: Cell: Mean", 0.9, 0.9),
        ("CD20: Cell: Mean", 0.0, 0.0),
    ]), 2)
    assert [r["feature"] for r in out] == ["CD45: Cell: Mean", "CD20: Cell: Mean"]
    assert out[1]["importance"] == 0.0
    assert out[1]["direction"] is None, "a zero coefficient must not claim a direction"


def test_rank_selected_features_all_zero_still_reports_every_feature():
    import generate_cell_type_selection as gcts
    out = gcts.rank_selected_features(_dfF([
        ("CD3: Cell: Mean", 0.0, 0.0),
        ("CD4: Cell: Mean", 0.0, 0.0),
    ]), 2)
    assert len(out) == 2
    assert all(r["direction"] is None for r in out)


def test_collapse_to_markers_keeps_strongest_feature_per_marker():
    import generate_cell_type_selection as gcts
    features = gcts.rank_selected_features(_dfF([
        ("CD45: Cell: Mean",   0.9,  0.9),
        ("CD3: Cell: Mean",    0.5, -0.5),
        ("CD45: Cell: StdDev", 0.4, -0.4),   # weaker CD45 feature -> must not win
    ]), 3)
    out = gcts.collapse_to_markers(features)
    assert [r["marker"] for r in out] == ["CD45", "CD3"]
    assert out[0] == {"marker": "CD45", "importance": 0.9, "direction": "positive"}
    assert out[1]["direction"] == "negative"


def test_collapse_to_markers_empty():
    import generate_cell_type_selection as gcts
    assert gcts.collapse_to_markers([]) == []
