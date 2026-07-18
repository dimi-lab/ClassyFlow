"""Unit tests for bin/fixup_columns.py column-cleaning helpers."""
import pandas as pd
import pytest

import fixup_columns as fx

pytestmark = pytest.mark.unit


def test_load_column_map(tmp_path):
    yaml_path = tmp_path / "map.yaml"
    yaml_path.write_text(
        "remove_columns:\n  - Batch\nrename_columns:\n  Old: New\nmarker_map:\n  ASMA: aSMA\n"
    )
    remove, rename, marker = fx.load_column_map(str(yaml_path))
    assert remove == ["Batch"]
    assert rename == {"Old": "New"}
    assert marker == {"ASMA": "aSMA"}


def test_load_column_map_missing_sections_default_empty(tmp_path):
    yaml_path = tmp_path / "map.yaml"
    yaml_path.write_text("remove_columns:\n  - X\n")
    remove, rename, marker = fx.load_column_map(str(yaml_path))
    assert remove == ["X"]
    assert rename == {}
    assert marker == {}


def test_remove_columns_func_drops_only_present():
    df = pd.DataFrame({"Batch": [1], "Keep": [2]})
    out = fx.remove_columns_func(df, ["Batch", "NotThere"])
    assert list(out.columns) == ["Keep"]


def test_rename_columns_func_renames_only_present():
    df = pd.DataFrame({"Old": [1], "Other": [2]})
    out = fx.rename_columns_func(df, {"Old": "New", "Absent": "Nope"})
    assert list(out.columns) == ["New", "Other"]


def test_rename_marker_rewrites_prefix_before_colon():
    # marker_map replaces the "<old>:" prefix, preserving the metric suffix.
    df = pd.DataFrame({
        "ASMA: Mean": [1.0],
        "ASMA: Max": [2.0],
        "CD4: Mean": [3.0],   # untouched
    })
    out = fx.rename_marker(df, {"ASMA": "aSMA"})
    assert list(out.columns) == ["aSMA: Mean", "aSMA: Max", "CD4: Mean"]


def test_rename_marker_does_not_match_substring_without_colon():
    # "ASMA" should not rewrite "ASMAX: Mean" (only "<old>:" prefixes match).
    df = pd.DataFrame({"ASMAX: Mean": [1.0]})
    out = fx.rename_marker(df, {"ASMA": "aSMA"})
    assert list(out.columns) == ["ASMAX: Mean"]


def test_main_writes_fx_pickle_with_transformations(tmp_path):
    df = pd.DataFrame({
        "Batch": [1, 2],
        "ASMA: Mean": [10.0, 20.0],
        "CD4: Mean": [1.0, 2.0],
    })
    in_pkl = tmp_path / "merged_dataframe_b1.pkl"
    df.to_pickle(in_pkl)

    map_yaml = tmp_path / "map.yaml"
    map_yaml.write_text("remove_columns:\n  - Batch\nrename_columns: {}\nmarker_map:\n  ASMA: aSMA\n")

    fx.main(str(in_pkl), str(map_yaml))

    out_pkl = tmp_path / "merged_dataframe_b1_fx.pkl"
    assert out_pkl.exists()
    result = pd.read_pickle(out_pkl)
    assert "Batch" not in result.columns
    assert "aSMA: Mean" in result.columns
    assert "CD4: Mean" in result.columns
