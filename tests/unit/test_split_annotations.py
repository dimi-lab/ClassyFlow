"""Unit tests for bin/split_annotations_for_training.py stratified_split (seeded)."""
import numpy as np
import pandas as pd
import pytest

import split_annotations_for_training as sa

pytestmark = pytest.mark.unit


def _make_df(n_per_class=50, classes=("Tumor", "Fibroblast"), batch="b1"):
    rows = []
    for c in classes:
        for _ in range(n_per_class):
            rows.append({"Batch": batch, "CellType": c, "feat": np.random.rand()})
    return pd.DataFrame(rows)


def test_stratified_split_labels_and_proportions():
    df = _make_df(n_per_class=100)
    split = sa.stratified_split(
        df, stratify_cols=["Batch", "CellType"], holdout_frac=0.2,
        min_count=10, min_count_col="CellType",
    )
    assert set(split.unique()) <= {"train", "holdout", "Not Used"}
    # ~20% holdout of 200 rows.
    assert split.value_counts().get("holdout", 0) == pytest.approx(40, abs=2)
    # Index aligns with input.
    assert list(split.index) == list(df.index)


def test_stratified_split_is_deterministic_seeded():
    df = _make_df(n_per_class=60)
    a = sa.stratified_split(df, ["CellType"], 0.25, 5, "CellType")
    b = sa.stratified_split(df, ["CellType"], 0.25, 5, "CellType")
    pd.testing.assert_series_equal(a, b)


def test_stratified_split_excludes_low_count_classes():
    # Two classes clear the threshold; the rare one does not and must be "Not Used".
    df = pd.concat([
        _make_df(n_per_class=50, classes=("Tumor", "Fibroblast")),
        _make_df(n_per_class=3, classes=("Rare",)),
    ], ignore_index=True)
    split = sa.stratified_split(df, ["CellType"], 0.2, min_count=10, min_count_col="CellType")
    rare_idx = df.index[df["CellType"] == "Rare"]
    assert (split.loc[rare_idx] == "Not Used").all()
    # Valid classes are used (train + holdout).
    valid_idx = df.index[df["CellType"] != "Rare"]
    assert set(split.loc[valid_idx].unique()) <= {"train", "holdout"}


@pytest.mark.parametrize("bad_frac", [0.0, 1.0, -0.1, 1.5])
def test_stratified_split_rejects_bad_holdout_fraction(bad_frac):
    df = _make_df(n_per_class=20)
    with pytest.raises(ValueError, match="holdout_frac"):
        sa.stratified_split(df, ["CellType"], bad_frac, 5, "CellType")


def test_stratified_split_rejects_bad_min_count():
    df = _make_df(n_per_class=20)
    with pytest.raises(ValueError, match="min_count"):
        sa.stratified_split(df, ["CellType"], 0.2, 0, "CellType")


def test_stratified_split_rejects_missing_columns():
    df = _make_df(n_per_class=20)
    with pytest.raises(ValueError, match="not found"):
        sa.stratified_split(df, ["NoSuchCol"], 0.2, 5, "CellType")
