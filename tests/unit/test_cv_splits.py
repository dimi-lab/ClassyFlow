"""Unit tests for CV-split creation in bin/get_xgboost_parameter_search.py (seeded)."""
import pickle

import numpy as np
import pandas as pd
import pytest

import get_xgboost_parameter_search as gps

pytestmark = pytest.mark.unit


def _train_df(n=60, classes=("A", "B", "C")):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "CellType": [classes[i % len(classes)] for i in range(n)],
        "f1": rng.random(n),
        "f2": rng.random(n),
    })


def test_random_cv_splits_shape_and_determinism(tmp_path, monkeypatch):
    df = _train_df()
    monkeypatch.chdir(tmp_path)

    a = gps.create_random_cv_splits(df, "CellType", n_splits=4)
    b = gps.create_random_cv_splits(df, "CellType", n_splits=4)

    assert len(a) == 4
    for train_idx, test_idx in a:
        assert set(train_idx).isdisjoint(set(test_idx))
        assert set(train_idx) | set(test_idx) <= set(range(len(df)))
    # Seeded (random_state=42) -> identical across calls.
    for (ta, sa), (tb, sb) in zip(a, b):
        assert np.array_equal(ta, tb)
        assert np.array_equal(sa, sb)


def test_kfold_cv_splits_written_and_cover_all_rows(tmp_path, monkeypatch):
    df = _train_df()
    monkeypatch.chdir(tmp_path)

    gps.create_kfold_cv_splits(df, "CellType", n_splits=3)
    out = tmp_path / "cv_splits.pkl"
    assert out.exists()

    with open(out, "rb") as fh:
        splits = pickle.load(fh)
    assert len(splits) == 3
    # Each row appears in exactly one test fold across a k-fold split.
    all_test = np.concatenate([test for _, test in splits])
    assert sorted(all_test) == list(range(len(df)))
