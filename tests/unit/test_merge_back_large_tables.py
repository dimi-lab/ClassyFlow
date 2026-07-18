"""Unit tests for bin/merge_back_large_tables.py validation."""
import pandas as pd
import pytest

import merge_back_large_tables as mb

pytestmark = pytest.mark.unit


def _write_tsv(path, df):
    df.to_csv(path, sep="\t", index=False)
    return str(path)


def test_validate_ok_when_all_files_share_batch_id(tmp_path):
    f1 = _write_tsv(tmp_path / "a.tsv", pd.DataFrame({"original_batchID": ["b1", "b1"], "v": [1, 2]}))
    f2 = _write_tsv(tmp_path / "b.tsv", pd.DataFrame({"original_batchID": ["b1"], "v": [3]}))
    # Should not raise.
    mb.validate_originalBatchID([f1, f2])


def test_validate_raises_when_column_missing(tmp_path):
    f1 = _write_tsv(tmp_path / "a.tsv", pd.DataFrame({"v": [1]}))
    with pytest.raises(ValueError, match="original_batchID.*not found"):
        mb.validate_originalBatchID([f1])


def test_validate_raises_when_file_has_multiple_batch_ids(tmp_path):
    f1 = _write_tsv(tmp_path / "a.tsv", pd.DataFrame({"original_batchID": ["b1", "b2"], "v": [1, 2]}))
    with pytest.raises(ValueError, match="unique batch IDs"):
        mb.validate_originalBatchID([f1])


def test_validate_raises_when_files_disagree(tmp_path):
    f1 = _write_tsv(tmp_path / "a.tsv", pd.DataFrame({"original_batchID": ["b1"], "v": [1]}))
    f2 = _write_tsv(tmp_path / "b.tsv", pd.DataFrame({"original_batchID": ["b2"], "v": [2]}))
    with pytest.raises(ValueError, match="different original_BatchID"):
        mb.validate_originalBatchID([f1, f2])
