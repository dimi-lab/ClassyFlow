"""Functional unit tests for bin/merge_files.py merge routine.

``merge_tab_delimited_files`` writes its pickle output to the current working
directory, so tests run inside ``tmp_path``.
"""
import pandas as pd
import pytest

import merge_files as mf

pytestmark = pytest.mark.unit


def _write_quant(path, images, cd4_means):
    df = pd.DataFrame({
        "Image": images,
        "CD4: Mean": cd4_means,
        "CD8: Mean": [1.0] * len(images),
    })
    df.to_csv(path, sep="\t", index=False)


def test_merge_two_files_adds_tracking_columns(tmp_path, monkeypatch):
    indir = tmp_path / "batchA"
    indir.mkdir()
    _write_quant(indir / "roi1.tsv", ["roi1", "roi1"], [10.0, 20.0])
    _write_quant(indir / "roi2.tsv", ["roi2"], [30.0])

    monkeypatch.chdir(tmp_path)
    mf.merge_tab_delimited_files(
        directory_path=str(indir), excld="", slide_by_prefix=False,
        folder_is_slide=False, input_extension=".tsv", input_delimiter="\t",
        batchID="batchA", target_size=1_000_000,
    )

    out = tmp_path / "merged_dataframe_batchA-00000.pkl"
    assert out.exists()
    merged = pd.read_pickle(out)
    assert len(merged) == 3
    assert "Slide" in merged.columns
    assert (merged["original_batchID"] == "batchA").all()
    assert set(merged["Image"]) == {"roi1", "roi2"}


def test_merge_folder_is_slide_prefixes_image(tmp_path, monkeypatch):
    indir = tmp_path / "slideX"
    indir.mkdir()
    _write_quant(indir / "roi1.tsv", ["roi1"], [10.0])

    monkeypatch.chdir(tmp_path)
    mf.merge_tab_delimited_files(
        directory_path=str(indir), excld="", slide_by_prefix=False,
        folder_is_slide=True, input_extension=".tsv", input_delimiter="\t",
        batchID="slideX", target_size=1_000_000,
    )
    merged = pd.read_pickle(tmp_path / "merged_dataframe_slideX-00000.pkl")
    # folder_is_slide rewrites Image as "<dir>-<image>".
    assert merged["Image"].iloc[0].endswith("-roi1")


def test_merge_missing_image_column_raises(tmp_path, monkeypatch):
    indir = tmp_path / "bad"
    indir.mkdir()
    pd.DataFrame({"CD4: Mean": [1.0]}).to_csv(indir / "roi1.tsv", sep="\t", index=False)

    monkeypatch.chdir(tmp_path)
    with pytest.raises(AssertionError, match="Image"):
        mf.merge_tab_delimited_files(
            directory_path=str(indir), excld="", slide_by_prefix=False,
            folder_is_slide=False, input_extension=".tsv", input_delimiter="\t",
            batchID="bad", target_size=1_000_000,
        )
