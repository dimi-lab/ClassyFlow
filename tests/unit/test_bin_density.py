"""Unit test for bin/calculate_bin_density.py.

The binning logic lives inside ``main()``, so we exercise it end-to-end via a
subprocess on a tiny hand-built TSV (rather than refactoring production code).
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.unit

SCRIPT = Path(__file__).resolve().parents[2] / "bin" / "calculate_bin_density.py"


def test_bin_density_flags_sparse_bins(tmp_path):
    # Three cells packed into one 120um bin (dense) + one isolated cell (sparse).
    df = pd.DataFrame({
        "Centroid X µm": [10.0, 20.0, 30.0, 5000.0],
        "Centroid Y µm": [10.0, 20.0, 30.0, 5000.0],
        "CellTypePrediction": ["Tumor", "Tumor", "Tumor", "B Cell"],
    })
    in_tsv = tmp_path / "sample_PRED.tsv"
    df.to_csv(in_tsv, sep="\t", index=False)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input_tsv", str(in_tsv),
         "--bin_size", "120", "--density_cutoff", "3"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    out_tsv = tmp_path / "sample_qPRED.tsv"
    assert out_tsv.exists()
    out = pd.read_csv(out_tsv, sep="\t")

    # New QC columns are added.
    for col in ["binX", "binY", "bin_density", "low_bin_density"]:
        assert col in out.columns

    # The 3 clustered cells share a bin of density 3 (<=cutoff -> low), and the
    # isolated cell has density 1 (also low). With cutoff 3, all are low here;
    # assert the clustered cells landed in the same bin.
    clustered = out.iloc[:3]
    assert clustered["binX"].nunique() == 1 and clustered["binY"].nunique() == 1
    assert (clustered["bin_density"] == 3).all()
    assert bool(out.iloc[3]["low_bin_density"]) is True


def test_bin_density_dense_region_not_flagged(tmp_path):
    # 5 cells in one bin, cutoff 3 -> density 5 > 3 -> not low.
    df = pd.DataFrame({
        "Centroid X µm": [1.0, 2.0, 3.0, 4.0, 5.0],
        "Centroid Y µm": [1.0, 2.0, 3.0, 4.0, 5.0],
        "CellTypePrediction": ["Tumor"] * 5,
    })
    in_tsv = tmp_path / "dense_PRED.tsv"
    df.to_csv(in_tsv, sep="\t", index=False)

    subprocess.run(
        [sys.executable, str(SCRIPT), "--input_tsv", str(in_tsv),
         "--bin_size", "120", "--density_cutoff", "3"],
        capture_output=True, text=True, check=True,
    )
    out = pd.read_csv(tmp_path / "dense_qPRED.tsv", sep="\t")
    assert (out["bin_density"] == 5).all()
    assert (~out["low_bin_density"]).all()
