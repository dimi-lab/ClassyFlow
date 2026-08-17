"""Shared pytest fixtures and path setup for the ClassyFlow test suite.

The pipeline's computational logic lives in ``bin/`` as standalone argparse
scripts (not an installable package). We add ``bin/`` to ``sys.path`` so tests
can ``import <script_name>`` and call the functions inside directly.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

# --- Repo paths -----------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = REPO_ROOT / "bin"
DATA_DIR = REPO_ROOT / "data"

# Make bin/ scripts importable as modules (e.g. `import fixup_columns`).
sys.path.insert(0, str(BIN_DIR))


@pytest.fixture
def data_dir() -> Path:
    """Absolute path to the bundled sample QuPath data (2 batches)."""
    return DATA_DIR


@pytest.fixture
def write_pickle(tmp_path):
    """Return a helper that writes a DataFrame to a .pkl in tmp and returns the path."""
    def _write(df: pd.DataFrame, name: str = "df.pkl") -> str:
        path = tmp_path / name
        df.to_pickle(path)
        return str(path)
    return _write


@pytest.fixture
def sample_quant_df() -> pd.DataFrame:
    """A tiny synthetic quantification-style DataFrame with an ``Image`` column,
    a couple of marker ``Mean`` columns and a class label column."""
    return pd.DataFrame({
        "Image": ["A.ome.tif", "A.ome.tif", "B.ome.tif", "B.ome.tif"],
        "Classification": ["Tumor", "Fibroblast", "Tumor", "Tumor"],
        "CD4: Mean": [10.0, 20.0, 30.0, 40.0],
        "CD8: Mean": [1.0, 2.0, 3.0, 4.0],
        "Centroid X µm": [0.0, 100.0, 200.0, 300.0],
        "Centroid Y µm": [0.0, 100.0, 200.0, 300.0],
    })
