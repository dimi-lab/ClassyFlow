"""Unit tests for bin/add_empty_marker_noise.py helper functions.

NOTE: the module top-level does ``len = __builtins__.len`` etc., which raises
``AttributeError`` on normal import (``__builtins__`` is a dict in an imported
module, only a module object under ``__main__``). We load it here with the
builtins module injected so its functions can be tested without editing the
production script. (Tracked as a cleanup item.)
"""
import builtins
import types
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.unit


def _load_ann():
    path = Path(__file__).resolve().parents[2] / "bin" / "add_empty_marker_noise.py"
    mod = types.ModuleType("add_empty_marker_noise")
    mod.__file__ = str(path)
    mod.__dict__["__builtins__"] = builtins  # module (not dict) so `.len` resolves
    exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
    return mod


ann = _load_ann()


def test_get_unique_sets_detection_object():
    suffixes = ann.getUniqueSets("DetectionObject")
    assert suffixes == [": Min", ": Max", ": Median", ": Mean", ": Std.Dev.", ": Variance"]


def test_get_unique_sets_cell_object_has_all_compartments():
    suffixes = ann.getUniqueSets("CellObject")
    # 4 compartments x 6 stats = 24 suffixes.
    assert len(suffixes) == 24
    assert ": Nucleus: Mean" in suffixes
    assert ": Membrane: Variance" in suffixes


def _panel(cols_values: dict, markers=("CD4", "CD8", "PanCK")):
    """Build a panel-design DataFrame (markers as index, batches as columns)."""
    return pd.DataFrame(cols_values, index=list(markers))


def test_resolve_batch_column_exact_match():
    panel = _panel({"b1": [1, 1, 1], "b2": [1, 0, 1]})
    col, prefix, suffix = ann.resolve_batch_column(panel, "b1")
    assert col == "b1" and prefix == "b1" and suffix == ""


def test_resolve_batch_column_single_prefix_match():
    panel = _panel({"TMA1990-abcde": [1, 1, 0]})
    col, prefix, suffix = ann.resolve_batch_column(panel, "TMA1990-xyz99")
    assert col == "TMA1990-abcde"
    assert prefix == "TMA1990"


def test_resolve_batch_column_no_prefix_raises_keyerror():
    panel = _panel({"b1": [1, 1, 1]})
    with pytest.raises(KeyError, match="No columns in panel design start with prefix"):
        ann.resolve_batch_column(panel, "ZZZ")


def test_resolve_batch_column_ambiguous_raises_keyerror():
    panel = _panel({"TMA-aaaaa": [1, 1, 1], "TMA-bbbbb": [1, 1, 1]})
    with pytest.raises(KeyError, match="Ambiguous"):
        ann.resolve_batch_column(panel, "TMA-ccccc")


def test_get_missing_markers_returns_zero_valued_markers():
    panel = _panel({"b1": [1, 0, 1]})  # CD8 missing
    missing = ann.get_missing_markers(panel, "b1", "b1")
    assert missing == ["CD8"]


def test_check_header_conflicts_raises_on_prefix_collision():
    df = pd.DataFrame({"b1_extra": [1], "CD4: Mean": [2]})
    with pytest.raises(ValueError, match="Header columns found"):
        ann.check_header_conflicts(df, "b1")


def test_check_header_conflicts_ok_when_no_collision():
    df = pd.DataFrame({"CD4: Mean": [1], "CD8: Mean": [2]})
    # Should not raise.
    ann.check_header_conflicts(df, "b1")
