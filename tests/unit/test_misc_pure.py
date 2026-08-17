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


# --- bin/scimap_clustering.py : find_best_coord ---------------------------
# scimap_clustering imports scanpy/scimap (numba stack); skip if unavailable
# (e.g. on Python 3.12 where numba can't install). Covered in the container env.
def test_find_best_coord_prefers_specific_micron_column():
    pytest.importorskip("scanpy")
    import scimap_clustering as sc
    cols = ["Centroid X µm", "Centroid X", "X", "CD4: Mean"]
    assert sc.find_best_coord(cols, "X") == "Centroid X µm"


def test_find_best_coord_raises_when_absent():
    pytest.importorskip("scanpy")
    import scimap_clustering as sc
    with pytest.raises(ValueError, match="No column found"):
        sc.find_best_coord(["CD4: Mean", "CD8: Mean"], "X")
