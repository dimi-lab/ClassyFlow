"""Unit tests for bin/fixup_columns.py marker-harmonization helpers."""
import pandas as pd
import pytest

import fixup_columns as fx

pytestmark = pytest.mark.unit


VOCAB = (
    "markers:\n"
    "  aSMA:\n"
    "    aliases: [ASMA]\n"
    "  DAPI:\n"
    "    aliases: [DAPI_AF_R01]\n"
    "drop:\n"
    "  - Batch\n"
)


def test_load_marker_vocab(tmp_path):
    yaml_path = tmp_path / "markers.yaml"
    yaml_path.write_text(VOCAB)
    canonical, alias_to_canonical, drop = fx.load_marker_vocab(str(yaml_path))
    assert canonical == {"aSMA", "DAPI"}
    assert alias_to_canonical == {"ASMA": "aSMA", "DAPI_AF_R01": "DAPI"}
    assert drop == ["Batch"]


def test_load_marker_vocab_missing_sections_default_empty(tmp_path):
    yaml_path = tmp_path / "markers.yaml"
    yaml_path.write_text("markers:\n  CD3:\n    aliases: [CD3e]\n")
    canonical, alias_to_canonical, drop = fx.load_marker_vocab(str(yaml_path))
    assert canonical == {"CD3"}
    assert alias_to_canonical == {"CD3e": "CD3"}
    assert drop == []


def test_remove_columns_func_drops_only_present():
    df = pd.DataFrame({"Batch": [1], "Keep": [2]})
    out = fx.remove_columns_func(df, ["Batch", "NotThere"])
    assert list(out.columns) == ["Keep"]


def test_resolve_markers_keeps_canonical_and_rewrites_alias():
    # Canonical column kept; alias rewritten preserving the metric suffix.
    df = pd.DataFrame({
        "aSMA: Mean": [1.0],       # already canonical
        "ASMA: Max": [2.0],        # alias -> aSMA
        "CD4: Mean": [3.0],        # not in vocab, no ':' resolution needed
        "Image": ["x"],            # metadata, no ':' -> untouched
    })
    out, unresolved = fx.resolve_markers(df, {"aSMA"}, {"ASMA": "aSMA"})
    assert list(out.columns) == ["aSMA: Mean", "aSMA: Max", "CD4: Mean", "Image"]
    # CD4 is a marker token that did not resolve.
    assert unresolved == ["CD4"]


def test_resolve_markers_does_not_match_substring_without_colon():
    # "ASMA" alias should not rewrite "ASMAX: Mean" (only the token before ':').
    df = pd.DataFrame({"ASMAX: Mean": [1.0]})
    out, unresolved = fx.resolve_markers(df, {"aSMA"}, {"ASMA": "aSMA"})
    assert list(out.columns) == ["ASMAX: Mean"]
    assert unresolved == ["ASMAX"]


def test_resolve_markers_metadata_columns_never_flagged():
    df = pd.DataFrame({"Image": ["a"], "Centroid X µm": [0.0], "Classification": ["T"]})
    out, unresolved = fx.resolve_markers(df, {"CD3"}, {})
    assert list(out.columns) == ["Image", "Centroid X µm", "Classification"]
    assert unresolved == []


def test_dedupe_collapses_alias_onto_existing_canonical():
    # DAPI_AF_R01 -> DAPI while a real DAPI: Mean is present would duplicate.
    df = pd.DataFrame({
        "DAPI: Mean": [1.0],
        "DAPI_AF_R01: Mean": [2.0],
    })
    df, _ = fx.resolve_markers(df, {"DAPI"}, {"DAPI_AF_R01": "DAPI"})
    provenance = [("DAPI: Mean", "DAPI: Mean"),
                  ("DAPI_AF_R01: Mean", "DAPI: Mean")]
    df, dropped = fx.dedupe_renamed_columns(df, provenance)
    assert list(df.columns) == ["DAPI: Mean"]
    assert dropped == ["DAPI_AF_R01: Mean -> DAPI: Mean"]
    # kept the canonical/original column's values
    assert df["DAPI: Mean"].tolist() == [1.0]


def test_main_writes_fx_pickle_with_transformations(tmp_path):
    df = pd.DataFrame({
        "Batch": [1, 2],
        "ASMA: Mean": [10.0, 20.0],
        "CD4: Mean": [1.0, 2.0],
    })
    in_pkl = tmp_path / "merged_dataframe_b1.pkl"
    df.to_pickle(in_pkl)

    vocab = tmp_path / "markers.yaml"
    vocab.write_text(VOCAB)

    fx.main(str(in_pkl), str(vocab))

    out_pkl = tmp_path / "merged_dataframe_b1_fx.pkl"
    assert out_pkl.exists()
    result = pd.read_pickle(out_pkl)
    assert "Batch" not in result.columns
    assert "aSMA: Mean" in result.columns
    assert "CD4: Mean" in result.columns


def test_main_strict_exits_on_unresolved_marker(tmp_path):
    df = pd.DataFrame({"MysteryMarker: Mean": [1.0]})
    in_pkl = tmp_path / "merged_dataframe_b1.pkl"
    df.to_pickle(in_pkl)

    vocab = tmp_path / "markers.yaml"
    vocab.write_text(VOCAB)

    with pytest.raises(SystemExit) as exc:
        fx.main(str(in_pkl), str(vocab), strict=True)
    assert exc.value.code == 1


def test_main_warn_mode_continues_on_unresolved_marker(tmp_path):
    df = pd.DataFrame({"MysteryMarker: Mean": [1.0]})
    in_pkl = tmp_path / "merged_dataframe_b1.pkl"
    df.to_pickle(in_pkl)

    vocab = tmp_path / "markers.yaml"
    vocab.write_text(VOCAB)

    # Default (warn) mode: no exception, output still written unchanged.
    fx.main(str(in_pkl), str(vocab))
    out = pd.read_pickle(tmp_path / "merged_dataframe_b1_fx.pkl")
    assert "MysteryMarker: Mean" in out.columns
