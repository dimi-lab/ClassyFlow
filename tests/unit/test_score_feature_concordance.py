"""Unit tests for bin/score_feature_concordance.py (YAML cell-type profile)."""
import pytest

import score_feature_concordance as sc

pytestmark = pytest.mark.unit


PROFILE = (
    "cell_types:\n"
    "  - name: T cell\n"
    "    markers: { CD3: positive }\n"
    "  - name: Helper T\n"
    "    parent: T cell\n"
    "    markers: { CD3: positive, CD4: positive, CD8: negative }\n"
    "  - name: Cytotoxic T\n"
    "    parent: T cell\n"
    "    markers: { CD3: positive, CD8: positive, CD4: negative }\n"
)


def _load(tmp_path):
    p = tmp_path / "celltype_profile.yaml"
    p.write_text(PROFILE)
    return sc.load_profile(str(p))


def test_load_profile_name_parent_markers(tmp_path):
    nodes, key_index = _load(tmp_path)
    assert set(nodes) == {"T cell", "Helper T", "Cytotoxic T"}
    assert nodes["Helper T"]["parent"] == "T cell"
    assert nodes["Helper T"]["sig"] == {"CD3": "positive", "CD4": "positive", "CD8": "negative"}
    # key_index maps normalized names back to the canonical node name.
    assert key_index[sc.norm_key("helper t")] == "Helper T"


def test_load_profile_ignores_invalid_states(tmp_path):
    p = tmp_path / "prof.yaml"
    p.write_text("cell_types:\n  - name: X\n    markers: { CD3: maybe, CD4: positive }\n")
    nodes, _ = sc.load_profile(str(p))
    assert nodes["X"]["sig"] == {"CD4": "positive"}


def test_ancestors_walks_parent_chain(tmp_path):
    nodes, _ = _load(tmp_path)
    assert sc.ancestors("Helper T", nodes) == {"T cell"}
    assert sc.ancestors("T cell", nodes) == set()


def test_relation_variants(tmp_path):
    nodes, _ = _load(tmp_path)
    assert sc.relation("Helper T", "Helper T", nodes) == "self"
    assert sc.relation("T cell", "Helper T", nodes) == "ancestor"
    assert sc.relation("Helper T", "T cell", nodes) == "descendant"
    assert sc.relation("Cytotoxic T", "Helper T", nodes) == "sibling"
    assert sc.relation("Helper T", None, nodes) == "no_self_def"


def test_score_against_profile_canonical_join(tmp_path):
    nodes, _ = _load(tmp_path)
    # A CF class whose learned directions match Helper T exactly.
    cf_dirs = {"CD3": "positive", "CD4": "positive", "CD8": "negative"}
    ranked = sc.score_against_profile(cf_dirs, nodes)
    best = ranked[0]
    assert best["cell_type"] == "Helper T"
    assert best["score"] == 1.0
    assert best["shared_markers"] == 3


def _write_fs_result(tmp_path, celltype, marker_importance, name=None):
    import json
    p = tmp_path / (name or f"feature_selection_{celltype}_results.json")
    p.write_text(json.dumps({"celltype": celltype,
                             "marker_importance": marker_importance}))
    return p


def test_read_fs_result_uses_upstream_marker_importance(tmp_path):
    # The feature->marker collapse happens upstream; this only reads it back.
    p = _write_fs_result(tmp_path, "Helper T", [
        {"marker": "CD4", "importance": 0.9, "direction": "positive"},
        {"marker": "CD8", "importance": 0.5, "direction": "negative"},
    ])
    cf_class, dirs = sc.read_fs_result(str(p))
    assert cf_class == "Helper T"
    assert dirs == {"CD4": "positive", "CD8": "negative"}


def test_read_fs_result_preserves_label_with_spaces(tmp_path):
    # cf_class comes from the JSON's celltype field, so labels are not mangled
    # by filename sanitization (which used to collapse "T cell" -> "T_cell").
    p = _write_fs_result(tmp_path, "T cell", [
        {"marker": "CD3", "importance": 0.4, "direction": "positive"}],
        name="feature_selection_T_cell_results.json")
    cf_class, _ = sc.read_fs_result(str(p))
    assert cf_class == "T cell"


def test_read_fs_result_warns_on_missing_marker_importance(tmp_path, capsys):
    # A broken upstream contract must be visible: without a warning, every
    # profile type scores "no shared markers", which looks like a real result.
    import json
    p = tmp_path / "feature_selection_Empty_results.json"
    p.write_text(json.dumps({"celltype": "Empty"}))
    cf_class, dirs = sc.read_fs_result(str(p))
    assert cf_class == "Empty"
    assert dirs == {}
    out = capsys.readouterr().out
    assert "no 'marker_importance'" in out and "Empty" in out
