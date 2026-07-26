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


def test_marker_directions_highest_importance_per_marker(tmp_path):
    csv_path = tmp_path / "coefficients_Helper_T.csv"
    csv_path.write_text(
        "Name,Feature_Importance,Coefficient,Direction\n"
        "CD4: Mean,0.9,0.9,Positive\n"
        "CD4: Max,0.2,-0.2,Negative\n"   # lower importance -> ignored for CD4
        "CD8: Mean,0.5,-0.5,Negative\n"
        "CD3: Mean,0.0,0.0,Positive\n"   # zero importance -> dropped
    )
    dirs = sc.marker_directions(str(csv_path))
    assert dirs == {"CD4": "positive", "CD8": "negative"}
