#!/usr/bin/env python3
"""
Feature-selection concordance scorer.

Validates ClassyFlow's per-class feature selection against an independent,
PI-owned cell-type profile. For each class ClassyFlow modeled, it asks: which
profile cell type do the selected features best fit? The healthy result is that
the class labeled "Helper T" best-matches the "Helper T" definition; a mismatch
(e.g. it best fits "Cytotoxic T", or a distant lineage) flags suspect training
labels or feature leakage.

The profile is authored in CANONICAL marker names (assets/celltype_profile.yaml).
Because ClassyFlow's feature columns are harmonized to those same canonical names
(bin/fixup_columns.py), scoring joins canonical-to-canonical with no silent
marker drop.

Inputs
------
--profile         YAML cell-type profile (assets/celltype_profile.yaml). A list of
                  cell_types, each with `name`, optional `parent` (a cell-type
                  name), and `markers: {canonical_marker: positive|negative}`.
--fs-results      One or more feature_selection_<celltype>_results.json files
                  (from generate_cell_type_selection.py). Read: `celltype` and
                  `marker_importance` — a list of {marker, importance, direction}
                  over the selected features, already collapsed to one record per
                  marker upstream. That list is the single source of truth for
                  marker-level importance and direction; nothing is re-derived
                  here.

Scoring
-------
Per CF class:
  1. For each profile cell type, over markers present in BOTH the class's
     marker->direction map and the profile node's signature, score = an F1 over
     agreements (positive<->positive, negative<->negative).
  2. Rank profile types; pick best_match.
  3. Resolve the CF class to its own profile node (by normalized name) and report
     the relation of best_match to that self node.

Outputs
  <prefix>.csv   one row per CF class (headline table)
  <prefix>.json  full ranked profile-type list per CF class

Usage:
    score_feature_concordance.py --profile celltype_profile.yaml \
        --fs-results feature_selection_*_results.json \
        --out-prefix feature_concordance
"""

import re
import csv
import json
import argparse

import yaml


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def norm_key(s):
    """Normalize a label for joining a CF class name to a profile node.

    CF class labels come from the training data; profile nodes are hand-authored
    by the PI. Stripping to lowercase alphanumerics lets "Helper T" match
    "helper-T" without demanding the two sources agree on punctuation.
    """
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


# --------------------------------------------------------------------------- #
# profile
# --------------------------------------------------------------------------- #

def load_profile(path):
    """Load the cell-type profile YAML and index its cell types.

    Returns (nodes_by_name, key_index) where each node carries name, parent
    (a cell-type name or None), and sig = {canonical_marker: state}.
    """
    with open(path) as f:
        profile = yaml.safe_load(f) or {}

    nodes = {}
    for t in profile.get("cell_types", []) or []:
        name = t.get("name")
        if not name:
            continue
        sig = {}
        for marker, state in (t.get("markers") or {}).items():
            st = str(state).strip().lower()
            if st in ("positive", "negative"):
                sig[marker] = st
        nodes[name] = {
            "name": name,
            "parent": t.get("parent") or None,
            "sig": sig,
        }

    # index for joining a CF label to a self node (by normalized name)
    key_index = {}
    for n in nodes.values():
        key_index.setdefault(norm_key(n["name"]), n["name"])

    return nodes, key_index


def ancestors(name, nodes):
    """Set of ancestor names (exclusive of `name`)."""
    out, guard = set(), 0
    cur = nodes.get(name, {}).get("parent")
    while cur and guard < 100:
        out.add(cur)
        cur = nodes.get(cur, {}).get("parent")
        guard += 1
    return out


def relation(best_name, self_name, nodes):
    """Relation of best_match to the CF class's own profile node."""
    if self_name is None:
        return "no_self_def"
    if best_name == self_name:
        return "self"
    if best_name in ancestors(self_name, nodes):
        return "ancestor"
    if self_name in ancestors(best_name, nodes):
        return "descendant"
    if nodes.get(best_name, {}).get("parent") == nodes.get(self_name, {}).get("parent"):
        return "sibling"
    return "unrelated"


# --------------------------------------------------------------------------- #
# feature-selection results -> per-marker direction
# --------------------------------------------------------------------------- #

def read_fs_result(fs_json):
    """Read (cf_class, {marker: direction}) from a feature-selection results JSON.

    ``marker_importance`` is produced by collapse_to_markers() in
    generate_cell_type_selection.py — already one record per marker over the
    selected features.

    Markers whose coefficient was exactly zero carry ``direction: null``; they
    are skipped here because there is no sign to agree or disagree with a
    profile definition. They are still charted in the report.
    """
    with open(fs_json) as f:
        data = json.load(f)
    records = data.get("marker_importance", []) or []
    if not records:
        # Without this the class scores against nothing and every profile type
        # comes back "no shared markers", which reads like a real biological
        # finding instead of a broken upstream contract.
        print(f"WARNING: {fs_json} has no 'marker_importance'; "
              f"class '{data.get('celltype', '')}' cannot be scored.")
    cf_dirs = {r["marker"]: r["direction"] for r in records if r.get("direction")}
    return data.get("celltype", ""), cf_dirs


def score_against_profile(cf_dirs, nodes):
    """Rank every profile node against one CF class's marker->direction map.

    ``score`` is an F1 that balances purity (agreements among shared markers)
    with coverage (agreements against the definition's full marker set). This
    penalises both direction conflicts and expected markers that went unpicked,
    and stops a lone 1/1 match from outranking a broad 7/8 one.
    """
    ranked = []
    for n in nodes.values():
        expected = len(n["sig"])
        shared, agree, conflicts = 0, 0, []
        for marker, st in n["sig"].items():
            if marker in cf_dirs:
                shared += 1
                if cf_dirs[marker] == st:
                    agree += 1
                else:
                    conflicts.append(marker)
        precision = (agree / shared) if shared else None
        recall = (agree / expected) if expected else None
        if agree > 0 and precision and recall:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0 if shared else None
        ranked.append({
            "cell_type": n["name"],
            "score": f1,
            "precision": None if precision is None else round(precision, 4),
            "recall": None if recall is None else round(recall, 4),
            "shared_markers": shared,
            "expected_markers": expected,
            "agreements": agree,
            "conflicting_markers": conflicts,
        })
    # best = highest F1, tie-broken by most shared markers; unscored last
    ranked.sort(
        key=lambda r: (r["score"] is not None,
                       r["score"] if r["score"] is not None else -1,
                       r["shared_markers"]),
        reverse=True,
    )
    return ranked


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Score ClassyFlow feature selection against a cell-type profile.")
    parser.add_argument("--profile", required=True,
                        help="Cell-type profile YAML (assets/celltype_profile.yaml).")
    parser.add_argument("--fs-results", required=True, nargs="+",
                        help="feature_selection_<celltype>_results.json file(s).")
    parser.add_argument("--out-prefix", default="feature_concordance",
                        help="Output file prefix.")
    args = parser.parse_args()

    nodes, key_index = load_profile(args.profile)

    rows, detail = [], []
    for fs_json in args.fs_results:
        cf_class, cf_dirs = read_fs_result(fs_json)
        ranked = score_against_profile(cf_dirs, nodes)

        self_name = key_index.get(norm_key(cf_class))
        best = next((r for r in ranked if r["score"] is not None), None)
        self_row = next((r for r in ranked if r["cell_type"] == self_name), None)

        note = ""
        if best is None:
            note = "no shared markers between features and any profile definition"

        rows.append({
            "cf_class": cf_class,
            "self_cell_type": self_name or "",
            "self_score": "" if not self_row or self_row["score"] is None
                          else round(self_row["score"], 4),
            "best_match": best["cell_type"] if best else "",
            "best_score": "" if not best else round(best["score"], 4),
            "best_precision": "" if not best or best["precision"] is None
                              else best["precision"],
            "best_recall": "" if not best or best["recall"] is None
                           else best["recall"],
            "relation": relation(best["cell_type"], self_name, nodes) if best else "no_match",
            "shared_markers": best["shared_markers"] if best else 0,
            "conflicting_markers": "|".join(best["conflicting_markers"]) if best else "",
            "note": note,
        })
        detail.append({
            "cf_class": cf_class,
            "self_cell_type": self_name,
            "cf_marker_directions": cf_dirs,
            "ranked": ranked,
        })

    csv_path = f"{args.out_prefix}.csv"
    json_path = f"{args.out_prefix}.json"

    fieldnames = ["cf_class", "self_cell_type", "self_score", "best_match",
                  "best_score", "best_precision", "best_recall", "relation",
                  "shared_markers", "conflicting_markers", "note"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["cf_class"]):
            w.writerow(r)

    with open(json_path, "w") as f:
        json.dump({"results": detail}, f, indent=2)

    print(f"Wrote {csv_path} and {json_path} for {len(rows)} class(es).")


if __name__ == "__main__":
    main()
