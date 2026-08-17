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
--coefficients    One or more coefficients_<celltype>.csv files (from
                  generate_cell_type_selection.py). Columns: Name,
                  Feature_Importance, Coefficient, Direction. "Name" is a
                  measurement column "<marker>: <compartment>: <statistic>".

Scoring
-------
Per CF class:
  1. marker = Name.split(":")[0].strip()
  2. Reduce to one signed direction per marker: the direction of that marker's
     highest-importance feature (rule "a").
  3. For each profile cell type, over markers present in BOTH the CF
     marker->direction map and the profile node's signature, score =
     agreements / shared (Positive<->positive, Negative<->negative).
  4. Rank profile types; pick best_match.
  5. Resolve the CF class to its own profile node (by normalized name) and report
     the relation of best_match to that self node.

Outputs (standalone; HTML integration deferred)
  <prefix>.csv   one row per CF class (headline table)
  <prefix>.json  full ranked profile-type list per CF class

Usage:
    score_feature_concordance.py --profile celltype_profile.yaml \
        --coefficients coefficients_*.csv --out-prefix feature_concordance
"""

import os
import re
import csv
import json
import argparse

import yaml


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def norm_key(s):
    """Normalize a label for joining CF class names to profile nodes.

    CF coefficient files are named coefficients_<safe>.csv where <safe> replaced
    spaces / '|' / '/' with '_'. Profile nodes carry human names. Stripping to
    lowercase alphanumerics makes the two comparable regardless of that
    sanitization.
    """
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def marker_of(feature_name):
    """Marker token of a measurement column '<marker>: <compartment>: ...'.

    Mirrors the convention in bin/fixup_columns.py (token before the first ':').
    """
    return str(feature_name).split(":")[0].strip()


def state_of_direction(direction):
    """Map a coefficient Direction to a +/- state string."""
    d = str(direction).strip().lower()
    if d.startswith("pos"):
        return "positive"
    if d.startswith("neg"):
        return "negative"
    return None


def cf_class_from_filename(path):
    """Recover the CF class label from coefficients_<safe>.csv."""
    base = os.path.basename(path)
    m = re.match(r'coefficients_(.+)\.csv$', base)
    return m.group(1) if m else os.path.splitext(base)[0]


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
# coefficients -> per-marker direction (rule "a")
# --------------------------------------------------------------------------- #

def marker_directions(coeff_csv):
    """Reduce a coefficients file to {marker: state} via highest-importance rule.

    Only non-zero coefficients contribute (a zero-coef feature was not selected
    by the Lasso and carries no directional signal).
    """
    best = {}  # marker -> (importance, state)
    with open(coeff_csv, newline="") as f:
        for row in csv.DictReader(f):
            try:
                imp = abs(float(row.get("Feature_Importance") or 0))
            except (TypeError, ValueError):
                imp = 0
            if imp == 0:
                continue
            state = state_of_direction(row.get("Direction"))
            if state is None and row.get("Coefficient") not in (None, ""):
                coef = float(row["Coefficient"])
                state = "positive" if coef > 0 else ("negative" if coef < 0 else None)
            if state is None:
                continue
            marker = marker_of(row["Name"])
            if marker not in best or imp > best[marker][0]:
                best[marker] = (imp, state)
    return {m: s for m, (_, s) in best.items()}


def score_against_profile(cf_dirs, nodes):
    """Rank every profile node against one CF class's marker->direction map."""
    ranked = []
    for n in nodes.values():
        shared, agree, conflicts = 0, 0, []
        for marker, st in n["sig"].items():
            if marker in cf_dirs:
                shared += 1
                if cf_dirs[marker] == st:
                    agree += 1
                else:
                    conflicts.append(marker)
        ranked.append({
            "cell_type": n["name"],
            "score": (agree / shared) if shared else None,
            "shared_markers": shared,
            "agreements": agree,
            "conflicting_markers": conflicts,
        })
    # best = highest score, tie-broken by most shared markers; unscored last
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
    parser.add_argument("--coefficients", required=True, nargs="+",
                        help="coefficients_<celltype>.csv file(s).")
    parser.add_argument("--out-prefix", default="feature_concordance",
                        help="Output file prefix.")
    args = parser.parse_args()

    nodes, key_index = load_profile(args.profile)

    rows, detail = [], []
    for coeff_csv in args.coefficients:
        cf_class = cf_class_from_filename(coeff_csv)
        cf_dirs = marker_directions(coeff_csv)
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
                  "best_score", "relation", "shared_markers",
                  "conflicting_markers", "note"]
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
