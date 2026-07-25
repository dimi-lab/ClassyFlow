#!/usr/bin/env python3
"""
Feature-selection concordance scorer.

Validates ClassyFlow's per-class feature selection against an independent,
PI-owned cell-type definition ("scoring card") exported from the phenotype
registry app. For each class ClassyFlow modeled, it asks: which card cell type
do the selected features best fit? The healthy result is that the class labeled
"Helper T" best-matches the "Helper T" definition; a mismatch (e.g. it best
fits "Cytotoxic T", or a distant lineage) flags suspect training labels or
feature leakage.

Inputs
------
--card            JSON exported from GET /api/export?project=<pid>. Carries the
                  cell-type tree and, per node, resolved_signature_raw =
                  [{raw_name, state}] in this panel's raw channel names.
--coefficients    One or more coefficients_<celltype>.csv files (from
                  generate_cell_type_selection.py). Columns: Name,
                  Feature_Importance, Coefficient, Direction. "Name" is a QuPath
                  measurement column "<marker>: <compartment>: <statistic>".

Scoring
-------
Per CF class:
  1. marker = Name.split(":")[0].strip()
  2. Reduce to one signed direction per marker: the direction of that marker's
     highest-importance feature (rule "a").
  3. For each card cell type, over markers present in BOTH the CF marker->direction
     map and the card node's resolved signature, score = agreements / shared
     (Positive<->positive, Negative<->negative).
  4. Rank card types; pick best_match.
  5. Resolve the CF class to its own card node (name/id, normalized) and report
     the relation of best_match to that self node.

Outputs (standalone; HTML integration deferred)
  <prefix>.csv   one row per CF class (headline table)
  <prefix>.json  full ranked card-type list per CF class

Usage:
    score_feature_concordance.py --card card.json \
        --coefficients coefficients_*.csv --out-prefix feature_concordance
"""

import os
import re
import csv
import json
import argparse


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def norm_key(s):
    """Normalize a label for joining CF class names to card nodes.

    CF coefficient files are named coefficients_<safe>.csv where <safe> replaced
    spaces / '|' / '/' with '_'. Card nodes carry human names and ids. Stripping
    to lowercase alphanumerics makes the two comparable regardless of that
    sanitization.
    """
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def marker_of(feature_name):
    """Marker token of a QuPath measurement column '<marker>: <compartment>: ...'.

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
# card
# --------------------------------------------------------------------------- #

def load_card(path):
    """Load the scoring card and index its cell types.

    Returns (nodes_by_id, node_key_index) where each node carries id, name,
    parent_id, and sig = {marker(raw): state}.
    """
    with open(path) as f:
        card = json.load(f)

    if card.get("marker_name_space") not in (None, "raw"):
        # Not fatal, but scoring assumes raw channel names matching CF columns.
        print(f"WARNING: card marker_name_space is "
              f"'{card.get('marker_name_space')}', expected 'raw'. "
              f"Marker names may not match ClassyFlow feature columns.")

    nodes = {}
    for t in card.get("cell_types", []):
        sig = {}
        for c in t.get("resolved_signature_raw", []):
            st = str(c.get("state", "")).strip().lower()
            if st in ("positive", "negative"):
                sig[c["raw_name"]] = st
        nodes[t["id"]] = {
            "id": t["id"],
            "name": t.get("name") or t["id"],
            "parent_id": t.get("parent_id") or None,
            "sig": sig,
        }

    # index for joining a CF label to a self node (by name, fallback id)
    key_index = {}
    for n in nodes.values():
        key_index.setdefault(norm_key(n["name"]), n["id"])
        key_index.setdefault(norm_key(n["id"]), n["id"])

    return nodes, key_index


def ancestors(node_id, nodes):
    """Set of ancestor ids (exclusive of node_id)."""
    out, cur, guard = set(), nodes.get(node_id), 0
    cur = nodes[node_id]["parent_id"] if node_id in nodes else None
    while cur and guard < 100:
        out.add(cur)
        cur = nodes.get(cur, {}).get("parent_id")
        guard += 1
    return out


def relation(best_id, self_id, nodes):
    """Relation of best_match to the CF class's own card node."""
    if self_id is None:
        return "no_self_def"
    if best_id == self_id:
        return "self"
    if best_id in ancestors(self_id, nodes):
        return "ancestor"
    if self_id in ancestors(best_id, nodes):
        return "descendant"
    if nodes.get(best_id, {}).get("parent_id") == nodes.get(self_id, {}).get("parent_id"):
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


def score_against_card(cf_dirs, nodes):
    """Rank every card node against one CF class's marker->direction map."""
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
            "card_id": n["id"],
            "card_name": n["name"],
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
        description="Score ClassyFlow feature selection against a cell-type card.")
    parser.add_argument("--card", required=True,
                        help="Scoring card JSON (from /api/export).")
    parser.add_argument("--coefficients", required=True, nargs="+",
                        help="coefficients_<celltype>.csv file(s).")
    parser.add_argument("--out-prefix", default="feature_concordance",
                        help="Output file prefix.")
    args = parser.parse_args()

    nodes, key_index = load_card(args.card)

    rows, detail = [], []
    for coeff_csv in args.coefficients:
        cf_class = cf_class_from_filename(coeff_csv)
        cf_dirs = marker_directions(coeff_csv)
        ranked = score_against_card(cf_dirs, nodes)

        self_id = key_index.get(norm_key(cf_class))
        best = next((r for r in ranked if r["score"] is not None), None)
        self_row = next((r for r in ranked if r["card_id"] == self_id), None)

        note = ""
        if best is None:
            note = "no shared markers between features and any card definition"

        rows.append({
            "cf_class": cf_class,
            "self_node_id": self_id or "",
            "self_score": "" if not self_row or self_row["score"] is None
                          else round(self_row["score"], 4),
            "best_match_id": best["card_id"] if best else "",
            "best_match_name": best["card_name"] if best else "",
            "best_score": "" if not best else round(best["score"], 4),
            "relation": relation(best["card_id"], self_id, nodes) if best else "no_match",
            "shared_markers": best["shared_markers"] if best else 0,
            "conflicting_markers": "|".join(best["conflicting_markers"]) if best else "",
            "note": note,
        })
        detail.append({
            "cf_class": cf_class,
            "self_node_id": self_id,
            "cf_marker_directions": cf_dirs,
            "ranked": ranked,
        })

    csv_path = f"{args.out_prefix}.csv"
    json_path = f"{args.out_prefix}.json"

    fieldnames = ["cf_class", "self_node_id", "self_score", "best_match_id",
                  "best_match_name", "best_score", "relation", "shared_markers",
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
