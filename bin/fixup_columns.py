#!/usr/bin/env python3

import pandas as pd
import sys
import re
import yaml
import argparse

# --- Helper functions ---
def load_marker_vocab(vocab_path):
    """Load the canonical marker vocabulary (assets/markers.yaml).

    Returns (canonical, alias_to_canonical, drop):
      canonical            - set of canonical marker names.
      alias_to_canonical   - {raw_alias: canonical_name}.
      drop                 - list of whole columns to remove outright.
    """
    with open(vocab_path, 'r') as f:
        config = yaml.safe_load(f) or {}
    markers = config.get('markers') or {}
    drop = config.get('drop') or []

    canonical = set(markers.keys())
    alias_to_canonical = {}
    for canon, spec in markers.items():
        for alias in (spec or {}).get('aliases', []) or []:
            alias_to_canonical[alias] = canon
    return canonical, alias_to_canonical, drop

def dedupe_renamed_columns(df, provenance):
    """Resolve duplicate column names produced by a rename step.

    ``provenance`` is a list of (original_name, new_name) in column order. When a
    rename maps an alias onto a name that already exists (e.g. an alias turns
    ``DAPI_AF_R01`` into ``DAPI`` while a real ``DAPI`` column is present), the
    naive rename yields duplicate columns that crash every downstream stage that
    does ``df[col]`` (it returns a DataFrame, not a Series).

    For each duplicated name we keep a single column, preferring the one that was
    ALREADY named that (the canonical/original column) and dropping the
    alias-derived duplicates. Returns (df, dropped) where ``dropped`` lists the
    "original -> new" renames that were discarded.
    """
    names = [new for _, new in provenance]
    dup_names = {n for n in names if names.count(n) > 1}
    if not dup_names:
        return df, []

    keep_positions = []
    dropped = []
    for name in dict.fromkeys(names):  # preserve order, unique names
        positions = [i for i, n in enumerate(names) if n == name]
        if len(positions) == 1:
            keep_positions.append(positions[0])
            continue
        # Prefer a column that was already canonically named `name`.
        canonical = [i for i in positions if provenance[i][0] == name]
        keep = canonical[0] if canonical else positions[0]
        keep_positions.append(keep)
        for i in positions:
            if i != keep:
                dropped.append(f"{provenance[i][0]} -> {provenance[i][1]}")

    df = df.iloc[:, sorted(keep_positions)]
    return df, dropped


def resolve_markers(df, canonical, alias_to_canonical):
    """Rewrite marker tokens to their canonical name.

    A marker/measurement column has the shape "<marker>: <compartment>: <stat>"
    (marker token = the part before the first ':'). Its token is resolved:
      1. exact canonical match -> kept as-is.
      2. explicit alias        -> rewritten to canonical, preserving the suffix.
      3. unresolved            -> left untouched and reported to the caller.
    Columns without a ':' (Image, Classification, Centroid ...) are metadata and
    pass through untouched.

    Returns (df, unresolved) where ``unresolved`` is the sorted list of distinct
    marker tokens that did not resolve.
    """
    newcols = []
    unresolved = set()
    for col in df.columns:
        if ":" not in col:
            newcols.append(col)
            continue
        token = col.split(":", 1)[0].strip()
        if token in canonical:
            newcols.append(col)
        elif token in alias_to_canonical:
            canon = alias_to_canonical[token]
            newcols.append(canon + col[len(token):])
        else:
            newcols.append(col)
            unresolved.add(token)
    df.columns = newcols
    return df, sorted(unresolved)

def remove_columns_func(df, drop):
    # Remove columns entirely
    for col in drop:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def main(input_path, vocab_path, strict=False):
    # Read the pickle file
    df = pd.read_pickle(input_path)
    # Load the canonical marker vocabulary from YAML
    canonical, alias_to_canonical, drop = load_marker_vocab(vocab_path)

    # Track modifications
    removed = [col for col in drop if col in df.columns]

    # Remove dropped columns
    df = remove_columns_func(df, drop)
    after_remove_cols = list(df.columns)

    # Resolve marker tokens to canonical names
    df, unresolved = resolve_markers(df, canonical, alias_to_canonical)
    after_marker_cols = list(df.columns)
    marker_renamed = [f"{b} -> {a}"
                      for b, a in zip(after_remove_cols, after_marker_cols)
                      if b != a]

    # An alias can map onto a name that already exists (e.g. DAPI_AF_R01 -> DAPI
    # when a real DAPI is present), producing duplicate columns that crash
    # downstream `df[col]` access. Collapse them, keeping the canonical/original
    # column and dropping the alias duplicate.
    provenance = list(zip(after_remove_cols, after_marker_cols))
    df, dropped_duplicates = dedupe_renamed_columns(df, provenance)

    # Save alongside the input
    output_path = re.sub(r'\.pkl$', '_fx.pkl', input_path)
    df.to_pickle(output_path)

    print(f"[INFO] Processed columns and saved updated DataFrame to {output_path}")
    print(f"[INFO] DataFrame shape: {df.shape}")
    print("[INFO] Modifications:")
    if removed:
        print(f"  Removed columns: {removed}")
    else:
        print("  No columns removed.")
    if marker_renamed:
        print(f"  Marker renames: {marker_renamed}")
    else:
        print("  No marker columns renamed.")
    if dropped_duplicates:
        print(f"  Dropped duplicate columns from rename collisions "
              f"(kept canonical): {dropped_duplicates}")

    if unresolved:
        msg = (f"Unresolved markers (no canonical/alias match in the vocabulary): "
               f"{unresolved}. Add them to assets/markers.yaml as a canonical "
               f"marker or an alias, or list them under `drop`.")
        if strict:
            print(f"[ERROR] {msg}")
            sys.exit(1)
        print(f"[WARNING] {msg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Harmonize marker columns to a canonical vocabulary (markers.yaml).")
    parser.add_argument('--input_table', type=str, help='Input pickle file (.pkl)')
    parser.add_argument('--marker_vocabulary', type=str,
                        help='Canonical marker vocabulary YAML (assets/markers.yaml)')
    parser.add_argument('--strict', action='store_true',
                        help='Fail (non-zero exit) on any unresolved marker instead of warning.')
    # For backward compatibility, allow positional arguments
    parser.add_argument('input_table_pos', nargs='?', help='Input pickle file (.pkl)')
    parser.add_argument('marker_vocabulary_pos', nargs='?',
                        help='Canonical marker vocabulary YAML')
    args = parser.parse_args()

    # Prefer named arguments, fallback to positional
    input_path = args.input_table if args.input_table else args.input_table_pos
    vocab_path = args.marker_vocabulary if args.marker_vocabulary else args.marker_vocabulary_pos

    if not input_path or not vocab_path:
        print("Usage: fixup_columns.py --input_table <input.pkl> --marker_vocabulary <markers.yaml> [--strict]")
        print("   or: fixup_columns.py <input.pkl> <markers.yaml>")
        sys.exit(1)
    main(input_path, vocab_path, strict=args.strict)
