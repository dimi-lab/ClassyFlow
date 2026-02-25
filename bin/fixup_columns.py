#!/usr/bin/env python3

import pandas as pd
import sys
import re
import os
import json
import yaml
import argparse

# --- Helper functions ---
def load_column_map(map_path):
    with open(map_path, 'r') as f:
        config = yaml.safe_load(f)
    remove_columns = config.get('remove_columns', [])
    rename_columns = config.get('rename_columns', {})
    marker_map = config.get('marker_map', {})
    return remove_columns, rename_columns, marker_map

def rename_columns_func(df, rename_columns):
    # Rename whole columns
    return df.rename(columns={k: v for k, v in rename_columns.items() if k in df.columns})

def rename_marker(df, marker_map):
    # marker_map: {old_marker: new_marker}
    newcols = []
    for col in df.columns:
        replaced = False
        for old, new in marker_map.items():
            if col.startswith(old + ":"):
                newcols.append(new + col[len(old):])
                replaced = True
                break
        if not replaced:
            newcols.append(col)
    df.columns = newcols
    return df

def remove_columns_func(df, remove_columns):
    # Remove columns entirely
    for col in remove_columns:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def main(input_path, map_path):
    # Read the pickle file
    df = pd.read_pickle(input_path)
    # Load config from YAML
    remove_columns, rename_columns, marker_map = load_column_map(map_path)

    # Track modifications
    removed = [col for col in remove_columns if col in df.columns]
    original_cols = list(df.columns)

    # Remove columns
    df = remove_columns_func(df, remove_columns)
    after_remove_cols = list(df.columns)

    # Rename whole columns
    renamed = [k for k, v in rename_columns.items() if k in after_remove_cols]
    df = rename_columns_func(df, rename_columns)
    after_rename_cols = list(df.columns)

    # Rename markers
    before_marker_cols = list(df.columns)
    df = rename_marker(df, marker_map)
    after_marker_cols = list(df.columns)
    marker_renamed = []
    for before, after in zip(before_marker_cols, after_marker_cols):
        if before != after:
            marker_renamed.append(f"{before} -> {after}")

    # Save back to the same file
    input_path = re.sub(r'\.pkl$', '_fx.pkl', input_path)
    df.to_pickle(input_path)

    print(f"[INFO] Processed columns and saved updated DataFrame to {input_path}")
    print(f"[INFO] DataFrame shape: {df.shape}")
    print("[INFO] Modifications:")
    if removed:
        print(f"  Removed columns: {removed}")
    else:
        print("  No columns removed.")
    if renamed:
        print(f"  Renamed columns: {renamed}")
    else:
        print("  No columns renamed.")
    if marker_renamed:
        print(f"  Marker renames: {marker_renamed}")
    else:
        print("  No marker columns renamed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix up columns in a DataFrame using a YAML mapping file.")
    parser.add_argument('--input_table', type=str, help='Input pickle file (.pkl)')
    parser.add_argument('--rename_yaml', type=str, help='YAML file with column rename/remove/map info')
    # For backward compatibility, allow positional arguments
    parser.add_argument('input_table_pos', nargs='?', help='Input pickle file (.pkl)')
    parser.add_argument('rename_yaml_pos', nargs='?', help='YAML file with column rename/remove/map info')
    args = parser.parse_args()

    # Prefer named arguments, fallback to positional
    input_path = args.input_table if args.input_table else args.input_table_pos
    map_path = args.rename_yaml if args.rename_yaml else args.rename_yaml_pos

    if not input_path or not map_path:
        print("Usage: python fixup_columns.py --input_table <input.pkl> --rename_yaml <map.yaml>")
        print("   or: python fixup_columns.py <input.pkl> <map.yaml>")
        sys.exit(1)
    main(input_path, map_path)
