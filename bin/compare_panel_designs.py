#!/usr/bin/env python3

import os
import sys
import pandas as pd
import json
import argparse
from collections import defaultdict

def compare_headers(pickle_files, output_file, metrics_output, threshold, class_column):
    # Read DataFrames and extract filtered headers
    headers = []
    dataframe_names = []

    # Metric collection:
    global_rois = set()
    global_total_cells = 0
    global_labeled_cells = 0
    global_label_counts = defaultdict(int)


    for file in pickle_files:
        df = pd.read_pickle(file)
        filtered_cols = [h.split(":")[0] for h in df.filter(regex='(Mean)', axis=1).columns]
        headers.append(set(filtered_cols))

        #Assert that this df only has 1 unique batchID
        assert df["original_batchID"].nunique(dropna=False) == 1
        base = str(df["original_batchID"].dropna().unique().item())
        dataframe_names.append(base)

        # Collect metrics
        global_rois.update(df["Image"].dropna().unique())
        global_total_cells += len(df)
        global_labeled_cells += df[class_column].notna().sum()
        for label, count in df[class_column].value_counts().items():
            global_label_counts[label] += count

    # Union of all headers
    all_headers = sorted(set.union(*headers))

    # Build presence/absence DataFrame
    presence_matrix = [
        [1 if header in df_headers else 0 for df_headers in headers]
        for header in all_headers
    ]
    presence_df = pd.DataFrame(presence_matrix, index=all_headers, columns=dataframe_names)

    # If duplicate columns (batch names), sum them
    presence_df = presence_df.groupby(level=0, axis=1).sum()

    # Save to CSV
    presence_df.to_csv(output_file)

    # Build label table rows
    rows = []
    for label, count in sorted(global_label_counts.items(), key=lambda x: -x[1]):
        pct = (count / global_labeled_cells * 100) if global_labeled_cells > 0 else 0
        if pct < 5:
            balance = "Severely Underrepresented"
        elif pct < 10:
            balance = "Underrepresented"
        elif pct > 40:
            balance = "Dominant"
        else:
            balance = "Balanced"
        rows.append({
            "Label": label,
            "Label Count": int(count),
            "Label Percent": round(pct, 2),
            "Class Balance": balance,
            "Status": "Included" if count >= threshold else "Excluded"
        })

    output = {
        "total_batches": len(set(dataframe_names)),
        "total_rois": len(global_rois),
        "total_cells": int(global_total_cells),
        "total_labeled_cells": int(global_labeled_cells),
        "num_unique_labels": len(global_label_counts),
        "label_table": {
            "headers": ["Label", "Label Count", "Label Percent", "Class Balance", "Status"],
            "rows": rows
        }
    }

    with open(metrics_output, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare headers across pickle files and collect batch metrics")
    parser.add_argument("files", nargs="+", help="Pickle files to process")
    parser.add_argument("--output", "-o", default="panel_design.csv", help="Output panel design CSV file (default: panel_design.csv)")
    parser.add_argument("--metrics", "-m", default="batch_metrics.json", help="Output JSON file for metrics (default: batch_metrics.json)")
    parser.add_argument("--threshold", "-t", type=int, default=100, help="Minimum cell count for a label to be used (default: 100)")
    parser.add_argument("--class_col", "-c", default="Classification", help="Name of the column that contains the class labels")

    args = parser.parse_args()

    compare_headers(args.files, args.output, args.metrics, args.threshold, args.class_col)