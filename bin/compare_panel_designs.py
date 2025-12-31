#!/usr/bin/env python3

import os
import sys
import pandas as pd
import json
import argparse
from collections import defaultdict

def compare_headers(pickle_files, output_file, metrics_output, threshold, class_column):
    headers = []
    dataframe_names = []
    batch_metrics = {}
    global_label_counts = defaultdict(int)
    global_labeled_cells = 0

    for file in pickle_files:
        df = pd.read_pickle(file)
        filtered_cols = [h.split(":")[0] for h in df.filter(regex='(Mean)', axis=1).columns]
        headers.append(set(filtered_cols))

        assert df["original_batchID"].nunique(dropna=False) == 1
        base = str(df["original_batchID"].dropna().unique().item())
        dataframe_names.append(base)

        # Per-batch metrics
        batch_metrics[base] = {
            "batch_id": base,
            "num_cells": len(df),
            "num_rois": df["Image"].nunique(),
            "num_features": len(filtered_cols)
        }

        # Global label counts
        global_labeled_cells += df[class_column].notna().sum()
        for label, count in df[class_column].value_counts().items():
            global_label_counts[label] += count

    # Union of all headers
    all_markers = sorted(set.union(*headers))

    # Build presence/absence DataFrame
    presence_matrix = [
        [1 if marker in df_headers else 0 for df_headers in headers]
        for marker in all_markers
    ]
    presence_df = pd.DataFrame(presence_matrix, index=all_markers, columns=dataframe_names)
    presence_df = presence_df.groupby(level=0, axis=1).sum()
    presence_df.to_csv(output_file)

    # Enrich per-batch metrics with missing marker info
    marker_status_matrix = {marker: {} for marker in all_markers}
    
    for batch in presence_df.columns:
        missing_mask = presence_df[batch] == 0
        missing_markers = presence_df.index[missing_mask].tolist()
        missing_count = len(missing_markers)
        
        batch_metrics[batch]["missing_features"] = missing_count
        batch_metrics[batch]["percent_synthetic"] = round(missing_count / len(all_markers) * 100, 1) if all_markers else 0
        batch_metrics[batch]["missing_markers"] = missing_markers
        
        # Build marker status matrix
        for marker in all_markers:
            marker_status_matrix[marker][batch] = "synthetic" if presence_df.loc[marker, batch] == 0 else "real"

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

    # Derive totals from per-batch metrics
    batches_list = list(batch_metrics.values())
    total_cells = sum(b["num_cells"] for b in batches_list)
    total_rois = sum(b["num_rois"] for b in batches_list)
    batches_with_synthetic = sum(1 for b in batches_list if b["missing_features"] > 0)

    output = {
        "total_batches": len(set(dataframe_names)),
        "total_rois": total_rois,
        "total_cells": int(total_cells),
        "total_labeled_cells": int(global_labeled_cells),
        "num_unique_labels": len(global_label_counts),
        "label_table": {
            "headers": ["Label", "Label Count", "Label Percent", "Class Balance", "Status"],
            "rows": rows
        },
        "batches": batches_list,
        "batches_with_synthetic": batches_with_synthetic,
        "all_markers": all_markers,
        "marker_status_matrix": marker_status_matrix
    }

    with open(metrics_output, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare headers across pickle files and collect batch metrics")
    parser.add_argument("files", nargs="+", help="Pickle files to process")
    parser.add_argument("--output", "-o", default="panel_design.csv", help="Output panel design CSV file")
    parser.add_argument("--metrics", "-m", default="batch_metrics.json", help="Output JSON file for metrics")
    parser.add_argument("--threshold", "-t", type=int, default=100, help="Minimum cell count for a label to be used")
    parser.add_argument("--class_col", "-c", default="Classification", help="Column containing class labels")

    args = parser.parse_args()
    compare_headers(args.files, args.output, args.metrics, args.threshold, args.class_col)