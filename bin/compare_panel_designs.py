#!/usr/bin/env python3

import os
import sys
import pandas as pd

def compare_headers(pickle_files):
    # Read DataFrames and extract filtered headers
    headers = []
    dataframe_names = []
    for file in pickle_files:
        df = pd.read_pickle(file)
        filtered_cols = [h.split(":")[0] for h in df.filter(regex='(Mean)', axis=1).columns]
        headers.append(set(filtered_cols))
        # Remove "-#####" suffix if present
        base = os.path.basename(file).replace('.pkl','').replace('merged_dataframe_','')
        if len(base) > 6 and base[-6] == '-' and base[-5:].isalnum():
            base = base[:-6]
        dataframe_names.append(base)

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
    presence_df.to_csv('panel_design.csv')

if __name__ == "__main__":
    compare_headers(sys.argv[1:])