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
        dataframe_names.append(os.path.basename(file).replace('.pkl','').replace('merged_dataframe_',''))

    # Union of all headers
    all_headers = sorted(set.union(*headers))

    # Build presence/absence DataFrame using a list comprehension
    presence_matrix = [
        [1 if header in df_headers else 0 for df_headers in headers]
        for header in all_headers
    ]
    presence_df = pd.DataFrame(presence_matrix, index=all_headers, columns=dataframe_names)

    # Save to CSV
    presence_df.to_csv('panel_design.csv')

if __name__ == "__main__":
    compare_headers(sys.argv[1:])