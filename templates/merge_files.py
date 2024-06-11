#!/usr/bin/env python3

import sys, os
import pandas as pd

def merge_tab_delimited_files(directory_path):
	# List all files in the directory
	files = [f for f in os.listdir(directory_path) if f.endswith('.tsv')]

	# Read and concatenate all tab-delimited files into a single DataFrame
	dataframes = []
	for file in files:
		file_path = os.path.join(directory_path, file)
		df = pd.read_csv(file_path, sep='\t')
		dataframes.append(df)

	# Concatenate all DataFrames
	merged_df = pd.concat(dataframes, ignore_index=True)

	# Save the merged DataFrame as a pickle file
	merged_df.to_pickle('merged_dataframe.pkl')

if __name__ == "__main__":
	directory_path = "${subdir}"
	merge_tab_delimited_files(directory_path)

