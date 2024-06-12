#!/usr/bin/env python3

import sys, os
import pandas as pd

slide_by_prefix=True

def merge_tab_delimited_files(directory_path, excld):
	# List all files in the directory
	files = [f for f in os.listdir(directory_path) if f.endswith('.tsv')]

	# Read and concatenate all tab-delimited files into a single DataFrame
	dataframes = []
	for file in files:
		file_path = os.path.join(directory_path, file)
		df = pd.read_csv(file_path, sep='\t')
		if excld != '':
			df = df[df.columns.drop(list(df.filter(regex='('+excld+')')))]
		if slide_by_prefix:
			df['Slide'] = [e.split('_')[0] for e in df['Image'].tolist() ]		
		dataframes.append(df)

	# Concatenate all DataFrames
	merged_df = pd.concat(dataframes, ignore_index=True)
	merged_df = merged_df.reset_index()

	# Save the merged DataFrame as a pickle file
	merged_df.to_pickle('merged_dataframe.pkl')

if __name__ == "__main__":
	directory_path = "${subdir}"
	excludingString = "${exMarks}"
	merge_tab_delimited_files(directory_path, excludingString)

