#!/usr/bin/env python3

import os, sys
import pandas as pd

# Static variable inherited from Nextflow Config
# quantType = '${params.qupath_object_type}'

def compare_headers(pickle_files):
	# Read the DataFrames from pickle files
	dataframes = [pd.read_pickle(file) for file in pickle_files]
	dataframe_names = [os.path.basename(file).replace('.pkl','').replace('merged_dataframe_','') for file in pickle_files]
	dataframes = [df.filter(regex='(Mean)',axis=1) for df in dataframes]

	# Extract the headers (column names)
	headers = [set( [h.split(":")[0] for h in df.columns]) for df in dataframes]
	print(headers)
	#headers = list(set([h.split(":")[0] for h in list(headers]))

	# Get the union of all headers
	all_headers = set.union(*headers)

	# Create a dictionary to store the presence/absence table
	presence_dict = {header: [] for header in all_headers}

	# Fill the dictionary with 1 (present) or 0 (absent)
	for header in all_headers:
		for df_headers in headers:
			presence_dict[header].append(1 if header in df_headers else 0)

	# Create a DataFrame from the presence dictionary
	presence_df = pd.DataFrame(presence_dict, index=dataframe_names).T

	# Save the presence DataFrame to a CSV file
	presence_df.to_csv('panel_design.csv')

if __name__ == "__main__":
	pickle_files = "${tables_pkl_collected}".split(' ')
	compare_headers(pickle_files)

