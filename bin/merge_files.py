#!/usr/bin/env python3

import sys, os
import pandas as pd

slide_by_prefix = ${params.slide_contains_prefix}
folder_is_slide = ${params.folder_is_slide}

input_extension="${params.quant_file_extension}"
input_delimiter="${params.quant_file_delimiter}"

def merge_tab_delimited_files(directory_path, excld):
	# List all files in the directory
	files = [f for f in os.listdir(directory_path) if f.endswith(input_extension)]

	# Read and concatenate all tab-delimited files into a single DataFrame
	dataframes = []
	for file in files:
		file_path = os.path.join(directory_path, file)
		df = pd.read_csv(file_path, sep=input_delimiter, low_memory=False)
		#print(file_path) - Used to debug, misformatted files.
		#print(df.columns.tolist())
		if excld != '':
			df = df[df.columns.drop(list(df.filter(regex='('+excld+')')))]
		if slide_by_prefix:
			df['Slide'] = [e.split('_')[0] for e in df['Image'].tolist() ]
		elif folder_is_slide:
			df['Slide'] = directory_path
		else:
			df['Slide'] = file
			
		if folder_is_slide:
			df['Image'] = directory_path+'-'+df['Image']
		dataframes.append(df)

	# Concatenate all DataFrames
	merged_df = pd.concat(dataframes, ignore_index=True)
	# merged_df = merged_df.sample(n=5000)  # remove this after testing
	merged_df = merged_df.reset_index()
	
	## Throw Error if Quant Files are empty.
	if merged_df.shape[0] == 0:
		sys.exit("Merged Input Files result in EMPTY data table: {}".format(directory_path))

	# Save the merged DataFrame as a pickle file
	merged_df.to_pickle('merged_dataframe_${batchID}.pkl')

if __name__ == "__main__":
	directory_path = "${subdir}"
	excludingString = "${exMarks}"
	merge_tab_delimited_files(directory_path, excludingString)

