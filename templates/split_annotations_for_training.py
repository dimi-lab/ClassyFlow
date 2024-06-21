#!/usr/bin/env python3

import os, sys
import pandas as pd

## Static Variables: File Formatting
classColumn = 'Classification'
batchColumn = 'Batch'
holdoutFraction = 0.1
cellTypeNegative = '0'

def gather_annotations(pickle_files):
	# Read the DataFrames from pickle files
	dataframes = []
	for file in pickle_files:
		df = pd.read_pickle(file)
		dataframe_name = os.path.basename(file).replace('.pkl','').replace('merged_dataframe_','')
		df[batchColumn] = dataframe_name
		dataframes.append(df)
	
	# Concatenate all DataFrames
	merged_df = pd.concat(dataframes, ignore_index=True)
	# merged_df = merged_df.sample(n=5000)  # remove this after testing
	merged_df[classColumn] = merged_df[classColumn].str.strip()
	merged_df = merged_df.dropna(subset=[classColumn])
	merged_df = merged_df.loc[~(merged_df[classColumn] in ["", "??", "?", cellTypeNegative])]
	merged_df = merged_df.reset_index()

	ct = merged_df[classColumn].value_counts()
	pt = merged_df[classColumn].value_counts(normalize=True).mul(100).round(2).astype(str) + '%'
	freqTable = pd.concat([ct,pt], axis=1, keys=['counts', '%'])
	print(freqTable)
	
	ctl = merged_df[classColumn].unique().tolist()
	with open("celltypes.csv", newline='') as csvfile:
	    f_writer = csv.writer(csvfile, delimiter=',')
	    f_writer.writerow(ctl)	

	# / Subset to just annotated cells
	
	holdoutDF = merged_df.groupby(batchColumn, group_keys=False).apply(lambda x: x.sample(frac=holdoutFraction))
	trainingDF = merged_df.loc[~merged_df.index.isin(holdoutDF.index)]
	
	holdoutDF = holdoutDF.reset_index()
	holdoutDF.to_pickle('holdout_dataframe.pkl')
	trainingDF = trainingDF.reset_index()
	trainingDF.to_pickle('training_dataframe.pkl')

if __name__ == "__main__":
	#pickle_files = "${norms_pkl_collected}".split(' ')
	pickle_files = "normalized_SET03.pkl normalized_SET04.pkl normalized_SET01.pkl normalized_SET02.pkl".split(' ')
	  	
	gather_annotations(pickle_files)


