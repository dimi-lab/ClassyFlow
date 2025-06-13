#!/usr/bin/env python3

import os, sys, re, csv, time, warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import dataframe_image as dfi

classColumn = 'Classification'
batchColumn = 'Batch'
varThreshold = 0.01
mim_class_label_threshold = ${params.minimum_label_count}
ifSubsetData=True
subSet_n=3000

def split_and_binarize(df, celltype):
	df["cnt"]=1
	df["Lasso_Binary"] = 0
	df.loc[df[classColumn] == celltype, 'Lasso_Binary'] = 1
	
	## Skip if too few labels exist
	totCls = df["Lasso_Binary"].sum()
	print("{} has {} lables".format(celltype, totCls))
	if totCls < mim_class_label_threshold:
		print("{} '{}' is not enough class labels to model!".format(totCls, celltype))
		return 	
	
	### Add optional parameter to speed up by reducing data amount. Half of target class size
	if ifSubsetData:
		totRow = df.shape[0]
		if totCls < subSet_n:
			df1 = df[df["Lasso_Binary"] == 1]
		else:
			df1 = df[df["Lasso_Binary"] == 1].sample( n=subSet_n )
			
		negN = totRow - totCls
		if negN < subSet_n:
			df2 = df[df["Lasso_Binary"] == 0]
		else:
			df2 = df[df["Lasso_Binary"] == 0].sample( n=subSet_n )
			
		df = pd.concat([df1,df2])
	
	# Remove all non-alphanumeric characters
	clean_celltype = re.sub(r'[^a-zA-Z0-9]', '', celltype)
	df.to_pickle('binary_df_{}.pkl'.format(clean_celltype))


if __name__ == "__main__":
	myData = pd.read_pickle("${trainingDataframe}")
	myLabel = "${celltype}".replace('[', '').replace(']', '')  ### figure out why this passes an array from nextflow...??
	#myLabel = myLabel = "[B cell|T reg]".replace('[', '').replace(']', '')	

	hshResults = split_and_binarize(myData, myLabel)





