#!/usr/bin/env python3

import os, sys, re, csv, time, warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import dataframe_image as dfi

from sklearn.feature_selection import VarianceThreshold

classColumn = 'Classification'
batchColumn = 'Batch'
varThreshold = 0.01
mim_class_label_threshold = ${params.minimum_label_count}
ifSubsetData=True

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
		if totCls < 1000:
			df1 = df[df["Lasso_Binary"] == 1]
		else:
			df1 = df[df["Lasso_Binary"] == 1].sample( n=1000 )
			
		negN = totRow - totCls
		if negN < 1000:
			df2 = df[df["Lasso_Binary"] == 0]
		else:
			df2 = df[df["Lasso_Binary"] == 0].sample( n=1000 )
			
		df = pd.concat([df1,df2])
	
	# Remove all non-alphanumeric characters
	clean_celltype = re.sub(r'[^a-zA-Z0-9]', '', celltype)
	df.to_pickle('binary_df_{}.pkl'.format(clean_celltype))
	
	## too big to plot
	print(df.groupby([batchColumn, 'Lasso_Binary']).size() )
	binaryCntTbl = df.groupby([batchColumn, 'Lasso_Binary']).size().reset_index()
	styled_df = binaryCntTbl.style.format({'Batches': "{}",
                      'Binary': "{:,}",
                      'Frequency': "{:,}"}).hide()
	dfi.export(styled_df, 'binary_count_table_{}.png'.format(celltype), table_conversion='matplotlib')
	
	XAll = df[list(df.select_dtypes(include=[np.number]).columns.values)]
	XAll = XAll[XAll.columns.drop(list(XAll.filter(regex='(Centroid|Binary|cnt|Name)')))].fillna(0)
	yAll = df['Lasso_Binary']

	# using sklearn variancethreshold to find constant features
	sel = VarianceThreshold(threshold=varThreshold)
	sel.fit(XAll)  # fit finds the features with zero variance
	# get_support is a boolean vector that indicates which features are retained
	# if we sum over get_support, we get the number of features that are not constant
	sum(sel.get_support())

	# print the constant features
	nonVarFeatures = [x for x in XAll.columns if x not in XAll.columns[sel.get_support()]]
	print("NonVariant Features: "+', '.join(nonVarFeatures))
	nvfDf = pd.DataFrame(nonVarFeatures, columns=['nonVarFeatures'])
	nvfDf.to_csv("non_variant_features_{}.csv".format(celltype)) 	



if __name__ == "__main__":
	myData = pd.read_pickle("${trainingDataframe}")
	myLabel = "${celltype}".replace('[', '').replace(']', '')  ### figure out why this passes an array from nextflow...??
	#myLabel = myLabel = "[B cell|T reg]".replace('[', '').replace(']', '')	

	hshResults = split_and_binarize(myData, myLabel)





