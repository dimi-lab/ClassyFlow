#!/usr/bin/env python3


import os, sys, csv, warnings, random, string
import pandas as pd
import numpy as np
from pprint import pprint




def calculate_rfe_of_n(df, celltype, best_alpha, idx):

	XAll = df[list(df.select_dtypes(include=[np.number]).columns.values)]
	XAll = XAll[XAll.columns.drop(list(XAll.filter(regex='(Centroid|Binary|cnt|Name)')))].fillna(0)
	yAll = df['Lasso_Binary']
	X_train, X_test, y_train, y_test = train_test_split(XAll, yAll, test_size=0.33, random_state=101, stratify=yAll)
	
	
	print(f"Starting task {idx}")
	rfe = RFE(estimator=Lasso(), n_features_to_select=idx)
	model = Lasso(alpha=a, max_iter=lasso_max_iteration)
	pipeline = Pipeline(steps=[('s',rfe),('m',model)])
	cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_folds, random_state=1)

	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", category=ConvergenceWarning)
		scores = cross_val_score(pipeline, x, y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')
	return idx, scores
	



if __name__ == "__main__":
	myData = pd.read_pickle("${binary_dataframe}")
	myLabel = "${celltype}".replace('[', '').replace(']', '')  ### figure out why this passes an array from nextflow...??
	best_alpha=float("${best_alpha}")
	num_of_features=int("${n_feats}")


	calculate_rfe_of_n(myData, myLabel, best_alpha, num_of_features)

