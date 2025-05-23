#!/usr/bin/env python3


import os, sys, re, csv, warnings, random, string
import pandas as pd
import numpy as np
from pprint import pprint


from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_val_score

n_splits = 4
n_folds = 9
lasso_max_iteration = 1000
#parallel_jobs=-1
parallel_cpus=8


def calculate_rfe_of_n(df, celltype, a, idx):
	XAll = df[list(df.select_dtypes(include=[np.number]).columns.values)]
	XAll = XAll[XAll.columns.drop(list(XAll.filter(regex='(Centroid|Binary|cnt|Name|Cytoplasm)')))].fillna(0)
	yAll = df['Lasso_Binary']
	
	
	print(f"Starting task {idx}")
	rfe = RFE(estimator=Lasso(), n_features_to_select=idx)
	model = Lasso(alpha=a, max_iter=lasso_max_iteration)
	pipeline = Pipeline(steps=[('s',rfe),('m',model)])
	cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_folds, random_state=1)

	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", category=ConvergenceWarning)
		scores = cross_val_score(pipeline, XAll, yAll, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=parallel_cpus, error_score='raise')
	
	#pprint(scores)
	searchAlphas = pd.DataFrame({
		'rfe_score': scores
	}) 
	searchAlphas['n_features'] = idx
	random_5_letters = ''.join(random.choice(string.ascii_letters) for _ in range(5))
	clean_celltype = re.sub(r'[^a-zA-Z0-9]', '', celltype)
	searchAlphas.to_csv("rfe_scores_{}_{}_{}.csv".format(clean_celltype, idx, random_5_letters), index=False) 	


if __name__ == "__main__":
	myData = pd.read_pickle("${binary_dataframe}")
	myLabel = "${celltype}".replace('[', '').replace(']', '')  ### figure out why this passes an array from nextflow...??
	best_alpha= ${best_alpha}
	num_of_features=int("${n_feats}")

	calculate_rfe_of_n(myData, myLabel, best_alpha, num_of_features)

