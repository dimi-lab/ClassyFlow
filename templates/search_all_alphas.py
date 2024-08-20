#!/usr/bin/env python3

import os, sys, csv, re, warnings, random, string
import pandas as pd
import numpy as np
from pprint import pprint

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

n_folds = 10 # n_folds = ${params.nnnnnnn}

def format_floats(float_list):
    # Check if the list has at least two elements
    if len(float_list) < 2:
        raise ValueError("The list must contain at least two float numbers.")
    
    # Extract the first and last float numbers
    first_float = float_list[0]
    last_float = float_list[-1]
    
    # Round to 2 decimal points
    first_float_rounded = round(first_float, 4)
    last_float_rounded = round(last_float, 4)
    
    # Convert to string and format as "first-last"
    formatted_string = f"{first_float_rounded}-{last_float_rounded}"
    
    return formatted_string


def grid_search_alpha_set(df, celltype, alphas):
	XAll = df[list(df.select_dtypes(include=[np.number]).columns.values)]
	XAll = XAll[XAll.columns.drop(list(XAll.filter(regex='(Centroid|Binary|cnt|Name)')))].fillna(0)
	yAll = df['Lasso_Binary']
	X_train, X_test, y_train, y_test = train_test_split(XAll, yAll, test_size=0.33, random_state=101, stratify=yAll)
	
	pprint(alphas)
	#alphas = np.arange(0.0002,0.004,0.0003)
	### Add Templating right here.
	#alphas = np.logspace(-5.1,-0.0004, n_alphas_to_search)
	pipeline = Pipeline([
	('scaler',StandardScaler(with_mean=False)),
	('model',Lasso())
	])
	search = GridSearchCV(pipeline,
	{'model__alpha': alphas},
	cv = n_folds, 
	scoring="neg_mean_squared_error",
	verbose=3
	)
	search.fit(X_train,y_train)
	#allPDFText['best_alpha'] = search.best_params_['model__alpha']
	print( "Best Alpha: {}".format( search.best_params_['model__alpha'] ) )

	scores = search.cv_results_["mean_test_score"]
	scores_std = search.cv_results_["std_test_score"]

	searchAlphas = pd.DataFrame({
		'mean_test_score': scores,
		'std_test_score': scores_std
	}) 
	searchAlphas['best_alpha'] = search.best_params_['model__alpha']
	searchAlphas['input_a'] = format_floats(alphas)
	searchAlphas['logspace'] = alphas
	print(searchAlphas)
	
	
	
	
	
	
	
	random_5_letters = ''.join(random.choice(string.ascii_letters) for _ in range(5))
	clean_celltype = re.sub(r'[^a-zA-Z0-9]', '', celltype)
	searchAlphas.to_csv("alphas_params_{}_{}_{}.csv".format(clean_celltype, format_floats(alphas),random_5_letters), index=False) 	



if __name__ == "__main__":
	myData = pd.read_pickle("${binary_dataframe}")
	myLabel = "${celltype}".replace('[', '').replace(']', '')  ### figure out why this passes an array from nextflow...??
	logspace="${logspace_chunk}".split(',')
	logspace = [float(e) for e in logspace]

	grid_search_alpha_set(myData, myLabel, logspace)


