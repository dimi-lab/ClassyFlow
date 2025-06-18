#!/usr/bin/env python3

import os, sys, csv, re, warnings, random, string
import argparse
import pandas as pd
import numpy as np
from pprint import pprint

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def format_floats(float_list):
    if len(float_list) < 2:
        raise ValueError("The list must contain at least two float numbers.")
    first_float = float_list[0]
    last_float = float_list[-1]
    first_float_rounded = round(first_float, 4)
    last_float_rounded = round(last_float, 4)
    formatted_string = f"{first_float_rounded}-{last_float_rounded}"
    return formatted_string

def grid_search_alpha_set(df, celltype, alphas, n_folds):
    XAll = df[list(df.select_dtypes(include=[np.number]).columns.values)]
    XAll = XAll[XAll.columns.drop(list(XAll.filter(regex='(Centroid|Binary|cnt|Name)')))].fillna(0)
    yAll = df['Lasso_Binary']
    X_train, X_test, y_train, y_test = train_test_split(XAll, yAll, test_size=0.33, random_state=101, stratify=yAll)
    
    pprint(alphas)
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
    
    random_5_letters = ''.join(random.choice(string.ascii_letters) for _ in range(5))
    clean_celltype = re.sub(r'[^a-zA-Z0-9]', '', celltype)
    searchAlphas.to_csv("alphas_params_{}_{}_{}.csv".format(clean_celltype, format_floats(alphas),random_5_letters), index=False)     

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid search for best Lasso alpha.")
    parser.add_argument('--logspace_chunk', required=True, help='Comma-separated list of alpha values')
    parser.add_argument('--binary_dataframe', required=True, help='Path to binary dataframe pickle')
    parser.add_argument('--celltype', required=True, help='Cell type label')
    parser.add_argument('--n_folds', type=int, default=10, help='Number of cross-validation folds')
    args = parser.parse_args()

    myData = pd.read_pickle(args.binary_dataframe)
    myLabel = args.celltype.replace('[', '').replace(']', '')
    logspace = [float(e) for e in args.logspace_chunk.split(',')]
    n_folds = args.n_folds

    grid_search_alpha_set(myData, myLabel, logspace, n_folds)


