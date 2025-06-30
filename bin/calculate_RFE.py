#!/usr/bin/env python3

import os, sys, re, csv, warnings, random, string
import argparse
import pandas as pd
import numpy as np
from pprint import pprint

from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_val_score

def calculate_rfe_of_n(df, celltype, a, idx, n_splits, n_folds, lasso_max_iteration, parallel_cpus):
    XAll = df[list(df.select_dtypes(include=[np.number]).columns.values)]
    XAll = XAll[XAll.columns.drop(list(XAll.filter(regex='(Centroid|Binary|cnt|Name|Cytoplasm)')))].fillna(0)
    yAll = df['Lasso_Binary']

    print(f"Starting task {idx}")
    rfe = RFE(estimator=Lasso(), n_features_to_select=idx)
    model = Lasso(alpha=a, max_iter=lasso_max_iteration)
    pipeline = Pipeline(steps=[('s', rfe), ('m', model)])
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_folds, random_state=1)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        scores = cross_val_score(
            pipeline, XAll, yAll,
            scoring='neg_root_mean_squared_error',
            cv=cv, n_jobs=parallel_cpus, error_score='raise'
        )

    searchAlphas = pd.DataFrame({
        'rfe_score': scores
    })
    searchAlphas['n_features'] = idx
    random_5_letters = ''.join(random.choice(string.ascii_letters) for _ in range(5))
    clean_celltype = re.sub(r'[^a-zA-Z0-9]', '', celltype)
    searchAlphas.to_csv(f"rfe_scores_{clean_celltype}_{idx}_{random_5_letters}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate RFE scores for Lasso model.")
    parser.add_argument('--binary_dataframe', required=True, help='Path to binary dataframe pickle')
    parser.add_argument('--celltype', required=True, help='Cell type label')
    parser.add_argument('--best_alpha', type=float, required=True, help='Best alpha value for Lasso')
    parser.add_argument('--n_feats', type=int, required=True, help='Number of features to select')
    parser.add_argument('--n_splits', type=int, default=4, help='Number of splits for cross-validation')
    parser.add_argument('--n_folds', type=int, default=9, help='Number of folds for cross-validation')
    parser.add_argument('--lasso_max_iteration', type=int, default=1000, help='Max iterations for Lasso')
    parser.add_argument('--parallel_cpus', type=int, default=8, help='Number of CPUs for parallel jobs')

    args = parser.parse_args()

    myData = pd.read_pickle(args.binary_dataframe)
    myLabel = args.celltype.replace('[', '').replace(']', '')

    calculate_rfe_of_n(
        myData,
        myLabel,
        args.best_alpha,
        args.n_feats,
        args.n_splits,
        args.n_folds,
        args.lasso_max_iteration,
        args.parallel_cpus
    )

