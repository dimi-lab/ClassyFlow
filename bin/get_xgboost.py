#!/usr/bin/env python3

import os, sys, re, random, math, time, glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
import pickle

from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb

def make_a_new_model(toTrainDF, cv_splits, classColumn, cpu_jobs, depth_d, eta_l, cv_c, uTaskID):

    X = toTrainDF.drop(columns=[classColumn]).select_dtypes(include='number')

    le = preprocessing.LabelEncoder()
    y_Encode = le.fit_transform(toTrainDF[classColumn])
    (unique, counts) = np.unique(y_Encode, return_counts=True)

    num_round = 200
    metricModel = []

    c = int(cv_c)
    if c >= len(cv_splits):
        raise ValueError(f"cv_c={c} is out of bounds for cv_splits with length {len(cv_splits)}")
    
    train_idx, test_idx = cv_splits[c]

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_Encode[train_idx], y_Encode[test_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    print("Training: {} Cells.   Test {} Cells.  Total Features: {}".format(X_train.shape[0], X_test.shape[0], X_train.shape[1]))

    d = int(depth_d)
    l = float(eta_l)

    start = time.time()
    param = {'max_depth': d, 'eta': l, 'objective': 'multi:softprob', 'n_jobs': cpu_jobs,
             'num_class': len(unique), 'eval_metric': 'mlogloss'}

    bst = xgb.train(param, dtrain, num_round)

    # Get class probabilities and convert to class predictions
    predTrain_probs = bst.predict(dtrain)
    predTrain = np.argmax(predTrain_probs, axis=1)
    GBCmpredTrain = le.inverse_transform(predTrain)
    yLabelTrain = le.inverse_transform(np.array(y_train, dtype=np.int32))
    accuracyTrain = accuracy_score(yLabelTrain, GBCmpredTrain)

    preds_probs = bst.predict(dtest)
    preds = np.argmax(preds_probs, axis=1)
    GBCmpred = le.inverse_transform(preds)
    yLabelTest = le.inverse_transform(np.array(y_test, dtype=np.int32))
    accuracy = accuracy_score(yLabelTest, GBCmpred)
    
    metricModel.append({'cv': c, 'max_depth': d, 'eta': l, 'Training': "%.2f%%" % (accuracyTrain * 100.0),
                        'Test': "%.2f%%" % (accuracy * 100.0), 'testf': accuracy})
    end = time.time()
    print("XGB CPU Time %.2f" % (end - start))

    xgboostParams = pd.DataFrame(metricModel)
    rnd = random.randint(1000, 9999)
    xgboostParams.to_csv(f"parameters_found_{rnd}_{uTaskID}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost model with specified parameters.")
    parser.add_argument('--classColumn', required=True, help='Name of the classified column')
    parser.add_argument('--cpu_jobs', type=int, default=16, help='Number of CPU jobs to use')
    parser.add_argument('--uTaskID', required=True, help='Unique task ID')
    parser.add_argument('--depth_d', required=True, help='XGBoost max_depth')
    parser.add_argument('--eta_l', required=True, help='XGBoost eta (learning rate)')
    parser.add_argument('--cv_c', required=True, help='Cross-validation chunk index')
    parser.add_argument('--trainingDataframe', required=True, help='Path to training dataframe pickle')
    parser.add_argument('--cv_splits', required=True, help='CV split indices')

    args = parser.parse_args()

    focusData = pd.read_pickle(args.trainingDataframe)

    splitCVs = pd.read_pickle(args.cv_splits)
    
    make_a_new_model(
        focusData,
        splitCVs,
        args.classColumn,
        args.cpu_jobs,
        args.depth_d,
        args.eta_l,
        args.cv_c,
        args.uTaskID
    )