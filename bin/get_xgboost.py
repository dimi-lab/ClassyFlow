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

import xgboost as xgb

def make_a_new_model(toTrainDF, classColumn, cpu_jobs, mim_class_label_threshold, depth_d, eta_l, cv_c, uTaskID):
    allPDFText = {}
    class_counts = toTrainDF[classColumn].value_counts()
    # Identify classes with fewer than the threshold
    classes_to_keep = class_counts[class_counts > mim_class_label_threshold].index
    # Filter the dataframe to remove these classes
    toTrainDF = toTrainDF[toTrainDF[classColumn].isin(classes_to_keep)]
    X = toTrainDF[list(toTrainDF.select_dtypes(include=[np.number]).columns.values)]

    le = preprocessing.LabelEncoder()
    y_Encode = le.fit_transform(toTrainDF[classColumn])
    (unique, counts) = np.unique(y_Encode, return_counts=True)

    num_round = 200
    metricModel = []
    c = int(cv_c)
    X_train, X_test, y_train, y_test = train_test_split(X, y_Encode, test_size=0.33, stratify=y_Encode)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    print("Training: {} Cells.   Test {} Cells.  Total Features: {}".format(X_train.shape[0], X_test.shape[0], X_train.shape[1]))

    d = int(depth_d)
    l = float(eta_l)
    start = time.time()
    param = {'max_depth': d, 'eta': l, 'objective': 'multi:softmax', 'n_jobs': cpu_jobs,
             'num_class': len(unique), 'eval_metric': 'mlogloss'}
    bst = xgb.train(param, dtrain, num_round)

    predTrain = bst.predict(dtrain)
    GBCmpredTrain = le.inverse_transform(np.array(predTrain, dtype=np.int32))
    yLabelTrain = le.inverse_transform(np.array(y_train, dtype=np.int32))
    accuracyTrain = accuracy_score(yLabelTrain, GBCmpredTrain)

    preds = bst.predict(dtest)
    GBCmpred = le.inverse_transform(np.array(preds, dtype=np.int32))
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
    parser.add_argument('--mim_class_label_threshold', type=int, required=True, help='Minimum label count')
    parser.add_argument('--depth_d', required=True, help='XGBoost max_depth')
    parser.add_argument('--eta_l', required=True, help='XGBoost eta (learning rate)')
    parser.add_argument('--cv_c', required=True, help='Cross-validation chunk index')
    parser.add_argument('--trainingDataframe', required=True, help='Path to training dataframe pickle')
    parser.add_argument('--select_features_csv', required=True, help='Path to selected features CSV')

    args = parser.parse_args()

    myData = pd.read_pickle(args.trainingDataframe)
    with open(args.select_features_csv, 'r') as file:
        next(file)  # Skip header
        featureList = file.readlines()
    featureList = list(set([line.strip() for line in featureList]))
    if 'level_0' in featureList:
        featureList.remove('level_0')
    featureList.append(args.classColumn)
    focusData = myData[featureList]

    make_a_new_model(
        focusData,
        args.classColumn,
        args.cpu_jobs,
        args.mim_class_label_threshold,
        args.depth_d,
        args.eta_l,
        args.cv_c,
        args.uTaskID
    )