#!/usr/bin/env python3

import argparse
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import pickle

def create_kfold_cv_splits(toTrainDF, classColumn, n_splits=5):
    #Create and save CV splits once
    y = toTrainDF[classColumn]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_splits = list(skf.split(toTrainDF, y))
    
    # Save to disk
    with open('cv_splits.pkl', 'wb') as f:
        pickle.dump(cv_splits, f)



def create_random_cv_splits(toTrainDF, classColumn, n_splits=5, test_size=0.33, random_state=42):
    y = toTrainDF[classColumn]
    
    # StratifiedShuffleSplit generates random stratified splits
    sss = StratifiedShuffleSplit(
        n_splits=n_splits, 
        test_size=test_size, 
        random_state=random_state
    )
    
    cv_splits = list(sss.split(toTrainDF, y))
    
    # Save to disk
    with open('cv_splits.pkl', 'wb') as f:
        pickle.dump(cv_splits, f)
    
    return cv_splits
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate XGBoost parameter search grid.")
    parser.add_argument('--max_cv', type=int, required=True, help='Maximum number of cross-validation iterations')
    parser.add_argument('--depth_start', type=int, default=2, help='Start of depth range (inclusive)')
    parser.add_argument('--depth_stop', type=int, default=22, help='End of depth range (exclusive)')
    parser.add_argument('--depth_step', type=int, default=4, help='Step size for depth range')
    parser.add_argument('--learnRates', type=str, default="0.1,0.7,1.0", help='Comma-separated list of learning rates')
    parser.add_argument('--classColumn', required=True, help='Name of the classified column')
    parser.add_argument('--trainingDataframe', required=True, help='Path to training dataframe pickle')
    parser.add_argument('--select_features_csv', required=True, help='Path to selected features CSV')
    args = parser.parse_args()

    #Create training df with selected features
    myData = pd.read_pickle(args.trainingDataframe)
    with open(args.select_features_csv, 'r') as file:
        next(file)  # Skip header
        featureList = file.readlines()
    featureList = list(set([line.strip() for line in featureList]))
    if 'level_0' in featureList:
        featureList.remove('level_0')
    featureList.append(args.classColumn)
    focusData = myData[featureList]

    # Save to disk
    with open('toTrainDF.pkl', 'wb') as f:
        pickle.dump(focusData, f)


    #create_kfold_cv_splits(focusData, args.classColumn, args.max_cv)
    create_random_cv_splits(focusData, args.classColumn, args.max_cv)

    max_cv = args.max_cv
    depthFeild = range(args.depth_start, args.depth_stop, args.depth_step)
    learnRates = [float(x) for x in args.learnRates.split(",")]

    with open("xgb_iterate_params.csv", 'w', newline='') as csvfile:
        f_writer = csv.writer(csvfile)
        f_writer.writerow(["CVIDX", "DEPTH", "ETA"])
        for c in range(0, max_cv):
            for d in depthFeild:
                for l in learnRates:
                    f_writer.writerow([c, d, l])


