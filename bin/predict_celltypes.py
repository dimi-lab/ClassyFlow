#!/usr/bin/env python3

import os, sys
import pickle
import argparse
import xgboost as xgb
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint

## Static Variables: File Formatting
# columnsToExport and cpu_jobs will be set from command line arguments

def predict_on_xgb_best_model(toCheckDF, xgbM, bID, leEncoderFile, columnsToExport, 
                               include_probabilities=False, top_n_probs=3, cpu_jobs=1):
    le = preprocessing.LabelEncoder()
    le.classes_ = np.load(leEncoderFile, allow_pickle=True)
    
    # Drop 'index' column if present
    if 'index' in toCheckDF.columns:
        print("[DEBUG] Dropping 'index' column from DataFrame.")
        toCheckDF = toCheckDF.drop(columns=['index'])
    
    print("[DEBUG] Model expects features:", xgbM.feature_names)
    toGetDataFrame = toCheckDF[xgbM.feature_names]
    print("[DEBUG] DataFrame used for prediction head:\n", toGetDataFrame.head())
    
    # Make predictions 
    dmatrix = xgb.DMatrix(toGetDataFrame, nthread=cpu_jobs)
    y_pred_proba = xgbM.predict(dmatrix)  # Shape: (n_samples, n_classes)
    
    # Get the predicted class (highest probability)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Convert to class names
    classCellNames = le.inverse_transform(y_pred)
    toCheckDF['CellTypePrediction'] = classCellNames
    
    # Add prediction confidence (max probability)
    max_probabilities = np.max(y_pred_proba, axis=1)
    toCheckDF['PredictionConfidence'] = max_probabilities
    
    # Optionally add top N class probabilities
    if include_probabilities:
        # Get top N class indices for each sample
        top_n_indices = np.argsort(y_pred_proba, axis=1)[:, -top_n_probs:][:, ::-1]
        
        for rank in range(top_n_probs):
            # Class names
            top_class_indices = top_n_indices[:, rank]
            top_class_names = le.inverse_transform(top_class_indices)
            toCheckDF[f'Top{rank+1}_Class'] = top_class_names
            
            # Probabilities
            top_probs = y_pred_proba[np.arange(len(y_pred_proba)), top_class_indices]
            toCheckDF[f'Top{rank+1}_Probability'] = top_probs
    
    # Prepare columns to export
    export_columns = columnsToExport.copy()
    if 'CellTypePrediction' not in export_columns:
        export_columns.append('CellTypePrediction')
    if 'PredictionConfidence' not in export_columns:
        export_columns.append('PredictionConfidence')
    
    if include_probabilities:
        for rank in range(top_n_probs):
            if f'Top{rank+1}_Class' not in export_columns:
                export_columns.append(f'Top{rank+1}_Class')
            if f'Top{rank+1}_Probability' not in export_columns:
                export_columns.append(f'Top{rank+1}_Probability')
    
    # Filter to only existing columns
    export_columns = [col for col in export_columns if col in toCheckDF.columns]
    toExport = toCheckDF[export_columns]
    
    # Export per image
    for img in toExport['Image'].unique():
        rand_suffix = str(randint(10000, 99999))
        outFh = os.path.join(f"{img}_{rand_suffix}_PRED.tsv")
        roiTbl = toExport[toExport['Image'] == img]
        roiTbl.to_csv(outFh, sep="\t", index=False)
        print(f"[INFO] Saved predictions for {img} to {outFh}")
    
    return toCheckDF

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict cell types using a trained XGBoost model.")
    parser.add_argument('--classColumn', required=True, help='Name of the classified column')
    parser.add_argument('--leEncoderFile', required=True, help='Path to label encoder .npy file')
    parser.add_argument('--batchID', required=True, help='Batch ID for output file naming')
    parser.add_argument('--infile', required=True, help='Input pickle or TSV file')
    parser.add_argument('--modelfile', required=True, help='Path to trained model file')
    parser.add_argument('--columnsToExport', required=True, help='Comma-separated list of columns to export')
    parser.add_argument('--cpu_jobs', type=int, default=16, help='Number of CPU jobs to use')
    parser.add_argument('--include_probabilities', action='store_true', 
                       help='Include top N class probabilities in output')
    parser.add_argument('--top_n_probs', type=int, default=3, 
                       help='Number of top class probabilities to include (default: 3)')

    args = parser.parse_args()

    classColumn = args.classColumn
    leEncoderFile = args.leEncoderFile
    batchID = args.batchID
    infile = args.infile
    modelfile = args.modelfile
    columnsToExport = [col.strip() for col in args.columnsToExport.split(',')]
    cpu_jobs = args.cpu_jobs

    # Load data
    if infile.endswith('.pkl'):
        myData = pd.read_pickle(infile)
    else:
        myData = pd.read_csv(infile, sep='\t', low_memory=False)

    # Load model
    with open(modelfile, 'rb') as file:
        xgbMdl = pickle.load(file)

    # Make predictions
    result_df = predict_on_xgb_best_model(
        myData, 
        xgbMdl, 
        batchID, 
        leEncoderFile, 
        columnsToExport,
        include_probabilities=args.include_probabilities,
        top_n_probs=args.top_n_probs,
        cpu_jobs=cpu_jobs
    )
    
    print(f"[INFO] Prediction complete for batch {batchID}")