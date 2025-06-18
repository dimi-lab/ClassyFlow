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

import fpdf
from fpdf import FPDF
import dataframe_image as dfi
from random import randint

## Static Variables: File Formatting
# columnsToExport and cpu_jobs will be set from command line arguments

############################ PDF REPORTING ############################
def create_letterhead(pdf, WIDTH, letterhead_path):
    pdf.image(letterhead_path, 0, 0, WIDTH)     

def create_title(title, pdf):
    # Add main title
    pdf.set_font('Helvetica', 'b', 20)  
    pdf.ln(40)
    pdf.write(5, title)
    pdf.ln(10)
    # Add date of report
    pdf.set_font('Helvetica', '', 14)
    pdf.set_text_color(r=128,g=128,b=128)
    today = pd.Timestamp.now().strftime("%d/%m/%Y")
    pdf.write(4, f'{today}')
    # Add line break
    pdf.ln(10)

def write_to_pdf(pdf, words):
    # Set text colour, font size, and font type
    pdf.set_text_color(r=0,g=0,b=0)
    pdf.set_font('Helvetica', '', 12)
    pdf.write(5, words)
############################ PDF REPORTING ############################

def predict_on_xgb_best_model(toCheckDF, xgbM, bID, leEncoderFile, columnsToExport):
    allPDFText = {}
    le = preprocessing.LabelEncoder()
    le.classes_ = np.load(leEncoderFile, allow_pickle=True)
    toGetDataFrame = toCheckDF[xgbM.feature_names]
    # Make predictions
    dmatrix = xgb.DMatrix(toGetDataFrame)
    y_pred_all = xgbM.predict(dmatrix)
    y_pred_all_int = np.array(y_pred_all, dtype=int)
    classCellNames = le.inverse_transform(y_pred_all_int)
    toCheckDF['CellTypePrediction'] = classCellNames
    toExport = toCheckDF[columnsToExport]
    for img in toExport['Image'].unique():
        roiTbl = toExport[toExport['Image'] == img]
        outFh = os.path.join(img+"_PRED.tsv")
        roiTbl.to_csv(outFh, sep="\t", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict cell types using a trained XGBoost model.")
    parser.add_argument('--classColumn', required=True, help='Name of the classified column')
    parser.add_argument('--leEncoderFile', required=True, help='Path to label encoder .npy file')
    parser.add_argument('--batchID', required=True, help='Batch ID for output file naming')
    parser.add_argument('--infile', required=True, help='Input pickle or TSV file')
    parser.add_argument('--modelfile', required=True, help='Path to trained model file')
    parser.add_argument('--columnsToExport', required=True, help='Comma-separated list of columns to export')
    parser.add_argument('--cpu_jobs', type=int, default=16, help='Number of CPU jobs to use')
    parser.add_argument('--letterhead', required=False, help='Path to letterhead image for PDF report (optional)')

    args = parser.parse_args()

    classColumn = args.classColumn
    leEncoderFile = args.leEncoderFile
    batchID = args.batchID
    infile = args.infile
    modelfile = args.modelfile
    columnsToExport = [col.strip() for col in args.columnsToExport.split(',')]
    cpu_jobs = args.cpu_jobs

    if infile.endswith('.pkl'):
        myData = pd.read_pickle(infile)
    else:
        myData = pd.read_csv(infile, sep='\t', low_memory=False)

    with open(modelfile, 'rb') as file:
        xgbMdl = pickle.load(file)

    predict_on_xgb_best_model(myData, xgbMdl, batchID, leEncoderFile, columnsToExport)






