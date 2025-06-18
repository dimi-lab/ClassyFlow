#!/usr/bin/env python3

import os, sys, re, random, math, time, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize

import pickle
import xgboost as xgb
import fpdf
from fpdf import FPDF
import dataframe_image as dfi

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
    today = time.strftime("%d/%m/%Y")
    pdf.write(4, f'{today}')
    # Add line break
    pdf.ln(10)

def write_to_pdf(pdf, words):
    # Set text colour, font size, and font type
    pdf.set_text_color(r=0,g=0,b=0)
    pdf.set_font('Helvetica', '', 12)
    pdf.write(5, words)
############################ PDF REPORTING ############################



def check_holdout(toCheckDF, xgbM, classColumn, leEncoderFile):
    allPDFText = {}
    X_holdout = toCheckDF[list(toCheckDF.select_dtypes(include=[np.number]).columns.values)]
    X_holdout = X_holdout[xgbM.feature_names]

    le = preprocessing.LabelEncoder()
    le.classes_ = np.load(leEncoderFile, allow_pickle=True)
    print(le.classes_.tolist())
    y_holdout = le.transform(toCheckDF[classColumn])


    # Make predictions
    dmatrix = xgb.DMatrix(X_holdout)
    y_pred_proba = xgbM.predict(dmatrix)

    allPDFText['accuracy'] = accuracy_score(y_holdout, y_pred_proba)

    """Plot ROC curve for binary or multiclass classification."""
    # Get unique values and their counts
    unique_values, counts = np.unique(y_holdout, return_counts=True)
    n_classes = len(unique_values)
    print(n_classes)
    uniqNames = le.inverse_transform(unique_values)
    plt.barh(uniqNames, counts)
    plt.savefig("label_bars.png", dpi=300, bbox_inches='tight')
    lableHash = dict(zip(unique_values, uniqNames))


    # Calculate confusion matrix
    metrics_df = pd.DataFrame(confusion_matrix(y_holdout, y_pred_proba))
    metrics_df.columns = [lableHash[u] for u in unique_values]
    metrics_df.rename(index=lableHash)
    #pprint(metrics_df)
    styled_df = metrics_df.style.format().hide()
    dfi.export(styled_df, 'crosstab.png', table_conversion='matplotlib')

    # Multiclass case
    y_true_binarized = label_binarize(y_holdout, classes=np.arange(n_classes))
    y_pred_binarized = label_binarize(y_pred_proba, classes=np.arange(n_classes))
    #fpr = dict()
    #tpr = dict()
    #roc_auc = dict()

    #for i in range(n_classes):
    #   fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_binarized[:, i])
    #   roc_auc[i] = auc(fpr[i], tpr[i])

    #plt.figure()
    #colors = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'yellow']
    #for i in range(n_classes):
    #   plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
    #   label=f'{lableHash[i]} (a={roc_auc[i]:.2f})')

    #plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.title('Receiver Operating Characteristic on Holdout')
    #plt.legend(loc="lower right")
    #plt.show()
    #plt.savefig("auc_curve_multiclass.png", dpi=300, bbox_inches='tight')

    # Compute AUC for each class
    auc_scores = {}
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_pred_binarized[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[i] = roc_auc
    # Rank AUC scores
    sorted_auc_scores = sorted(auc_scores.items(), key=lambda x: x[1], reverse=True)
    # Plot AUC for each class
    plt.figure(figsize=(10, 8))
    for i, (class_index, roc_auc) in enumerate(sorted_auc_scores):
        if class_index in lableHash:
            fpr, tpr, _ = roc_curve(y_true_binarized[:, class_index], y_pred_binarized[:, class_index])
            plt.plot(fpr, tpr, lw=2, label=f'{lableHash[class_index]} (AUC = {roc_auc:.2f})')
        else:
            print("Skip {} in ROC curve".format(class_index))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig("auc_curve_stacked.png", dpi=300, bbox_inches='tight')
    
    allPDFText['max_auc'] = sorted_auc_scores[0]
    allPDFText['min_auc'] = sorted_auc_scores[-1]
    
    return allPDFText

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate XGBoost model on holdout data and generate PDF report.")
    parser.add_argument('--classColumn', required=True, help='Name of the classified column')
    parser.add_argument('--leEncoderFile', required=True, help='Path to label encoder .npy file')
    parser.add_argument('--letterhead', required=True, help='Path to letterhead image')
    parser.add_argument('--model_pickle', required=True, help='Path to trained model pickle')
    parser.add_argument('--holdoutDataframe', required=True, help='Path to holdout dataframe pickle')
    parser.add_argument('--select_features_csv', required=True, help='Path to selected features CSV')
    args = parser.parse_args()

    classColumn = args.classColumn
    leEncoderFile = args.leEncoderFile

    myData = pd.read_pickle(args.holdoutDataframe)
    with open(args.select_features_csv, 'r') as file:
        next(file)  # Skip header
        featureList = [line.strip() for line in file if line.strip()]
    if 'level_0' in featureList:
        featureList.remove('level_0')
    featureList.append(classColumn)
    focusData = myData[featureList]
    focusData = focusData.loc[:, ~focusData.columns.duplicated()]

    with open(args.model_pickle, 'rb') as file:
        xgbMdl = pickle.load(file)
    textHsh = check_holdout(focusData, xgbMdl, classColumn, leEncoderFile)

    WIDTH = 215.9
    HEIGHT = 279.4
    pdf = FPDF()
    pdf.add_page()
    create_letterhead(pdf, WIDTH, args.letterhead)
    create_title("Holdout Evaluation: XGBoost", pdf)
    write_to_pdf(pdf, "Holdout Model Accuracy: %.2f%%" % (textHsh['accuracy'] * 100.0))
    pdf.ln(15)
    pdf.image('label_bars.png', w=(WIDTH * 0.85))
    pdf.ln(15)
    pdf.image('crosstab.png', w=(WIDTH * 0.8))
    pdf.ln(15)
    pdf.image('auc_curve_stacked.png', w=(WIDTH * 0.95))
    pdf.output("Holdout_on_{}.pdf".format(os.path.basename(args.model_pickle).replace(".pkl", "")), 'F')

    performanceDF = pd.DataFrame({
        "Model": [args.model_pickle],
        "Accuracy": [textHsh['accuracy']],
        "Max AUC": [textHsh['max_auc'][-1]],
        "Min AUC": [textHsh['min_auc'][-1]]
    })
    performanceDF.to_csv("holdout_{}.csv".format(os.path.basename(args.model_pickle).replace(".pkl", "")), index=False)





