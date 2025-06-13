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

## Static Variables: File Formatting
classColumn = "${params.classifed_column_name}" # 'Classification'
leEncoderFile = "${params.output_dir}/models/classes.npy"



############################ PDF REPORTING ############################
def create_letterhead(pdf, WIDTH):
    pdf.image("${projectDir}/images/ClassyFlow_Letterhead.PNG", 0, 0, WIDTH)   

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



def check_holdout(toCheckDF, xgbM):
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
    modelName = "${model_pickle}" #"XGBoost_Model_Second.pkl"
    myData = pd.read_pickle("${holdoutDataframe}") 
    with open("${select_features_csv}", 'r') as file:
        next(file) # Skip header
        featureList = file.readlines()
    featureList = list(set([line.strip() for line in featureList]))
    if 'level_0' in featureList:
        featureList.remove('level_0')
    featureList.append(classColumn)
    focusData = myData[featureList]

    with open(modelName, 'rb') as file:
        xgbMdl = pickle.load(file)
    textHsh = check_holdout(focusData, xgbMdl)
    
    WIDTH = 215.9
    HEIGHT = 279.4
    pdf = FPDF() # A4 (210 by 297 mm)
    pdf.add_page()
    # Add lettterhead and title
    create_letterhead(pdf, WIDTH)
    create_title("Holdout Evaluation: XGBoost", pdf)
    # Add some words to PDF
    write_to_pdf(pdf, "Holdout Model Accuracy: %.2f%%" % (textHsh['accuracy'] * 100.0)) 
    pdf.ln(15)
    pdf.image('label_bars.png', w= (WIDTH*0.85) )
    pdf.ln(15)
    pdf.image('crosstab.png', w= (WIDTH*0.8) )
    pdf.ln(15)
    pdf.image('auc_curve_stacked.png', w= (WIDTH*0.95) )

    # Generate the PDF
    pdf.output("Holdout_on_{}.pdf".format(modelName.replace(".pkl","")), 'F')
    
    # pprint(textHsh['min_auc'][-1])
    # Create a DataFrame from the presence dictionary
    performanceDF = pd.DataFrame({"Model":modelName, "Accuracy":textHsh['accuracy'], 
        "Max AUC": textHsh['max_auc'][-1], "Min AUC": textHsh['min_auc'][-1] }, index=[0])

    # Save the presence DataFrame to a CSV file
    performanceDF.to_csv("holdout_{}.csv".format(modelName.replace(".pkl","")), index=False)
    
    
    


