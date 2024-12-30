#!/usr/bin/env python3

import os, sys, re, random, math, time, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize

import pickle

import xgboost as xgb

import fpdf
from fpdf import FPDF
import dataframe_image as dfi

## Static Variables: File Formatting
classColumn = 'Classification' #"${params.classifed_column_name}" # 'Classification'

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
    pdf.set_text_color(r=72,g=72,b=72)
    pdf.set_font('Helvetica', '', 12)
    pdf.write(5, words)

def write_header(pdf, words):
    pdf.set_text_color(r=0,g=0,b=0)
    pdf.set_font('Helvetica', 'b', 14)
    pdf.write(5,words)

############################ PDF REPORTING ############################



def check_holdout(toCheckDF, xgbM):
	allPDFText = {}
	X_holdout = toCheckDF[list(toCheckDF.select_dtypes(include=[np.number]).columns.values)]
	X_holdout = X_holdout[xgbM.feature_names]

	le = preprocessing.LabelEncoder()
	le.classes_ = np.load(leEncoderFile, allow_pickle=True)
	y_holdout = le.transform(toCheckDF[classColumn])


	# Make predictions
	dmatrix = xgb.DMatrix(X_holdout)
	y_pred_proba = xgbM.predict(dmatrix)

	allPDFText['accuracy'] = accuracy_score(y_holdout, y_pred_proba)
	allPDFText['balanced_accuracy'] = balanced_accuracy_score(y_holdout, y_pred_proba)
	allPDFText['f1_score'] = f1_score(y_holdout, y_pred_proba, average="weighted")
	allPDFText['matthews_corrcoef'] = matthews_corrcoef(y_holdout, y_pred_proba)


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
	plt.figure(figsize=(10,7))
	plt.title("Confusion Matrix")
	ax = sns.heatmap(metrics_df, annot=True, fmt='.10g', cmap="YlGnBu")
	ax.set_xticklabels(metrics_df.columns)
	ax.set_yticklabels(metrics_df.columns, rotation=0)
	ax.set(ylabel="True Label", xlabel="Predicted Label")
	plt.savefig("confusion_heatmap.png",  bbox_inches="tight")


	# Multiclass case
	y_true_binarized = label_binarize(y_holdout, classes=np.arange(n_classes))
	y_pred_binarized = label_binarize(y_pred_proba, classes=np.arange(n_classes))
	#fpr = dict()
	#tpr = dict()
	#roc_auc = dict()

	#for i in range(n_classes):
	#	fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_binarized[:, i])
	#	roc_auc[i] = auc(fpr[i], tpr[i])

	#plt.figure()
	#colors = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'yellow']
	#for i in range(n_classes):
	#	plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
	#	label=f'{lableHash[i]} (a={roc_auc[i]:.2f})')

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
	#xgbMdl = pickle.load("${model_pickle}")

	textHsh = check_holdout(focusData, xgbMdl)
	
	WIDTH = 215.9
	HEIGHT = 279.4
	pdf = FPDF() # A4 (210 by 297 mm)
	pdf.add_page()
	# Add lettterhead and title
	create_letterhead(pdf, WIDTH)
	create_title("Holdout Evaluation: XGBoost", pdf)
	# Add some words to PDF
	# Brief description of this report
	write_to_pdf(pdf, "A hold-out set is a portion of your data that you set aside and do not use during the training of your machine learning model. Think of it as a test group that helps you evaluate how well your model performs on new, unseen data")
	pdf.ln(10)

	write_header(pdf, "Evaluation Metrics:")
	pdf.ln(10)
	write_to_pdf(pdf, "The table below summarizes some commonly used metrics to help us evaluate the model")
	pdf.ln(10)
	# Add some words to PDF
	eval_metrics = (
	("Metric", "Value"),
	("Accuracy", round(textHsh['accuracy']*100,2)),
	("Balanced Accuracy", round(textHsh['balanced_accuracy']*100,2)),
	("F1 Score", round(textHsh['f1_score']*100,2)),
	("Matthews Corr Coef", round(textHsh['matthews_corrcoef']*100,2)))
	pdf.set_font('Helvetica', '', 12)
	pdf.set_text_color(r=0,g=0,b=0)

	for data_row in eval_metrics:
		for data in data_row:
			pdf.cell(w=60, h=10, txt=str(data), border=1)
		pdf.ln()

	pdf.add_page()
	write_header(pdf, "Hold-out Data:")
	pdf.ln(10)
	write_to_pdf(pdf, 'The following bar plot summarizes the number of cells we held-out for each label. For this project, we used a hold out fraction of ${params.holdout_fraction}')
	pdf.ln(10)
	pdf.image('label_bars.png', w= (WIDTH*0.85) )

	pdf.add_page()
	write_header(pdf, "Confusion Matrix:")
	pdf.ln(10)
	write_to_pdf(pdf, "This is a heatmap that shows the number of true positive, true negative, false positive, and false negative predictions. It provides a detailed breakdown of the model's performance, helping to understand where the model is making errors.")
	pdf.ln(10)
	pdf.image('confusion_heatmap.png', w= (WIDTH*0.8) )

	pdf.add_page()
	write_header(pdf, "ROC/AUC:")
	pdf.ln(10)
	write_to_pdf(pdf, "ROC/AUC (Receiver Operating Characteristic/Area Under the Curve): The ROC curve is a graphical representation of the model's ability to distinguish between classes. The AUC is the area under this curve, providing a single value that summarizes the model's performance. A higher AUC indicates better performance.")
	pdf.ln(10)
	pdf.image('auc_curve_stacked.png', w= (WIDTH*0.85) )

	# Generate the PDF
	pdf.output("Holdout_on_{}.pdf".format(modelName.replace(".pkl","")), 'F')
	
	# pprint(textHsh['min_auc'][-1])
	# Create a DataFrame from the presence dictionary
	performanceDF = pd.DataFrame({"Model":modelName, "Accuracy":textHsh['accuracy'], 
		"Max AUC": textHsh['max_auc'][-1], "Min AUC": textHsh['min_auc'][-1] }, index=[0])

	# Save the presence DataFrame to a CSV file
	performanceDF.to_csv("holdout_{}.csv".format(modelName.replace(".pkl","")), index=False)
	
	
	


