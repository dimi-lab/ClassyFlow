#!/usr/bin/env python3

import os, sys, re, random, math, time, glob
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

import fpdf
from fpdf import FPDF
import dataframe_image as dfi

## Static Variables: File Formatting
classColumn = 'Classification' #"${params.classifed_column_name}" # 'Classification'

cpu_jobs=32


############################ PDF REPORTING ############################
def create_letterhead(pdf, WIDTH):
     #pdf.image("${projectDir}/images/ClassyFlow_Letterhead.PNG", 0, 0, WIDTH)
     pdf.image("/research/bsi/projects/staff_analysis/m088378/SupervisedClassifierFlow/images/ClassyFlow_Letterhead.PNG", 0, 0, WIDTH)

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





def make_a_new_model(toTrainDF):
	allPDFText = {}
	X = toTrainDF[list(toTrainDF.select_dtypes(include=[np.number]).columns.values)]

	le = preprocessing.LabelEncoder()
	y_Encode = le.fit_transform(toTrainDF[classColumn])
	(unique, counts) = np.unique(y_Encode, return_counts=True)
	plt.barh(unique, counts)
	plt.savefig("label_bars.png", dpi=300, bbox_inches='tight')


	num_round = 200
	X_train, X_test, y_train, y_test = train_test_split(X, y_Encode, test_size=0.33, stratify=y_Encode)
	## check for NA values?
	# X_train[X_train.isna().any(axis=1)]
	dtrain = xgb.DMatrix(X_train, label=y_train)
	dtest = xgb.DMatrix(X_test, label=y_test)
	print("Training: {} Cells.   Test {} Cells.  Total Features: {}".format(X_train.shape[0],X_test.shape[0], X_train.shape[1]))

	# specify parameters via map
	depthFeild = [3,15,20]
	learnRates = [0.1,0.7]
	metricModel = []

	if os.path.exists('parameters_found.csv'):
		xgboostParams = pd.read_csv('parameters_found.csv')
	else:
		for d in depthFeild:
			for l in learnRates:
				start = time.time()
				param = {'max_depth':d, 'eta': l, 'objective':'multi:softmax', 'n_jobs': cpu_jobs,
				'num_class': len(unique), 'eval_metric': 'mlogloss' }
				bst = xgb.train(param, dtrain, num_round)

				predTrain = bst.predict(dtrain) ## Exports lables of type Float
				GBCmpredTrain = le.inverse_transform(np.array(predTrain, dtype=np.int32))
				yLabelTrain = le.inverse_transform(np.array(y_train, dtype=np.int32)) 
				accuracyTrain = accuracy_score(yLabelTrain, GBCmpredTrain)

				preds = bst.predict(dtest) ## Exports lables of type Float
				GBCmpred = le.inverse_transform(np.array(preds, dtype=np.int32))
				yLabelTest = le.inverse_transform(np.array(y_test, dtype=np.int32)) 
				accuracy = accuracy_score(yLabelTest, GBCmpred)
				metricModel.append({'max_depth':d, 'eta':l, 'Training': "%.2f%%" % (accuracyTrain * 100.0), 
				'Test': "%.2f%%" % (accuracy * 100.0), 'testf':accuracy })
				end = time.time()
				print( "XGB CPU Time %.2f" % (end - start))

		xgboostParams = pd.DataFrame(metricModel)
		xgboostParams.to_csv('parameters_found.csv', index=False)
		
	print(xgboostParams)
	styled_df = xgboostParams.style.format({'Max Tree Depth': "{}",
                      'ETA': "{:,}",
                      'Training Acc.': "{}",
                      'Test Acc.': "{}"}).hide()
	dfi.export(styled_df, 'parameter_search_results.png', table_conversion='matplotlib')
	
	
	
	mx = np.max(xgboostParams['testf'])
	rr = xgboostParams.loc[xgboostParams['testf'] == mx,]
	print("Max Test Accuracy: %.2f%%" % (mx * 100.0) )
	rr

	param = {'max_depth':rr['max_depth'].values[0], 'eta': rr['eta'].values[0], 'objective':'multi:softmax', 'n_jobs': cpu_jobs,
		'num_class': len(unique), 'eval_metric': 'mlogloss' }
	bst = xgb.train(param, dtrain, num_round)

	# make prediction
	predTrain = bst.predict(dtrain) ## Exports lables of type Float
	GBCmpredTrain = le.inverse_transform(np.array(predTrain, dtype=np.int))
	yLabelTrain = le.inverse_transform(np.array(y_train, dtype=np.int)) 
	# evaluate predictions
	accuracyTrain = accuracy_score(yLabelTrain, GBCmpredTrain)
	print("Training Accuracy: %.2f%%" % (accuracyTrain * 100.0))
	#print(pd.crosstab(GBCmpredTrain,yLabelTrain))
	print("\n")
	# make prediction
	preds = bst.predict(dtest) ## Exports lables of type Float
	GBCmpred = le.inverse_transform(np.array(preds, dtype=np.int))
	yLabelTest = le.inverse_transform(np.array(y_test, dtype=np.int)) 
	# evaluate predictions
	accuracy = accuracy_score(yLabelTest, GBCmpred)
	print("Test Accuracy: %.2f%%" % (accuracy * 100.0))
	
	# save model to file
	pickle.dump(bst, open("XGBoost_Model.pickle.dat", "wb"))


if __name__ == "__main__":
	myData = pd.read_pickle("training_dataframe.pkl")    #pd.read_pickle("${trainingDataframe}")
	with open("selected_features.csv", 'r') as file:     #"${select_features_csv}"
		next(file) # Skip header
		featureList = file.readlines()
	featureList = list(set([line.strip() for line in featureList]))
	featureList.remove('level_0')
	featureList.append(classColumn)
	focusData = myData[featureList]

	make_a_new_model(focusData)	
	
	WIDTH = 215.9
	HEIGHT = 279.4
	pdf = FPDF() # A4 (210 by 297 mm)
	pdf.add_page()
	# Add lettterhead and title
	create_letterhead(pdf, WIDTH)
	create_title("Model Training: XGBoost", pdf)
	# Add some words to PDF
	write_to_pdf(pdf, "Selected Features: {}".format(', '.join(featureList)))	
	pdf.ln(5)
	write_to_pdf(pdf, "Training Data {} cells by {} features".format(focusData.shape[0], focusData.shape[1]) )	
	pdf.ln(5)
	pdf.image('parameter_search_results.png', w= (WIDTH*0.4) )
	
	# Generate the PDF
	pdf.output("Model_Development_Xgboost.pdf", 'F')


