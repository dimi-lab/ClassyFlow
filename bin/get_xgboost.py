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
from random import randint

## Static Variables: File Formatting
classColumn = "${params.classifed_column_name}" # 'Classification'
cpu_jobs=16
uTaskID="${task.index}"
mim_class_label_threshold = ${params.minimum_label_count}

def make_a_new_model(toTrainDF):
	allPDFText = {}
	class_counts = toTrainDF[classColumn].value_counts()
	# Identify classes with fewer than 2 instances
	classes_to_keep = class_counts[class_counts > mim_class_label_threshold].index
	# Filter the dataframe to remove these classes
	toTrainDF = toTrainDF[toTrainDF[classColumn].isin(classes_to_keep)]
	X = toTrainDF[list(toTrainDF.select_dtypes(include=[np.number]).columns.values)]

	le = preprocessing.LabelEncoder()
	y_Encode = le.fit_transform(toTrainDF[classColumn])
	(unique, counts) = np.unique(y_Encode, return_counts=True)
	#plt.barh(unique, counts)
	#plt.savefig("label_bars.png", dpi=300, bbox_inches='tight')


	num_round = 200
	# specify parameters via map
	metricModel = []
	c = int("${cv_c}")
	X_train, X_test, y_train, y_test = train_test_split(X, y_Encode, test_size=0.33, stratify=y_Encode)
	## check for NA values?
	dtrain = xgb.DMatrix(X_train, label=y_train)
	dtest = xgb.DMatrix(X_test, label=y_test)
	print("Training: {} Cells.   Test {} Cells.  Total Features: {}".format(X_train.shape[0],X_test.shape[0], X_train.shape[1]))
	
	d = int("${depth_d}")
	l = float("${eta_l}")
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
	metricModel.append({'cv':c,'max_depth':d, 'eta':l, 'Training': "%.2f%%" % (accuracyTrain * 100.0), 
	'Test': "%.2f%%" % (accuracy * 100.0), 'testf':accuracy })
	end = time.time()
	print( "XGB CPU Time %.2f" % (end - start))

	xgboostParams = pd.DataFrame(metricModel)
	rnd = randint(1000, 9999) 
	xgboostParams.to_csv("parameters_found_{}_{}.csv".format(rnd,uTaskID), index=False)

if __name__ == "__main__":
	myData = pd.read_pickle("${trainingDataframe}")
	with open("${select_features_csv}", 'r') as file:
		next(file) # Skip header
		featureList = file.readlines()
	featureList = list(set([line.strip() for line in featureList]))
	if 'level_0' in featureList:
		featureList.remove('level_0')
	featureList.append(classColumn)
	focusData = myData[featureList]

	make_a_new_model(focusData)


