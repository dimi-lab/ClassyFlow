#!/usr/bin/env python3

import os, sys
import pickle

import xgboost as xgb
from sklearn import preprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint


import fpdf
from fpdf import FPDF
import dataframe_image as dfi
from random import randint

## Static Variables: File Formatting
classColumn = 'Classification' #"${params.classifed_column_name}" # 'Classification'

leEncoderFile = "${params.output_dir}/models/classes.npy"
#leEncoderFile = "../../../output/models/classes.npy"
cpu_jobs=16

columnsToExport = ['Centroid X µm', 'Centroid Y µm', 'Image','CellTypePrediction']



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



def predict_on_xgb_best_model(toCheckDF, xgbM, bID):
	allPDFText = {}
	
	le = preprocessing.LabelEncoder()
	le.classes_ = np.load(leEncoderFile, allow_pickle=True)
	
	toGetDataFrame = toCheckDF[xgbM.feature_names]
	# Make predictions
	dmatrix = xgb.DMatrix(toGetDataFrame)
	y_pred_all = xgbM.predict(dmatrix)
	y_pred_all_int = np.array(y_pred_all, dtype=int)

	pprint(y_pred_all)
	classCellNames = le.inverse_transform(y_pred_all_int)
	pprint(classCellNames)

	toCheckDF['CellTypePrediction'] = classCellNames
	
	toExport = toCheckDF[columnsToExport]
	print(toExport.head())
	
	for img in toExport['Image'].unique():
		roiTbl = toExport[toExport['Image'] == img]
		outFh = os.path.join(img+"_PRED.tsv")
		roiTbl.to_csv(outFh, sep="\t", index=False)


if __name__ == "__main__":
	batchID = "${batchID}"
	infile = "${pickleTable}"
	if infile.endswith('.pkl'):
	    myData = pd.read_pickle(infile)
	else:
	    myData = pd.read_csv(infile, sep='\t', low_memory=False)
		
	modelfile = "${model_path}"
	with open(modelfile, 'rb') as file:
		xgbMdl = pickle.load(file)

	predict_on_xgb_best_model(myData, xgbMdl, batchID)	






