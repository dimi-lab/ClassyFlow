#!/usr/bin/env python3

import os, sys, csv, time, warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from matplotlib import pyplot
from numpy import mean
from numpy import std

import concurrent.futures
from functools import partial
from sklearn.exceptions import ConvergenceWarning

from pprint import pprint

### Following SciKit Learn recommended path
## https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html
# sphx-glr-auto-examples-linear-model-plot-lasso-model-selection-py
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoLarsIC
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import VarianceThreshold

import fpdf
from fpdf import FPDF
import dataframe_image as dfi

classColumn = 'Classification'
batchColumn = 'Batch'
varThreshold = 0.01
n_features_to_RFE = 16
n_folds = 10
lasso_max_iteration = 2000
ifSubsetData=False
max_workers = 2  # Limit the number of parallel processes
mim_class_label_threshold = 6
n_alphas_to_search = 26

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
def error_to_pdf(pdf, words):
    # Set text colour, font size, and font type
    pdf.set_text_color(r=200,g=0,b=55)
    pdf.set_font('Helvetica', '', 16)
    pdf.write(5, words)
############################ PDF REPORTING ############################

# evaluate a give model using cross-validation
# sklearn.metrics.SCORERS.keys()
## Passing in models by parameters will not work with Concurrent Processing...needs to be internal, like this.
def evaluate_model(idx, x, y, a):
	print(f"Starting task {idx}")
	rfe = RFE(estimator=Lasso(), n_features_to_select=idx)
	model = Lasso(alpha=a, max_iter=lasso_max_iteration)
	pipeline = Pipeline(steps=[('s',rfe),('m',model)])
	cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_folds, random_state=1)

	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", category=ConvergenceWarning)
		scores = cross_val_score(pipeline, x, y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')
	return idx, scores
	
	
def summary_table(data):
	# Create lists to store the data
	element = []
	mean_values = []
	std_values = []

	# Iterate over the data to extract mean and standard deviation
	for key, (elem, values) in data.items():
		element.append(elem)
		mean_values.append(np.mean(values))
		std_values.append(np.std(values))

	# Create a DataFrame
	summary_df = pd.DataFrame({
		'Features': element,
		'Mean': mean_values,
		'StdDev': std_values
	})

	return summary_df


def get_lasso_classification_features(df, celltype):
	allPDFText = {}
	df["cnt"]=1
	df["Lasso_Binary"] = 0
	df.loc[df[classColumn] == celltype, 'Lasso_Binary'] = 1
	
	## Skip if too few labels exist
	totCls = df["Lasso_Binary"].sum()
	print("{} has {} lables".format(celltype, totCls))
	if totCls < mim_class_label_threshold:
		allPDFText['too_few'] = "{} '{}' is not enough class labels to model!".format(totCls, celltype)
		return allPDFText
	else:
		allPDFText['too_few'] = ""
	
	
	### Add optional parameter to speed up by reducing data amount. Half of target class size
	if ifSubsetData:
		totRow = df.shape[0]
		if totCls < 1000:
			df1 = df[df["Lasso_Binary"] == 1]
		else:
			df1 = df[df["Lasso_Binary"] == 1].sample( n=1000 )
			
		negN = totRow - totCls
		if negN < 1000:
			df2 = df[df["Lasso_Binary"] == 0]
		else:
			df2 = df[df["Lasso_Binary"] == 0].sample( n=1000 )
			
		df = pd.concat([df1,df2])
			
	
	## too big to plot
	print(df.groupby([batchColumn, 'Lasso_Binary']).size() )
	binaryCntTbl = df.groupby([batchColumn, 'Lasso_Binary']).size().reset_index()
	styled_df = binaryCntTbl.style.format({'Batches': "{}",
                      'Binary': "{:,}",
                      'Frequency': "{:,}"}).hide()
	dfi.export(styled_df, 'binary_count_table.png', table_conversion='matplotlib')
	
	XAll = df[list(df.select_dtypes(include=[np.number]).columns.values)]
	XAll = XAll[XAll.columns.drop(list(XAll.filter(regex='(Centroid|Binary|cnt|Name)')))].fillna(0)
	yAll = df['Lasso_Binary']

	# using sklearn variancethreshold to find constant features
	sel = VarianceThreshold(threshold=varThreshold)
	sel.fit(XAll)  # fit finds the features with zero variance
	# get_support is a boolean vector that indicates which features are retained
	# if we sum over get_support, we get the number of features that are not constant
	sum(sel.get_support())

	# print the constant features
	nonVarFeatures = [x for x in XAll.columns if x not in XAll.columns[sel.get_support()]]
	print("NonVariant Features: "+', '.join(nonVarFeatures))
	allPDFText['nonVarFeatures'] = nonVarFeatures

	
	X_train, X_test, y_train, y_test = train_test_split(XAll, yAll, test_size=0.33, random_state=101, stratify=yAll)
	#alphas = np.arange(0.0002,0.004,0.0003)
	### Add Templating right here.
	alphas = np.logspace(-5.1,-0.0004, n_alphas_to_search)
	pipeline = Pipeline([
		('scaler',StandardScaler(with_mean=False)),
		('model',Lasso())
	])
	search = GridSearchCV(pipeline,
		{'model__alpha': alphas},
		cv = n_folds, 
		scoring="neg_mean_squared_error",
		verbose=3
	)
	search.fit(X_train,y_train)
	allPDFText['best_alpha'] = search.best_params_['model__alpha']
	#allPDFText['best_alpha'] = 0.002792543841237339
	print( "Best Alpha: {}".format( allPDFText['best_alpha'] ) )
		
	scores = search.cv_results_["mean_test_score"]
	scores_std = search.cv_results_["std_test_score"]
	plt.figure().set_size_inches(9, 6)
	plt.semilogx(alphas, scores)

	std_error = scores_std / np.sqrt(n_folds)
	plt.semilogx(alphas, scores + std_error, "b--")
	plt.semilogx(alphas, scores - std_error, "b--")

	# alpha=0.2 controls the translucency of the fill color
	plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)
	plt.axvline(allPDFText['best_alpha'], linestyle="--", color="green", label="alpha: Best Fit")
	plt.ylabel("CV score +/- std error")
	plt.xlabel("alpha")
	plt.axhline(np.max(scores), linestyle="--", color=".5")
	plt.xlim([alphas[0], alphas[-1]])
	# Save the plot as a PNG
	plt.savefig("best_alpha_plot.png", dpi=300, bbox_inches='tight')

	features = XAll.columns.values.tolist()
	coefficients = search.best_estimator_.named_steps['model'].coef_
	importance = np.abs(coefficients)
	featureRankDF = pd.DataFrame(data=importance, index=features, columns=["score"])
	frPlot = featureRankDF.nlargest(35, columns="score").sort_values(by = "score", ascending=True).plot(kind='barh', figsize = (8,12)) 
	fig = frPlot.get_figure()
	fig.savefig("feature_ranking_plot.png")

	
	########## Start output rank file
	dfF = pd.DataFrame( list(zip(features, importance)), columns=['Name', 'Feature_Importance'])
	dfF = dfF.sort_values(by=['Feature_Importance'], ascending=False)


	# evaluate the models and store results
	results = {}

	with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
		evalModelPart = partial(evaluate_model, x=XAll, y=yAll, a=allPDFText['best_alpha'])
		future_to_task = {executor.submit(evalModelPart, idx): idx for idx in list(range(2, n_features_to_RFE)) }
		for future in concurrent.futures.as_completed(future_to_task):
			name = future_to_task[future]
			try:
				result = future.result()
				results[name] = result
			except Exception as exc:
				print(f'{name} generated an exception: {exc}')

	pprint(results)
	summary_df = summary_table(results)
	styled_df = summary_df.style.format({'Number of Features': "{}",
                      'Mean (-RMSE)': "{:,}",
                      'Std.Dev. (-RMSE)': "{:,}"}).hide()
	dfi.export(styled_df, 'ref_summary_table.png', table_conversion='matplotlib')
		
	idxnames = [str(key) for key in results.keys()]
	rfevalues = [val[1] for val in results.values()]
	
	# plot model performance for comparison
	pyplot.cla()  
	pyplot.boxplot(rfevalues, labels=idxnames, showmeans=True)  ## add axis labels
	pyplot.savefig("recursive_elimination_plot.png")
	
	##### NEED TO ADD SMARTER CUTOFF = "One Standard Error Rule"
	featureCutoff = int(n_features_to_RFE/3)
	ctl = dfF['Name'].tolist()[:featureCutoff]	
	with open("top_rank_features_{}.csv".format(celltype.replace(' ','_').replace('|','_')), 'w', newline='') as csvfile:
		f_writer = csv.writer(csvfile)
		f_writer.writerow(["Features"])
		for ln in ctl:
			f_writer.writerow([ln])
	
	return allPDFText

if __name__ == "__main__":
	myData = pd.read_pickle("${trainingDataframe}")
	myLabel = "${celltype}".replace('[', '').replace(']', '')  ### figure out why this passes an array from nextflow...??
	#myLabel = myLabel = "[B cell|T reg]".replace('[', '').replace(']', '')	

	hshResults = get_lasso_classification_features(myData, myLabel)

	WIDTH = 215.9
	HEIGHT = 279.4
	pdf = FPDF() # A4 (210 by 297 mm)
	pdf.add_page()
	# Add lettterhead and title
	create_letterhead(pdf, WIDTH)
	create_title("Feature Evaluation: {}".format(myLabel), pdf)
	if hshResults['too_few'] == "":
		# Add some words to PDF
		write_to_pdf(pdf, "In-Variant Feature Threshold: {}".format(varThreshold))	
		pdf.ln(5)
		# Add table
		pdf.image('binary_count_table.png', w= (WIDTH*0.3) )
		pdf.ln(10)
		write_to_pdf(pdf, "In-Variant Features: {}".format(', '.join(hshResults['nonVarFeatures'])))
		pdf.ln(10)
		write_to_pdf(pdf, "Best Alpha: {}".format( hshResults['best_alpha'] ))
		pdf.ln(5)
		pdf.image('best_alpha_plot.png', w= (WIDTH*0.8) )
		pdf.image('feature_ranking_plot.png', w= (WIDTH*0.8) )
		pdf.image('ref_summary_table.png', w= (WIDTH*0.4) )
		pdf.image('recursive_elimination_plot.png', w= (WIDTH*0.8) )
	else:
		error_to_pdf(pdf,hshResults['too_few'])
		### Supply blank file
		with open("top_rank_features_{}.csv".format(myLabel.replace(' ','_').replace('|','_')), 'w', newline='') as csvfile:
			f_writer = csv.writer(csvfile)
			f_writer.writerow(["Features"])
		
		
		
	# Generate the PDF
	pdf.output("{}_Features.pdf".format(myLabel.replace(' ','_').replace('|','_')), 'F')


