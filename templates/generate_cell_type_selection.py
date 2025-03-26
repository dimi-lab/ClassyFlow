#!/usr/bin/env python3

import os, sys, re, csv, time, warnings
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
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import VarianceThreshold

import fpdf
from fpdf import FPDF
import dataframe_image as dfi

classColumn = 'Classification'
batchColumn = 'Batch'
varThreshold = 0.01
n_features_to_RFE = 20
n_folds = 12

ifSubsetData=True
max_workers = 8  # Limit the number of parallel processes
#mim_class_label_threshold = ${params.minimum_label_count}
mim_class_label_threshold = 20
n_alphas_to_search = 8
varThreshold = 0.01

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


def get_lasso_classification_features(df, celltype, a, aTbl, rfeTbl):
	allPDFText = {}
	allPDFText['best_alpha'] = a
	## too big to plot
	print(df.groupby([batchColumn, 'Lasso_Binary']).size() )
	binaryCntTbl = df.groupby([batchColumn, 'Lasso_Binary']).size().reset_index()
	styled_df = binaryCntTbl.style.format({'Batches': "{}",
                      'Binary': "{:,}",
                      'Frequency': "{:,}"}).hide()
	dfi.export(styled_df, 'binary_count_table.png'.format(celltype), table_conversion='matplotlib')


	print( "Best Alpha: {}".format( allPDFText['best_alpha'] ) )
	scores = aTbl["mean_test_score"]
	scores_std = aTbl["std_test_score"]
	#	alphas = aTbl["logspace"]
	alphas = list(aTbl["input_a"].str.split('-').str[0].astype(float))
	#print(alphas[0], alphas[-1])
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
	allPDFText['nonVarFeatures'] = ', '.join(nonVarFeatures)



	clf = Lasso(alpha=a)
	clf.fit(XAll, yAll)
	features = XAll.columns.values.tolist()
	coefficients = clf.coef_
	importance = np.abs(coefficients)
	featureRankDF = pd.DataFrame(data=importance, index=features, columns=["score"])
	frPlot = featureRankDF.nlargest(35, columns="score").sort_values(by = "score", ascending=True).plot(kind='barh', figsize = (8,12)) 
	fig = frPlot.get_figure()
	fig.savefig("feature_ranking_plot.png")

	########## Start output rank file
	dfF = pd.DataFrame( list(zip(features, importance)), columns=['Name', 'Feature_Importance'])
	dfF = dfF.sort_values(by=['Feature_Importance'], ascending=False)

	summary_df = rfeTbl.groupby('n_features')['rfe_score'].agg(['mean', 'std', 'median'])
	styled_df = summary_df.style.format({'Number of Features': "{}",
                      'Mean (-RMSE)': "{:,}",
                      'Std.Dev. (-RMSE)': "{:,}"}).hide()
	dfi.export(styled_df, 'ref_summary_table.png', table_conversion='matplotlib')
	
	#print(rfeTbl.head())
	# Group the data by the 'Category' column
	categories = sorted(rfeTbl['n_features'].unique())
	grouped_data = [rfeTbl[rfeTbl['n_features'] == cat]['rfe_score'] for cat in categories]
	

	# Plot model performance for comparison
	pyplot.cla()  
	box = pyplot.boxplot(grouped_data, labels=categories, patch_artist=True, showmeans=True)
	
	
	##### ADD SMARTER CUTOFF = "One Standard Error Rule"
	# Identify the smallest n_features within 1 standard deviation of the median
	# Find the global median across all n_features
	global_median = rfeTbl['rfe_score'].median()
	global_sd = ( rfeTbl['rfe_score'].std() / 8 ) ## modify to increase asymtotic behaviour
	print(global_median, global_sd)
	print(summary_df)
	# Find the smallest n_features within 1 standard deviation of the global median
	filtered = summary_df[summary_df['median'] >= (global_median-global_sd)]
	print(filtered)
	featureCutoff = filtered.index.min()	
	allPDFText['Optimal_N_Features'] = featureCutoff

	# Set colors for each box
	for patch, category in zip(box['boxes'], categories):
		if category == featureCutoff:
			patch.set_facecolor('lightgreen')
		else:
			patch.set_facecolor('white')

	# Set colors for other elements of the boxplot (e.g., medians, means)
	for element in ['medians', 'means', 'whiskers', 'caps', 'fliers']:
		plt.setp(box[element], color='black')

	
	pyplot.xlabel('Number of Features')  
	pyplot.ylabel('RFE Score')  
	pyplot.title('Recursive Feature Elimination Plot')  
	pyplot.savefig("recursive_elimination_plot.png")
	
	
	
	ctl = dfF['Name'].tolist()[:featureCutoff]	
	with open("top_rank_features_{}.csv".format(celltype.replace(' ','_').replace('|','_').replace('/','')), 'w', newline='') as csvfile:
		f_writer = csv.writer(csvfile)
		f_writer.writerow(["Features"])
		for ln in ctl:
			f_writer.writerow([ln])
	
	allPDFText['too_few'] = ""
	return allPDFText

if __name__ == "__main__":
	myData = pd.read_pickle("${trainingDataframe}")
	myLabel = "${celltype}".replace('[', '').replace(']', '')  ### figure out why this passes an array from nextflow...??
	#myLabel = myLabel = "[B cell|T reg]".replace('[', '').replace(']', '')
	rfeScores = pd.read_csv("${rfe_scores}")
	best_alpha= ${best_alpha}
	alphaScores = pd.read_csv("${alpha_scores}")
	
	hshResults = get_lasso_classification_features(myData, myLabel, best_alpha, alphaScores, rfeScores)

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
		write_to_pdf(pdf, "In-Variant Features: {}".format(hshResults['nonVarFeatures']))
		pdf.ln(10)
		write_to_pdf(pdf, "Best Alpha: {}".format( hshResults['best_alpha'] ))
		pdf.ln(5)
		pdf.image('best_alpha_plot.png', w= (WIDTH*0.8) )
		pdf.image('feature_ranking_plot.png', w= (WIDTH*0.8) )
		pdf.image('ref_summary_table.png', w= (WIDTH*0.4) )
		write_to_pdf(pdf, "Optimal Number of Features: {}".format(hshResults['Optimal_N_Features']))
		pdf.ln(10)
		pdf.image('recursive_elimination_plot.png', w= (WIDTH*0.8), h=(HEIGHT*0.58) )
	else:
		error_to_pdf(pdf,hshResults['too_few'])
		### Supply blank file
		with open("top_rank_features_{}.csv".format(myLabel.replace(' ','_').replace('|','_').replace('/','')), 'w', newline='') as csvfile:
			f_writer = csv.writer(csvfile)
			f_writer.writerow(["Features"])
		
		
		
	# Generate the PDF
	pdf.output("{}_Features.pdf".format(myLabel.replace(' ','_').replace('|','_').replace('/','')), 'F')


