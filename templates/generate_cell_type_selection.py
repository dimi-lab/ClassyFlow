#!/usr/bin/env python3

import os, sys, csv, time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from matplotlib import pyplot


from numpy import mean
from numpy import std

### Following SciKit Learn recommended path
## https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html
# sphx-glr-auto-examples-linear-model-plot-lasso-model-selection-py
import time
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

import fpdf
from fpdf import FPDF
import dataframe_image as dfi

classColumn = 'Classification'
batchColumn = 'Batch'
varThreshold = 0.01
WIDTH = 215.9
HEIGHT = 279.4

############################ PDF REPORTING ############################
def create_letterhead(pdf, WIDTH):
    # pdf.image("${projectDir}/images/ClassyFlow_Letterhead.PNG", 0, 0, WIDTH)
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


def get_lasso_classification_features(df, celltype):
	pdf = FPDF() # A4 (210 by 297 mm)
	pdf.add_page()
	# Add lettterhead and title
	create_letterhead(pdf, WIDTH)
	create_title("Feature Evaluation: {}".format(celltype), pdf)
	# Add some words to PDF
	write_to_pdf(pdf, "In-Variant Feature Threshold: {}".format(varThreshold))	
	pdf.ln(5)



	df["cnt"]=1
	df["Lasso_Binary"] = 0
	df.loc[df[classColumn] == celltype, 'Lasso_Binary'] = 1
	## too big to plot
	# print(df.groupby([batchColumn, 'Lasso_Binary']).size() )
	binaryCntTbl = df.groupby([batchColumn, 'Lasso_Binary']).size().reset_index()
	styled_df = binaryCntTbl.style.format({'Batches': "{}",
                      'Binary': "{:,}",
                      'Frequency': "{:,}"}).hide()
	dfi.export(styled_df, 'binary_count_table.png')
	# Add table
	pdf.image('binary_count_table.png', w= (WIDTH*0.3) )


	XAll = df[list(df.select_dtypes(include=[np.number]).columns.values)]
	XAll = XAll[XAll.columns.drop(list(XAll.filter(regex='(Centroid|Binary|cnt|Name)')))].fillna(0)
	yAll = df['Lasso_Binary']

	# using sklearn variancethreshold to find constant features
	from sklearn.feature_selection import VarianceThreshold
	sel = VarianceThreshold(threshold=varThreshold)
	sel.fit(XAll)  # fit finds the features with zero variance

	# get_support is a boolean vector that indicates which features are retained
	# if we sum over get_support, we get the number of features that are not constant
	sum(sel.get_support())

	# print the constant features
	nonVarFeatures = [x for x in XAll.columns if x not in XAll.columns[sel.get_support()]]
	pdf.ln(10)
	write_to_pdf(pdf, "In-Variant Features: {}".format(', '.join(nonVarFeatures)))
	pdf.ln(10)
	
	
	X_train, X_test, y_train, y_test = train_test_split(XAll, yAll, test_size=0.33, random_state=101, stratify=yAll)
	#alphas = np.arange(0.0002,0.004,0.0003)
	alphas = np.logspace(-5.1,-0.008, 3)
	n_folds = 10

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

	write_to_pdf(pdf, "Best Alpha: {}".format(search.best_params_['model__alpha']))
	pdf.ln(5)

	scores = search.cv_results_["mean_test_score"]
	scores_std = search.cv_results_["std_test_score"]
	plt.figure().set_size_inches(9, 6)
	plt.semilogx(alphas, scores)

	std_error = scores_std / np.sqrt(n_folds)
	plt.semilogx(alphas, scores + std_error, "b--")
	plt.semilogx(alphas, scores - std_error, "b--")

	# alpha=0.2 controls the translucency of the fill color
	plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)
	bestAlpha = search.best_params_['model__alpha']
	plt.axvline(bestAlpha, linestyle="--", color="green", label="alpha: Best Fit")
	plt.ylabel("CV score +/- std error")
	plt.xlabel("alpha")
	plt.axhline(np.max(scores), linestyle="--", color=".5")
	plt.xlim([alphas[0], alphas[-1]])
	# Save the plot as a PNG
	plt.savefig("best_alpha_plot.png", dpi=300, bbox_inches='tight')
	pdf.image('best_alpha_plot.png', w= (WIDTH*0.8) )




	features = XAll.columns.values.tolist()
	coefficients = search.best_estimator_.named_steps['model'].coef_
	importance = np.abs(coefficients)
	featureRankDF = pd.DataFrame(data=importance, index=features, columns=["score"])
	frPlot = featureRankDF.nlargest(35, columns="score").sort_values(by = "score", ascending=True).plot(kind='barh', figsize = (8,12)) 
	fig = frPlot.get_figure()
	fig.savefig("feature_ranking_plot.png")
	pdf.image('feature_ranking_plot.png', w= (WIDTH*0.8) )
	
	########## Start output rank file
	dfF = pd.DataFrame( list(zip(features, importance)), columns=['Name', 'Feature_Importance'])
	dfF.sort_values(by=['Feature_Importance'], ascending=False)
	



	# get a list of models to evaluate
	def get_models():
		models = dict()
		for i in range(2, 25):
			rfe = RFE(estimator=Lasso(), n_features_to_select=i)
			model = Lasso(alpha=bestAlpha)
			models[str(i)] = Pipeline(steps=[('s',rfe),('m',model)])
		return models

	# evaluate a give model using cross-validation
	# import sklearn
	# sklearn.metrics.SCORERS.keys()
	def evaluate_model(model, X, y):
		cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
		scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')
		return scores

	# get the models to evaluate
	models = get_models()
	# evaluate the models and store results
	results, names = list(), list()
	for name, model in models.items():
		scores = evaluate_model(model, XAll, yAll)
		results.append(scores)
		names.append(name)
		print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
	# plot model performance for comparison
	pyplot.boxplot(results, labels=names, showmeans=True)
	pyplot.savefig("recursive_elimination_plot.png")

	

	# Generate the PDF
	pdf.output("{}_Features.pdf".format(celltype.replace(' ','_')), 'F')


if __name__ == "__main__":
	#	myData = pd.read_pickle("${trainingDataframe}")
	myData = pd.read_pickle("training_dataframe.pkl")
	#myLabel = "${celltype}"
	myLabel = "Tumor Cell"	

	get_lasso_classification_features(myData, myLabel)


	

