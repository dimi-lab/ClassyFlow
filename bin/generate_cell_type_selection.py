#!/usr/bin/env python3

import os, sys, re, csv, time, warnings
import argparse
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

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoLarsIC, Lasso
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.feature_selection import RFE, VarianceThreshold

import fpdf
from fpdf import FPDF
import dataframe_image as dfi

batchColumn = 'Batch'



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


def get_lasso_classification_features(
    df, celltype, a, aTbl, rfeTbl, varThreshold, n_folds, n_features_to_RFE, ifSubsetData, max_workers, mim_class_label_threshold, classColumn
):
    allPDFText = {}
    allPDFText['best_alpha'] = a
    print(df.groupby([batchColumn, 'Lasso_Binary']).size())
    binaryCntTbl = df.groupby([batchColumn, 'Lasso_Binary']).size().reset_index()
    styled_df = binaryCntTbl.style.format({'Batches': "{}",
                      'Binary': "{:,}",
                      'Frequency': "{:,}"}).hide()
    dfi.export(styled_df, 'binary_count_table.png'.format(celltype), table_conversion='matplotlib')

    print("Best Alpha: {}".format(allPDFText['best_alpha']))
    scores = aTbl["mean_test_score"]
    scores_std = aTbl["std_test_score"]
    alphas = list(aTbl["input_a"].str.split('-').str[0].astype(float))
    plt.figure().set_size_inches(9, 6)
    plt.semilogx(alphas, scores)
    std_error = scores_std / np.sqrt(n_folds)
    plt.semilogx(alphas, scores + std_error, "b--")
    plt.semilogx(alphas, scores - std_error, "b--")
    plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)
    plt.axvline(allPDFText['best_alpha'], linestyle="--", color="green", label="alpha: Best Fit")
    plt.ylabel("CV score +/- std error")
    plt.xlabel("alpha")
    plt.axhline(np.max(scores), linestyle="--", color=".5")
    plt.xlim([alphas[0], alphas[-1]])
    plt.savefig("best_alpha_plot.png", dpi=300, bbox_inches='tight')

    XAll = df[list(df.select_dtypes(include=[np.number]).columns.values)]
    XAll = XAll[XAll.columns.drop(list(XAll.filter(regex='(Centroid|Binary|cnt|Name)')))].fillna(0)
    yAll = df['Lasso_Binary']

    sel = VarianceThreshold(threshold=varThreshold)
    sel.fit(XAll)
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

    dfF = pd.DataFrame(list(zip(features, importance)), columns=['Name', 'Feature_Importance'])
    dfF = dfF.sort_values(by=['Feature_Importance'], ascending=False)

    summary_df = rfeTbl.groupby('n_features')['rfe_score'].agg(['mean', 'std', 'median'])
    styled_df = summary_df.style.format({'Number of Features': "{}",
                      'Mean (-RMSE)': "{:,}",
                      'Std.Dev. (-RMSE)': "{:,}"}).hide()
    dfi.export(styled_df, 'ref_summary_table.png', table_conversion='matplotlib')

    categories = sorted(rfeTbl['n_features'].unique())
    grouped_data = [rfeTbl[rfeTbl['n_features'] == cat]['rfe_score'] for cat in categories]

    pyplot.cla()  
    box = pyplot.boxplot(grouped_data, labels=categories, patch_artist=True, showmeans=True)

    global_median = rfeTbl['rfe_score'].median()
    global_sd = ( rfeTbl['rfe_score'].std() / 8 )
    filtered = summary_df[summary_df['median'] >= (global_median-global_sd)]
    featureCutoff = filtered.index.min()    
    allPDFText['Optimal_N_Features'] = featureCutoff

    for patch, category in zip(box['boxes'], categories):
        if category == featureCutoff:
            patch.set_facecolor('lightgreen')
        else:
            patch.set_facecolor('white')
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
    parser = argparse.ArgumentParser(description="Feature selection and reporting for cell type classification.")
    parser.add_argument('--trainingDataframe', required=True, help='Path to the training dataframe pickle file')
    parser.add_argument('--celltype', required=True, help='Cell type label')
    parser.add_argument('--rfe_scores', required=True, help='Path to RFE scores CSV')
    parser.add_argument('--best_alpha', type=float, required=True, help='Best alpha value')
    parser.add_argument('--alpha_scores', required=True, help='Path to alpha scores CSV')
    parser.add_argument('--classColumn', default='Classification', help='Column name for cell type classification')
    parser.add_argument('--varThreshold', type=float, default=0.01, help='Variance threshold')
    parser.add_argument('--n_features_to_RFE', type=int, default=20, help='Number of features to RFE')
    parser.add_argument('--n_folds', type=int, default=12, help='Number of folds for cross-validation')
    parser.add_argument('--ifSubsetData', type=lambda x: (str(x).lower() == 'true'), default=True, help='Whether to subset data')
    parser.add_argument('--max_workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--mim_class_label_threshold', type=int, default=20, help='Minimum class label threshold')
    parser.add_argument('--n_alphas_to_search', type=int, default=8, help='Number of alphas to search')
    parser.add_argument('--letterhead', required=True, help='Path to letterhead image for report')
    args = parser.parse_args()

    myData = pd.read_pickle(args.trainingDataframe)
    myLabel = args.celltype.replace('[', '').replace(']', '')
    rfeScores = pd.read_csv(args.rfe_scores)
    best_alpha = args.best_alpha
    alphaScores = pd.read_csv(args.alpha_scores)

    hshResults = get_lasso_classification_features(
        myData,
        myLabel,
        best_alpha,
        alphaScores,
        rfeScores,
        args.varThreshold,
        args.n_folds,
        args.n_features_to_RFE,
        args.ifSubsetData,
        args.max_workers,
        args.mim_class_label_threshold,
        args.classColumn
    )

    WIDTH = 215.9
    HEIGHT = 279.4
    pdf = FPDF()
    pdf.add_page()
    create_letterhead(pdf, WIDTH, args.letterhead)
    create_title("Feature Evaluation: {}".format(myLabel), pdf)
    if hshResults['too_few'] == "":
        write_to_pdf(pdf, "In-Variant Feature Threshold: {}".format(args.varThreshold))    
        pdf.ln(5)
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
        with open("top_rank_features_{}.csv".format(myLabel.replace(' ','_').replace('|','_').replace('/','')), 'w', newline='') as csvfile:
            f_writer = csv.writer(csvfile)
            f_writer.writerow(["Features"])
    pdf.output("{}_Features.pdf".format(myLabel.replace(' ','_').replace('|','_').replace('/','')), 'F')


