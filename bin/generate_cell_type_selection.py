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
from datetime import datetime
import json

import concurrent.futures
from functools import partial
from sklearn.exceptions import ConvergenceWarning

from pprint import pprint

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoLarsIC, Lasso
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.feature_selection import RFE, VarianceThreshold

batchColumn = 'Batch'

############################ PLOT AND TABLE GENERATION ############################
def create_binary_count_table(df, output_path):
    """Create binary count table and save to CSV"""
    binaryCntTbl = df.groupby([batchColumn, 'Lasso_Binary']).size().reset_index()
    binaryCntTbl.columns = ['Batch', 'Lasso_Binary', 'Count']
    binaryCntTbl.to_csv(output_path, index=False)
    print(f"Binary count table saved: {output_path}")
    return binaryCntTbl

def plot_best_alpha(scores, scores_std, alphas, best_alpha, n_folds, output_path):
    """Create best alpha plot and save to file"""
    plt.figure().set_size_inches(9, 6)
    plt.semilogx(alphas, scores)
    std_error = scores_std / np.sqrt(n_folds)
    plt.semilogx(alphas, scores + std_error, "b--")
    plt.semilogx(alphas, scores - std_error, "b--")
    plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)
    plt.axvline(best_alpha, linestyle="--", color="green", label="alpha: Best Fit")
    plt.ylabel("CV score +/- std error")
    plt.xlabel("alpha")
    plt.axhline(np.max(scores), linestyle="--", color=".5")
    plt.xlim([alphas[0], alphas[-1]])
    plt.title('Alpha Parameter Optimization')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Best alpha plot saved: {output_path}")

def plot_feature_ranking(featureRankDF, output_path):
    """Create feature ranking plot and save to file"""
    fig, ax = plt.subplots(figsize=(8, 12))
    top_features = featureRankDF.nlargest(35, columns="score").sort_values(by="score", ascending=True)
    bars = ax.barh(range(len(top_features)), top_features['score'], color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features.index)
    ax.set_xlabel('Feature Importance (|Coefficient|)')
    ax.set_title('Top 35 Feature Rankings from Lasso')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + max(top_features['score'])*0.01, bar.get_y() + bar.get_height()/2, 
               f'{width:.3f}', ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature ranking plot saved: {output_path}")

def create_rfe_summary_table(rfeTbl, output_path):
    """Create RFE summary table and save to CSV"""
    summary_df = rfeTbl.groupby('n_features')['rfe_score'].agg(['mean', 'std', 'median']).reset_index()
    summary_df.columns = ['n_features', 'mean_score', 'std_score', 'median_score']
    summary_df.to_csv(output_path, index=False)
    print(f"RFE summary table saved: {output_path}")
    return summary_df

def plot_recursive_elimination(rfeTbl, summary_df, output_path):
    """Create recursive feature elimination plot and save to file"""
    categories = sorted(rfeTbl['n_features'].unique())
    grouped_data = [rfeTbl[rfeTbl['n_features'] == cat]['rfe_score'] for cat in categories]

    fig, ax = plt.subplots(figsize=(10, 6))
    box = ax.boxplot(grouped_data, labels=categories, patch_artist=True, showmeans=True)

    # Calculate optimal number of features
    global_median = rfeTbl['rfe_score'].median()
    global_sd = (rfeTbl['rfe_score'].std() / 8)
    filtered = summary_df[summary_df['median_score'] >= (global_median - global_sd)]
    featureCutoff = int(filtered['n_features'].min())

    # Color the optimal box
    for patch, category in zip(box['boxes'], categories):
        if category == featureCutoff:
            patch.set_facecolor('lightgreen')
        else:
            patch.set_facecolor('white')
    
    for element in ['medians', 'means', 'whiskers', 'caps', 'fliers']:
        plt.setp(box[element], color='black')

    ax.set_xlabel('Number of Features')
    ax.set_ylabel('RFE Score')
    ax.set_title('Recursive Feature Elimination Analysis')
    ax.grid(True, alpha=0.3)
    
    # Add optimal feature count annotation
    ax.axvline(x=categories.index(featureCutoff) + 1, color='red', linestyle='--', alpha=0.7)
    ax.text(categories.index(featureCutoff) + 1, ax.get_ylim()[1] * 0.95, 
           f'Optimal: {featureCutoff}', ha='center', va='top', 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Recursive elimination plot saved: {output_path}")
    
    return featureCutoff

############################ PLOT AND TABLE GENERATION ############################

def get_lasso_classification_features(
    df, celltype, a, aTbl, rfeTbl, varThreshold, n_folds, n_features_to_RFE, 
    ifSubsetData, max_workers, mim_class_label_threshold, classColumn, output_prefix
):
    """Perform feature selection analysis and save outputs as separate files"""
    
    results = {
        'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'celltype': celltype
    }
    
    print(f"\n=== FEATURE SELECTION FOR {celltype} ===")
    print(df.groupby([batchColumn, 'Lasso_Binary']).size())
    
    # Create binary count table
    binary_table_path = f"{output_prefix}_binary_counts.csv"
    binary_table = create_binary_count_table(df, binary_table_path)
    results['binary_count_table_path'] = binary_table_path

    # Process alpha optimization results
    scores = aTbl["mean_test_score"].values
    scores_std = aTbl["std_test_score"].values
    alphas = list(aTbl["input_a"].str.split('-').str[0].astype(float))
    
    # Create alpha plot
    alpha_plot_path = f"{output_prefix}_alpha_optimization.png"
    plot_best_alpha(scores, scores_std, alphas, a, n_folds, alpha_plot_path)
    results['alpha_plot_path'] = alpha_plot_path

    # Prepare feature data
    XAll = df[list(df.select_dtypes(include=[np.number]).columns.values)]
    XAll = XAll[XAll.columns.drop(list(XAll.filter(regex='(Centroid|Binary|cnt|Name)')))].fillna(0)
    yAll = df['Lasso_Binary']

    # Variance threshold filtering
    sel = VarianceThreshold(threshold=varThreshold)
    sel.fit(XAll)
    nonVarFeatures = [x for x in XAll.columns if x not in XAll.columns[sel.get_support()]]
    print(f"Non-variant Features: {', '.join(nonVarFeatures)}")

    # Feature importance from Lasso
    clf = Lasso(alpha=a)
    clf.fit(XAll, yAll)
    features = XAll.columns.values.tolist()
    coefficients = clf.coef_
    importance = np.abs(coefficients)
    
    featureRankDF = pd.DataFrame(data=importance, index=features, columns=["score"])
    
    # Create feature ranking plot
    feature_ranking_path = f"{output_prefix}_feature_ranking.png"
    plot_feature_ranking(featureRankDF, feature_ranking_path)
    results['feature_ranking_plot_path'] = feature_ranking_path

    # Create feature importance dataframe
    dfF = pd.DataFrame(list(zip(features, importance)), columns=['Name', 'Feature_Importance'])
    dfF = dfF.sort_values(by=['Feature_Importance'], ascending=False)

    # Process RFE results
    print("\n=== PROCESSING RFE RESULTS ===")
    rfe_summary_path = f"{output_prefix}_rfe_summary.csv"
    summary_df = create_rfe_summary_table(rfeTbl, rfe_summary_path)
    results['rfe_summary_csv_path'] = rfe_summary_path

    # Create RFE plot and determine optimal features
    rfe_plot_path = f"{output_prefix}_rfe_analysis.png"
    featureCutoff = plot_recursive_elimination(rfeTbl, summary_df, rfe_plot_path)
    results['rfe_plot_path'] = rfe_plot_path
    results['optimal_n_features'] = int(featureCutoff)

    # Feature selection summary
    results['feature_selection_summary'] = {
        'original_features': len(features),
        'non_variant_removed': len(nonVarFeatures),
        'features_after_variance_filter': len(XAll.columns[sel.get_support()]),
    }

    return results

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
    args = parser.parse_args()

    # Load data
    myData = pd.read_pickle(args.trainingDataframe)
    myLabel = args.celltype.replace('[', '').replace(']', '')
    rfeScores = pd.read_csv(args.rfe_scores)
    best_alpha = args.best_alpha
    alphaScores = pd.read_csv(args.alpha_scores)

    # Create output prefix from celltype
    safe_celltype = myLabel.replace(' ', '_').replace('|', '_').replace('/', '')
    output_prefix = f"feature_selection_{safe_celltype}"

    # Run feature selection analysis
    feature_results = get_lasso_classification_features(
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
        args.classColumn,
        output_prefix
    )

    # Save comprehensive results as JSON
    json_path = f"{output_prefix}_results.json"
    with open(json_path, 'w') as f:
        json.dump(feature_results, f, indent=2)
    print(f"Results JSON saved: {json_path}")

    print(f"Cell type: {myLabel}")
    print(f"• Feature selection results JSON: {json_path}")
