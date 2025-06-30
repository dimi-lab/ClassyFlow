#!/usr/bin/env python3

import os, sys, re, csv, time, warnings
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
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


def plot_feature_ranking(featureRankDF, output_path, top_n=35, model_name="XGBoost"):
    """
    Advanced version with color-coded importance levels and enhanced styling
    """
    
    # Prepare data
    top_features = featureRankDF.nlargest(top_n, columns="score").sort_values(by="score", ascending=True)
    
    # Create figure - single plot with better proportions
    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.3)))
    
    # Color mapping based on percentile
    percentiles = np.percentile(top_features['score'], [25, 50, 75])
    
    def get_color(score):
        if score >= percentiles[2]:
            return '#1f77b4'  # High importance - blue
        elif score >= percentiles[1]:
            return '#ff7f0e'  # Medium importance - orange  
        elif score >= percentiles[0]:
            return '#2ca02c'  # Low-medium importance - green
        else:
            return '#d62728'  # Low importance - red
    
    colors = [get_color(score) for score in top_features['score']]
    
    # Create bars with enhanced styling
    bars = ax.barh(range(len(top_features)), top_features['score'], 
                   color=colors, alpha=0.8, height=0.7)
    
    # Add subtle border and shadow effect
    for i, bar in enumerate(bars):
        bar.set_edgecolor('white')
        bar.set_linewidth(0.8)
        
        # Add subtle shadow
        shadow = Rectangle((0, bar.get_y() + 0.05), bar.get_width(), bar.get_height(),
                          facecolor='gray', alpha=0.1, zorder=bar.get_zorder()-1)
        ax.add_patch(shadow)
    
    # Customize plot
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features.index, fontsize=10, fontweight='medium')
    ax.set_xlabel('Feature Importance Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Most Important Features - {model_name}', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Enhanced grid and styling
    ax.grid(True, alpha=0.3, axis='x', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_facecolor('#fafafa')
    
    # Improve spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('lightgray')
    ax.spines['bottom'].set_color('lightgray')
    
    # Value labels with smart positioning
    max_score = top_features['score'].max()
    for i, (bar, score) in enumerate(zip(bars, top_features['score'])):
        width = bar.get_width()
        
        # Choose label position based on bar width
        if width > max_score * 0.15:
            # Label inside bar if it's wide enough
            label_x = width - max_score * 0.02
            ha = 'right'
            color = 'white'
            weight = 'bold'
        else:
            # Label outside bar if it's narrow
            label_x = width + max_score * 0.01
            ha = 'left'
            color = 'black'
            weight = 'normal'
        
        ax.text(label_x, bar.get_y() + bar.get_height()/2, 
               f'{width:.3f}', ha=ha, va='center', 
               fontsize=9, color=color, fontweight=weight)
    
    # Create legend for importance levels
    legend_elements = [
        mpatches.Patch(color='#1f77b4', label=f'High (≥{percentiles[2]:.3f})'),
        mpatches.Patch(color='#ff7f0e', label=f'Med-High ({percentiles[1]:.3f}-{percentiles[2]:.3f})'),
        mpatches.Patch(color='#2ca02c', label=f'Med-Low ({percentiles[0]:.3f}-{percentiles[1]:.3f})'),
        mpatches.Patch(color='#d62728', label=f'Low (<{percentiles[0]:.3f})')
    ]
    
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
              frameon=True, fancybox=True, shadow=True)
    
    # Add summary statistics box
    mean_importance = top_features['score'].mean()
    std_importance = top_features['score'].std()
    
    stats_text = f'Mean: {mean_importance:.3f}\nStd: {std_importance:.3f}'
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
            fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.6))
    
    # Set margins
    ax.margins(x=0.1)
    
    # Tight layout with padding
    plt.tight_layout(pad=2.0)
    
    # Save with high quality
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

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

    ctl = dfF['Name'].tolist()[:featureCutoff]  
    with open("top_rank_features_{}.csv".format(celltype.replace(' ','_').replace('|','_').replace('/','')), 'w', newline='') as csvfile:
        f_writer = csv.writer(csvfile)
        f_writer.writerow(["Features"])
        for ln in ctl:
            f_writer.writerow([ln])

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
