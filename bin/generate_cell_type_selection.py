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


def plot_feature_ranking_with_cutoff(featureRankDF, output_path, cutoff_n, top_n=35, model_name="Lasso"):
    """
    Enhanced feature ranking plot with cutoff line showing selected features
    """
    # Prepare data
    top_features = featureRankDF.nlargest(top_n, columns="score").sort_values(by="score", ascending=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.3)))
    
    # Determine colors based on cutoff
    colors = []
    for i, (idx, row) in enumerate(top_features.iterrows()):
        if i >= (top_n - cutoff_n):
            colors.append('#1f77b4')  # Selected features - blue
        else:
            colors.append('#cccccc')  # Non-selected features - gray
    
    # Create bars with enhanced styling
    bars = ax.barh(range(len(top_features)), top_features['score'], 
                   color=colors, alpha=0.8, height=0.7)
    
    # Add subtle border
    for bar in bars:
        bar.set_edgecolor('white')
        bar.set_linewidth(0.8)
    
    # Add cutoff line
    if cutoff_n <= top_n:
        cutoff_position = top_n - cutoff_n - 0.5
        ax.axhline(y=cutoff_position, color='red', linestyle='--', linewidth=2, 
                  label=f'Selection cutoff (top {cutoff_n} features)')
    
    # Customize plot
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features.index, fontsize=10, fontweight='medium')
    ax.set_xlabel('Feature Importance Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Features - {model_name} (Selecting {cutoff_n})', 
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
    
    # Value labels
    max_score = top_features['score'].max()
    for i, (bar, score) in enumerate(zip(bars, top_features['score'])):
        width = bar.get_width()
        
        # Label positioning
        if width > max_score * 0.15:
            label_x = width - max_score * 0.02
            ha = 'right'
            color = 'white'
            weight = 'bold'
        else:
            label_x = width + max_score * 0.01
            ha = 'left'
            color = 'black'
            weight = 'normal'
        
        ax.text(label_x, bar.get_y() + bar.get_height()/2, 
               f'{width:.3f}', ha=ha, va='center', 
               fontsize=9, color=color, fontweight=weight)
    
    # Create legend
    legend_elements = [
        mpatches.Patch(color='#1f77b4', label=f'Selected ({cutoff_n} features)'),
        mpatches.Patch(color='#cccccc', label=f'Not selected'),
        mpatches.Line2D([0], [0], color='red', linestyle='--', label='Selection cutoff')
    ]
    
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
              frameon=True, fancybox=True, shadow=True)
    
    # Add summary statistics box
    selected_features = top_features.tail(cutoff_n)
    mean_importance = selected_features['score'].mean()
    std_importance = selected_features['score'].std()
    
    stats_text = f'Selected features stats:\nMean: {mean_importance:.3f}\nStd: {std_importance:.3f}'
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
    print(f"Feature ranking plot with cutoff saved: {output_path}")


def plot_improved_rfe(rfeTbl, summary_df, output_path, optimal_n):
    """
    Improved RFE plot with better visualization
    """
    categories = sorted(rfeTbl['n_features'].unique())
    grouped_data = [rfeTbl[rfeTbl['n_features'] == cat]['rfe_score'] for cat in categories]
    
    # Create figure with better proportions
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create violin plots for better distribution visualization
    parts = ax.violinplot(grouped_data, positions=range(len(categories)), 
                          widths=0.7, showmeans=True, showextrema=True, showmedians=True)
    
    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        if categories[i] == optimal_n:
            pc.set_facecolor('#90EE90')  # Light green for optimal
            pc.set_alpha(0.8)
        else:
            pc.set_facecolor('#87CEEB')  # Sky blue for others
            pc.set_alpha(0.6)
    
    # Style the violin plot elements
    parts['cmeans'].set_color('red')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.5)
    parts['cbars'].set_color('gray')
    parts['cmaxes'].set_color('gray')
    parts['cmins'].set_color('gray')
    
    # Add trend line with confidence interval
    mean_scores = [np.mean(data) for data in grouped_data]
    std_scores = [np.std(data) for data in grouped_data]
    x_pos = range(len(categories))
    
    ax.plot(x_pos, mean_scores, 'b-', linewidth=2, alpha=0.7, label='Mean score')
    ax.fill_between(x_pos, 
                    [m - s for m, s in zip(mean_scores, std_scores)],
                    [m + s for m, s in zip(mean_scores, std_scores)],
                    alpha=0.2, color='blue', label='±1 std')
    
    # Highlight optimal region
    optimal_idx = categories.index(optimal_n)
    ax.axvspan(optimal_idx - 0.5, optimal_idx + 0.5, alpha=0.3, color='green', 
              label=f'Optimal ({optimal_n} features)')
    
    # Add optimal marker
    ax.scatter([optimal_idx], [mean_scores[optimal_idx]], s=200, c='red', 
              marker='*', zorder=5, label='Selected configuration')
    
    # Customize axes
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('RFE Score', fontsize=12, fontweight='bold')
    ax.set_title('Recursive Feature Elimination Analysis', fontsize=16, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Improve spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add annotations
    best_score = mean_scores[optimal_idx]
    ax.annotate(f'Optimal: {optimal_n} features\nScore: {best_score:.3f}',
                xy=(optimal_idx, best_score),
                xytext=(optimal_idx + len(categories)*0.1, best_score + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Add legend
    ax.legend(loc='best', fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save with high quality
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Improved RFE plot saved: {output_path}")
    
    return optimal_n


def create_feature_correlation_matrix(df, selected_features, output_path, celltype):
    """
    Create correlation matrix heatmap for selected features
    """
    if len(selected_features) == 0:
        print("No selected features for correlation matrix")
        return None
    
    # Filter dataframe to only selected features
    available_features = [f for f in selected_features if f in df.columns]
    if len(available_features) < 2:
        print("Not enough features available for correlation matrix")
        return None
    
    # Limit to top 20 features for visibility
    features_to_plot = available_features[:min(20, len(available_features))]
    
    # Calculate correlation matrix
    corr_matrix = df[features_to_plot].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', vmin=-1, vmax=1, center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax)
    
    # Customize plot
    ax.set_title(f'Feature Correlation Matrix - {celltype}\n(Top {len(features_to_plot)} selected features)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Rotate labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Correlation matrix saved: {output_path}")
    
    return corr_matrix


def analyze_feature_stability(df, selected_features, n_folds, output_path, celltype):
    """
    Analyze feature selection stability across CV folds
    """
    if len(selected_features) == 0:
        print("No selected features for stability analysis")
        return None
    
    from sklearn.model_selection import StratifiedKFold
    from collections import defaultdict
    
    # Prepare data
    available_features = [f for f in selected_features if f in df.columns]
    if len(available_features) == 0:
        return None
    
    X = df[available_features]
    y = df['Lasso_Binary'] if 'Lasso_Binary' in df.columns else np.zeros(len(df))
    
    # Track feature selection frequency
    feature_frequency = defaultdict(int)
    fold_selections = []
    
    # Perform CV fold analysis
    skf = StratifiedKFold(n_splits=min(n_folds, 5), shuffle=True, random_state=42)
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train = X.iloc[train_idx]
        y_train = y[train_idx] if hasattr(y, 'iloc') else y[train_idx]
        
        # Simple variance-based selection for stability check
        variances = X_train.var()
        top_features_fold = variances.nlargest(len(available_features)).index.tolist()
        
        fold_selections.append(top_features_fold)
        for feat in top_features_fold:
            feature_frequency[feat] += 1
    
    # Calculate stability scores
    stability_scores = {feat: freq/n_folds * 100 for feat, freq in feature_frequency.items()}
    
    # Create stability plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort features by stability
    sorted_features = sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)
    features_to_plot = sorted_features[:min(30, len(sorted_features))]
    
    feature_names = [f[0] for f in features_to_plot]
    scores = [f[1] for f in features_to_plot]
    
    # Color based on stability level
    colors = []
    for score in scores:
        if score >= 80:
            colors.append('#2ecc71')  # Green - highly stable
        elif score >= 60:
            colors.append('#f39c12')  # Orange - moderately stable
        else:
            colors.append('#e74c3c')  # Red - unstable
    
    # Create bar plot
    bars = ax.bar(range(len(feature_names)), scores, color=colors, alpha=0.7)
    
    # Customize plot
    ax.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Selection Frequency (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Selection Stability Analysis - {celltype}\n(Frequency across {n_folds} CV folds)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis labels
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
    
    # Add horizontal line at 60% (stability threshold)
    ax.axhline(y=60, color='gray', linestyle='--', alpha=0.5, label='Stability threshold (60%)')
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.0f}%', ha='center', va='bottom', fontsize=8)
    
    # Create legend
    legend_elements = [
        mpatches.Patch(color='#2ecc71', label='Highly stable (≥80%)'),
        mpatches.Patch(color='#f39c12', label='Moderately stable (60-80%)'),
        mpatches.Patch(color='#e74c3c', label='Unstable (<60%)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Improve spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Feature stability analysis saved: {output_path}")
    
    return stability_scores


def create_rfe_summary_table(rfeTbl, output_path):
    """Create RFE summary table and save to CSV"""
    summary_df = rfeTbl.groupby('n_features')['rfe_score'].agg(['mean', 'std', 'median']).reset_index()
    summary_df.columns = ['n_features', 'mean_score', 'std_score', 'median_score']
    summary_df.to_csv(output_path, index=False)
    print(f"RFE summary table saved: {output_path}")
    return summary_df


############################ MAIN FEATURE SELECTION FUNCTION ############################

def get_lasso_classification_features(
    df, celltype, a, aTbl, rfeTbl, varThreshold, n_folds, n_features_to_RFE, 
    ifSubsetData, max_workers, mim_class_label_threshold, classColumn, output_prefix
):
    """Perform feature selection analysis and save outputs as separate files"""
    
    results = {
        'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'celltype': celltype,
        'n_folds': n_folds,  # Add CV folds to results
        'cv_folds': n_folds  # Alternative name for compatibility
    }
    
    print(f"\n=== FEATURE SELECTION FOR {celltype} ===")
    print(df.groupby([batchColumn, 'Lasso_Binary']).size())

    # Prepare feature data
    XAll = df[list(df.select_dtypes(include=[np.number]).columns.values)]
    XAll = XAll[XAll.columns.drop(list(XAll.filter(regex='(Centroid|Binary|cnt|Name)')))].fillna(0)
    yAll = df['Lasso_Binary']

    # Print input summary
    print("\nInput summary:")
    print(f"• Data shape: {df.shape}")
    print(f"• Celltype: {celltype}")
    print(f"• Alpha value: {a}")
    print(f"• Alpha scores shape: {aTbl.shape}")
    print(f"• RFE scores shape: {rfeTbl.shape}")
    print(f"• Variance threshold: {varThreshold}")
    print(f"• n_folds: {n_folds}")
    print(f"• n_features_to_RFE: {n_features_to_RFE}")
    print(f"• ifSubsetData: {ifSubsetData}")
    print(f"• max_workers: {max_workers}")
    print(f"• mim_class_label_threshold: {mim_class_label_threshold}")
    print(f"• classColumn: {classColumn}")
    print(f"• output_prefix: {output_prefix}")

    # Variance threshold filtering
    sel = VarianceThreshold(threshold=varThreshold)
    sel.fit(XAll)
    nonVarFeatures = [x for x in XAll.columns if x not in XAll.columns[sel.get_support()]]
    print(f"Non-variant Features: {', '.join(nonVarFeatures)}")

    # Print model description
    print("\nModel description:")
    print(f"• Model type: Lasso regression")
    print(f"• Alpha parameter: {a}")
    print(f"• Number of input features: {XAll.shape[1]}")
    print(f"• Number of samples: {XAll.shape[0]}")
    print(f"• Target variable: Lasso_Binary")
    clf = Lasso(alpha=a)
    print(f"• Model object: {clf}")
    clf.fit(XAll, yAll)
    print("• Model fit complete.")

    features = XAll.columns.values.tolist()
    coefficients = clf.coef_
    importance = np.abs(coefficients)
    featureRankDF = pd.DataFrame(data=importance, index=features, columns=["score"])
    
    # Create feature importance dataframe
    dfF = pd.DataFrame(list(zip(features, importance)), columns=['Name', 'Feature_Importance'])
    dfF = dfF.sort_values(by=['Feature_Importance'], ascending=False)

    # Process RFE results
    print("\n=== PROCESSING RFE RESULTS ===")
    rfe_summary_path = f"{output_prefix}_rfe_summary.csv"
    summary_df = create_rfe_summary_table(rfeTbl, rfe_summary_path)
    results['rfe_summary_csv_path'] = rfe_summary_path

    # Calculate optimal number of features
    global_median = rfeTbl['rfe_score'].median()
    global_sd = (rfeTbl['rfe_score'].std() / 8)
    filtered = summary_df[summary_df['median_score'] >= (global_median - global_sd)]
    featureCutoff = int(filtered['n_features'].min())
    results['optimal_n_features'] = int(featureCutoff)
    
    # Check for RFE warning (2% threshold)
    total_features = len(features)
    min_features_threshold = max(3, int(total_features * 0.02))
    if featureCutoff < min_features_threshold:
        print(f"\n⚠️ WARNING: Selected features ({featureCutoff}) is below 2% threshold ({min_features_threshold})")
        print(f"   This may lead to overfitting. Consider increasing the feature count.")
        results['rfe_warning'] = True
        results['min_features_threshold'] = min_features_threshold
    else:
        results['rfe_warning'] = False
        results['min_features_threshold'] = min_features_threshold

    # Create improved RFE plot
    rfe_plot_path = f"{output_prefix}_rfe_analysis.png"
    plot_improved_rfe(rfeTbl, summary_df, rfe_plot_path, featureCutoff)
    results['rfe_plot_path'] = rfe_plot_path

    # Get selected features
    selected_features = dfF['Name'].tolist()[:featureCutoff]
    results['selected_features'] = selected_features
    results['selected_features_count'] = len(selected_features)
    
    # Save selected features with their importance scores
    feature_importance_data = {}
    for idx, row in dfF.head(featureCutoff).iterrows():
        feature_importance_data[row['Name']] = float(row['Feature_Importance'])
    results['feature_importance_data'] = feature_importance_data

    # Create feature ranking plot with cutoff line
    feature_ranking_path = f"{output_prefix}_feature_ranking.png"
    plot_feature_ranking_with_cutoff(featureRankDF, feature_ranking_path, featureCutoff, top_n=35, model_name="Lasso")
    results['feature_ranking_plot_path'] = feature_ranking_path

    # Create feature correlation matrix
    correlation_plot_path = f"{output_prefix}_correlation_matrix.png"
    corr_matrix = create_feature_correlation_matrix(df, selected_features, correlation_plot_path, celltype)
    results['correlation_matrix_path'] = correlation_plot_path
    
    # Save correlation data
    if corr_matrix is not None:
        correlation_data = {}
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                feat1 = corr_matrix.index[i]
                feat2 = corr_matrix.columns[j]
                correlation_data[f"{feat1}_vs_{feat2}"] = float(corr_matrix.iloc[i, j])
        results['feature_correlations'] = correlation_data

    # Analyze feature stability
    stability_plot_path = f"{output_prefix}_stability_analysis.png"
    stability_scores = analyze_feature_stability(df, selected_features, n_folds, stability_plot_path, celltype)
    results['stability_analysis_path'] = stability_plot_path
    
    if stability_scores:
        results['feature_stability'] = {k: float(v) for k, v in stability_scores.items()}

    # Generate tile plot (heatmap) of selected features vs Lasso_Binary
    print("\nGenerating tile plot (heatmap) of selected features vs Lasso_Binary...")
    
    # Use only selected features for the tile plot
    available_selected = [f for f in selected_features if f in df.columns]
    if available_selected:
        tile_data = df.groupby('Lasso_Binary')[available_selected].mean()
        tile_data = tile_data.transpose()
        
        # Calculate dimensions
        max_width, max_height = 15, 12
        min_width, min_height = 8, 6
        
        width = min(max_width, max(min_width, len(tile_data.columns) * 2))
        height = min(max_height, max(min_height, len(tile_data.index) * 0.4))
        
        # Always show annotations for selected features (should be manageable)
        plt.figure(figsize=(width, height))
        sns.heatmap(tile_data, annot=True, fmt='.2f', cmap='viridis', cbar=True)
        plt.title(f'Mean Values of Selected Features by Lasso_Binary - {celltype}')
        plt.xlabel('Lasso_Binary')
        plt.ylabel('Selected Features')
        tile_plot_path = f"{output_prefix}_tileplot.png"
        plt.tight_layout()
        plt.savefig(tile_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Tile plot saved: {tile_plot_path}")
        results['tile_plot_path'] = tile_plot_path

    # Feature selection summary
    results['feature_selection_summary'] = {
        'original_features': len(features),
        'non_variant_removed': len(nonVarFeatures),
        'features_after_variance_filter': len(XAll.columns[sel.get_support()]),
        'optimal_features_selected': featureCutoff,
        'reduction_percentage': round((1 - featureCutoff/len(features)) * 100, 2)
    }

    # Save selected features to CSV
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

    print(f"\n=== FEATURE SELECTION COMPLETE ===")
    print(f"Cell type: {myLabel}")
    print(f"• Selected features: {feature_results['selected_features_count']}")
    print(f"• RFE warning: {feature_results.get('rfe_warning', False)}")
    print(f"• Feature selection results JSON: {json_path}")