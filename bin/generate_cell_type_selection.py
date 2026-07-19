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
from matplotlib.lines import Line2D
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
def plot_feature_ranking_with_cutoff(featureRankDF, output_path, cutoff_n, top_n=35, model_name="Lasso"):
    """
    Enhanced feature ranking plot with cutoff line and directionality coloring
    Blue = positive association, Red = negative association
    """
    # Prepare data
    top_features = featureRankDF.nlargest(top_n, columns="score").sort_values(by="score", ascending=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, max(10, top_n * 0.4)))
    
    # Determine colors based on directionality (positive=blue, negative=red)
    colors = []
    for idx, row in top_features.iterrows():
        if row['Coefficient'] > 0:
            colors.append('#1f77b4')  # Positive - blue
        else:
            colors.append('#d62728')  # Negative - red
    
    # Adjust alpha based on selection status
    alphas = []
    for i in range(len(top_features)):
        if i >= (top_n - cutoff_n):
            alphas.append(0.9)  # Selected features - more opaque
        else:
            alphas.append(0.4)  # Non-selected features - more transparent
    
    # Create bars with enhanced styling
    bars = ax.barh(range(len(top_features)), top_features['score'], 
                   color=colors, height=0.7)
    
    # Apply alpha values individually
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)
        bar.set_edgecolor('white')
        bar.set_linewidth(0.8)
    
    # Add cutoff line
    if cutoff_n <= top_n:
        cutoff_position = top_n - cutoff_n - 0.5
        ax.axhline(y=cutoff_position, color='black', linestyle='--', linewidth=2, 
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
    
    # Create legend with directionality explanation
    legend_elements = [
        mpatches.Patch(color='#1f77b4', alpha=0.9, label='Positive association (selected)'),
        mpatches.Patch(color='#d62728', alpha=0.9, label='Negative association (selected)'),
        mpatches.Patch(color='#1f77b4', alpha=0.4, label='Positive association (not selected)'),
        mpatches.Patch(color='#d62728', alpha=0.4, label='Negative association (not selected)'),
        Line2D([0], [0], color='black', linestyle='--', label='Selection cutoff')
    ]
    
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
              frameon=True, fancybox=True, shadow=True)
    
    # Add summary statistics box
    selected_features = top_features.tail(cutoff_n)
    mean_importance = selected_features['score'].mean()
    std_importance = selected_features['score'].std()
    n_positive = (selected_features['Coefficient'] > 0).sum()
    n_negative = (selected_features['Coefficient'] < 0).sum()
    
    stats_text = f'Selected features:\nMean: {mean_importance:.3f}\nStd: {std_importance:.3f}\n' \
                 f'Positive: {n_positive} | Negative: {n_negative}'
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

def plot_feature_expression_heatmap(df, featureRankDF, selected_features, celltype, 
                                   output_path, n_features=35, class_column='Lasso_Binary',
                                   normalization='zscore'):

    # Get top N features from feature ranking
    top_features_df = featureRankDF.nlargest(n_features, columns="score")
    
    # Sort by coefficient (positive first, then negative) for better visual clustering
    top_features_df = top_features_df.sort_values(by='Coefficient', ascending=False)
    top_features = top_features_df.index.tolist()
    
    # Filter to available columns
    available_features = [f for f in top_features if f in df.columns]
    
    if not available_features:
        print("Warning: No features available for heatmap")
        return None
    
    # Create a copy of the dataframe with selected features and class column
    df_subset = df[[class_column] + available_features].copy()
    
    # Apply normalization to the ORIGINAL data (before grouping)
    if normalization == 'zscore':
        # Z-score normalization: (x - mean) / std for each feature
        for feature in available_features:
            mean_val = df_subset[feature].mean()
            std_val = df_subset[feature].std()
            if std_val > 0:  # Avoid division by zero
                df_subset[feature] = (df_subset[feature] - mean_val) / std_val
        cbar_label = 'Mean Z-score (normalized per feature)'
        center_value = 0
        fmt_string = '.2f'        
    else:
        # No normalization
        cbar_label = 'Mean Expression Level'
        center_value = None  # Will be calculated from data
        fmt_string = '.2f'
    
    # NOW calculate mean expression by class (on normalized data)
    tile_data = df_subset.groupby(class_column)[available_features].mean()
    tile_data = tile_data.transpose()
    
    # Set center value for non-normalized data
    if center_value is None:
        center_value = tile_data.values.mean()
    
    # Enhanced dimensions
    width = 20
    height = max(12, len(tile_data.index) * 0.45)
    
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Create heatmap
    sns.heatmap(tile_data, annot=True, fmt=fmt_string, cmap='RdBu_r', 
                cbar_kws={'label': cbar_label},
                linewidths=0.5, linecolor='lightgray',
                ax=ax, center=center_value)
    
    # Add visual separator between positive and negative coefficients
    n_positive = (top_features_df['Coefficient'] > 0).sum()
    if 0 < n_positive < len(available_features):
        ax.axhline(y=n_positive, color='black', linestyle='-', linewidth=3, alpha=0.7)
        
        # Add simple labels centered above and below the line
        ax.text(tile_data.shape[1]/2, n_positive - 0.3, 
            'Positive Association ↑', 
            va='top', ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        
        ax.text(tile_data.shape[1]/2, n_positive + 0.3, 
            'Negative Association ↓', 
            va='bottom', ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
    
    # Customize y-axis labels with selection markers
    y_labels = []
    for feature in tile_data.index:
        if feature in selected_features:
            y_labels.append(f"{feature} *")
        else:
            y_labels.append(feature)
    
    ax.set_yticklabels(y_labels, fontsize=10, rotation=0)
    
    # Customize x-axis labels
    class_labels = [f"Class {int(col)}\n({'Negative' if col == 0 else celltype})" 
                   for col in tile_data.columns]
    ax.set_xticklabels(class_labels, fontsize=12, fontweight='bold')
    
    # Title with informative subtitle
    n_selected = len([f for f in available_features if f in selected_features])
    norm_text = f" | Normalization: {normalization}" if normalization != 'none' else ""
    title = f'Top {n_features} Features: Mean Expression by Class - {celltype}\n'
    subtitle = f'* = Selected features ({n_selected}/{len(available_features)}) | Features sorted by coefficient direction{norm_text}'
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, 
           ha='center', fontsize=10, style='italic')
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Top {n_features} Features (sorted by coefficient direction)', 
                 fontsize=12, fontweight='bold')
    
    # Improve layout
    plt.tight_layout()
    
    # Save with high quality
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Feature expression heatmap saved: {output_path}")
    return output_path

def plot_improved_rfe(rfeTbl, summary_df, output_path, optimal_n):
    """
    Improved RFE plot with better visualization
    """
    categories = sorted(rfeTbl['n_features'].unique())
    grouped_data = [rfeTbl[rfeTbl['n_features'] == cat]['rfe_score'] for cat in categories]
    
    # Create figure with better proportions for page width utilization
    fig, ax = plt.subplots(figsize=(16, 8))
    
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
                    alpha=0.2, color='lightgrey', label='±1 std')
    
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
    ax.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)
    
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
    fig, ax = plt.subplots(figsize=(16, 12))
    
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
        
        # Fix: Use iloc for pandas Series/DataFrame, direct indexing for numpy arrays
        if hasattr(y, 'iloc'):
            y_train = y.iloc[train_idx]  # Use iloc for pandas Series
        else:
            y_train = y[train_idx]  # Use direct indexing for numpy arrays
        
        # Simple variance-based selection for stability check
        variances = X_train.var()
        top_features_fold = variances.nlargest(len(available_features)).index.tolist()
        
        fold_selections.append(top_features_fold)
        for feat in top_features_fold:
            feature_frequency[feat] += 1
    
    # Calculate stability scores
    stability_scores = {feat: freq/n_folds * 100 for feat, freq in feature_frequency.items()}
    
    # Create stability plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
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

    #XAll = XAll[XAll.columns[sel.get_support()]]


    # Print model description
    print("\nModel description:")
    print(f"• Model type: Lasso regression")
    print(f"• Alpha parameter: {a}")
    print(f"• Number of input features: {XAll.shape[1]}")
    print(f"• Number of samples: {XAll.shape[0]}")
    print(f"• Target variable: Lasso_Binary")
    
    scaler = StandardScaler(with_mean=True)
    X_scaled = scaler.fit_transform(XAll)

    clf = Lasso(alpha=a)
    print(f"• Model object: {clf}")
    clf.fit(X_scaled, yAll)
    print("• Model fit complete.")

    features = XAll.columns.values.tolist()
    coefficients = clf.coef_
    importance = np.abs(coefficients)
    featureRankDF = pd.DataFrame(data={'score': importance, 'Coefficient': coefficients}, 
                             index=features)
    
    # Create feature importance dataframe
    dfF = pd.DataFrame(list(zip(features, importance, coefficients)), columns=['Name', 'Feature_Importance', 'Coefficient'])
    dfF = dfF.sort_values(by=['Feature_Importance'], ascending=False)
    dfF['Direction'] = dfF['Coefficient'].apply(lambda x: 'Positive' if x > 0 else 'Negative')

    dfF.to_csv("coefficients.csv", index=False)

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
    plot_feature_ranking_with_cutoff(featureRankDF, feature_ranking_path, featureCutoff, top_n=featureCutoff+5, model_name="Lasso")
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

    tile_plot_path = f"{output_prefix}_tileplot.png"
    plot_feature_expression_heatmap(df, featureRankDF, selected_features, 
                                celltype, tile_plot_path, n_features=featureCutoff+5)
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