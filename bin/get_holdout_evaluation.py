#!/usr/bin/env python3

import os, sys, re, random, math, time, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pprint import pprint
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap

import pickle
import xgboost as xgb

############################ PLOT AND TABLE GENERATION ############################

def create_class_distribution_plot(unique_names, counts, output_path):
    """Create improved class distribution bar chart and save to file"""
    # Sort by counts (descending order)
    sorted_indices = np.argsort(counts)[::-1]
    sorted_names = unique_names[sorted_indices]
    sorted_counts = counts[sorted_indices]
    
    # Create figure with better styling
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create horizontal bar chart with gradient colors
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sorted_names)))
    bars = ax.barh(sorted_names, sorted_counts, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # Styling to match theme
    ax.set_xlabel('Number of Samples', fontsize=14, fontweight='bold', color='#2c3e50')
    ax.set_ylabel('Cell Types', fontsize=14, fontweight='bold', color='#2c3e50')
    ax.set_title('Class Distribution in Holdout Dataset', fontsize=16, fontweight='bold', 
                 color='#2c3e50', pad=20)
    
    # Grid styling
    ax.grid(True, alpha=0.3, axis='x', linestyle='-', color='#bdc3c7')
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for bar, count in zip(bars, sorted_counts):
        width = bar.get_width()
        ax.text(width + max(sorted_counts)*0.01, bar.get_y() + bar.get_height()/2, 
               f'{count:,}', ha='left', va='center', fontweight='bold', 
               fontsize=11, color='#2c3e50')
    
    # Style the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')
    
    # Background color
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Class distribution plot saved: {output_path}")

def create_confusion_matrix_plot(cm_df, class_names, output_path):
    """Create larger, improved confusion matrix heatmap and save to file with percentile-based color scaling"""
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    # Enhanced figure size for better page utilization
    base_size = 3.0
    n_classes = len(class_names)
    figsize = (max(16, base_size * n_classes), max(12, base_size * n_classes))

    fig, ax = plt.subplots(figsize=figsize)


    colors = ['#ffffff', '#9ecae1', '#08519c']  # white → light blue → dark blue
    cmap = LinearSegmentedColormap.from_list('white_to_blue', colors, N=256)

    # Compute vmax as the 90th percentile of the matrix (excluding zeros for better scaling)
    non_zero_values = cm_df.values[cm_df.values > 0]
    if len(non_zero_values) > 0:
        vmax = np.percentile(non_zero_values, 90)
    else:
        vmax = cm_df.values.max()
    
    # Ensure vmax is at least 1 to avoid issues
    vmax = max(vmax, 1)
    norm = Normalize(vmin=0, vmax=vmax)

    # Determine annotation font size dynamically
    annot_font = max(12, 18 - n_classes // 2)

    # Create heatmap
    sns.heatmap(
        cm_df,
        annot=True,
        fmt='d',
        cmap=cmap,
        norm=norm,  # <- Use normalization
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Number of Predictions', 'shrink': 0.8},
        square=True,
        linewidths=0.5,
        linecolor='white',
        annot_kws={
            'fontsize': annot_font,
            'fontweight': 'bold',
            'color': '#2c3e50'
        }
    )

    # Colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_label('Number of Predictions', size=16, weight='bold', color='#2c3e50')
    cbar.ax.set_yticks([])

    # Labels and title
    ax.set_xlabel('Predicted Class', fontsize=16, fontweight='bold', color='#2c3e50')
    ax.set_ylabel('Actual Class', fontsize=16, fontweight='bold', color='#2c3e50')
    ax.set_title('Confusion Matrix', fontsize=20, fontweight='bold', color='#2c3e50', pad=25)

    # Rotate labels if needed
    if n_classes > 8:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    else:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Tick styling
    tick_fontsize = max(10, 16 - n_classes // 4)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, colors='#2c3e50')

    # Background and layout
    fig.patch.set_facecolor('white')
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Confusion matrix plot saved: {output_path}")



def create_roc_curves_plot(y_true_binarized, y_pred_scores, label_hash, n_classes, auc_scores, output_path):
    """Create improved ROC curves for all classes and save to file"""
    fig, ax = plt.subplots(figsize=(16, 12))
    
    if len(auc_scores) == 0:
        ax.text(0.5, 0.5, 'No ROC curves available\nCheck class distribution', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#bdc3c7'))
    else:
        # Sort by AUC score (descending)
        sorted_auc_scores = sorted(auc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Use better color palette
        colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(sorted_auc_scores))))
        if len(sorted_auc_scores) > 10:
            # Use viridis for more classes
            colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sorted_auc_scores)))
        
        for (class_index, roc_auc), color in zip(sorted_auc_scores, colors):
            if class_index in label_hash:
                fpr, tpr, _ = roc_curve(y_true_binarized[:, class_index], y_pred_scores[:, class_index])
                ax.plot(fpr, tpr, color=color, lw=2.5, 
                       label=f'{label_hash[class_index]} (AUC = {roc_auc:.3f})')
    
    # Add reference line
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.6, label='Random Classifier')
    
    # Styling
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold', color='#2c3e50')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold', color='#2c3e50')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves', 
                 fontsize=16, fontweight='bold', color='#2c3e50', pad=20)
    
    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='-', color='#bdc3c7')
    ax.set_axisbelow(True)
    
    # Improved legend
    legend = ax.legend(loc="lower right", fontsize=10, frameon=True, 
                      fancybox=True, shadow=True, framealpha=0.9)
    legend.get_frame().set_facecolor('#f8f9fa')
    legend.get_frame().set_edgecolor('#bdc3c7')
    
    # Style axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')
    ax.tick_params(axis='both', which='major', labelsize=12, colors='#2c3e50')
    
    # Background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"ROC curves plot saved: {output_path}")

def create_auc_rankings_table(sorted_auc_scores, label_hash, output_path):
    """Create AUC rankings table and save to CSV"""
    rankings_data = []
    for rank, (class_index, auc_score) in enumerate(sorted_auc_scores, 1):
        if class_index in label_hash:
            # Determine performance level
            if auc_score >= 0.9:
                performance = 'Excellent'
            elif auc_score >= 0.8:
                performance = 'Good'
            elif auc_score >= 0.7:
                performance = 'Fair'
            else:
                performance = 'Poor'
            
            rankings_data.append({
                'Rank': rank,
                'Class': label_hash[class_index],
                'AUC_Score': round(auc_score, 3),
                'Performance': performance
            })
    
    rankings_df = pd.DataFrame(rankings_data)
    rankings_df.to_csv(output_path, index=False)
    print(f"AUC rankings table saved: {output_path}")
    return rankings_df

def detect_class_imbalance(counts, threshold=0.1):
    """Detect if there's significant class imbalance"""
    total_samples = sum(counts)
    min_ratio = min(counts) / total_samples
    max_ratio = max(counts) / total_samples
    
    # Consider imbalanced if smallest class is less than threshold of total
    # or if ratio between largest and smallest is > 10:1
    return min_ratio < threshold or (max_ratio / min_ratio) > 10

def check_holdout(toCheckDF, xgbM, classColumn, leEncoderFile, output_prefix):
    """Evaluate model on holdout data and save outputs as separate files"""

    results = {
        'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    model_features = xgbM.feature_names

    X_holdout = toCheckDF[model_features]

    le = preprocessing.LabelEncoder()
    le.classes_ = np.load(leEncoderFile, allow_pickle=True)
    print(le.classes_.tolist())
    y_holdout = le.transform(toCheckDF[classColumn])

    # Make predictions
    dmatrix = xgb.DMatrix(X_holdout)
    y_pred_proba = xgbM.predict(dmatrix)

    #Convert probabilities to class labels
    y_pred = np.argmax(y_pred_proba, axis=1)

    """Plot ROC curve for binary or multiclass classification."""
    # Get unique values and their counts
    unique_values, counts = np.unique(y_holdout, return_counts=True)
    n_classes = len(unique_values)
    print(n_classes)
    uniqNames = le.inverse_transform(unique_values)

    accuracy = accuracy_score(y_holdout, y_pred)
    results['accuracy'] = float(accuracy)
    f1 = f1_score(y_holdout, y_pred, average='weighted')
    results['f1_score'] = float(f1)
    results['class_imbalance_detected'] = bool(detect_class_imbalance(counts))
    
    # Create class distribution plot
    class_distribution_plot = f"{output_prefix}_class_distribution.png"
    create_class_distribution_plot(uniqNames, counts, class_distribution_plot)
    results['class_distribution_plot_path'] = class_distribution_plot

    lableHash = dict(zip(unique_values, uniqNames))

    # Calculate confusion matrix
    cm = confusion_matrix(y_holdout, y_pred)
    cm_df = pd.DataFrame(cm, columns=uniqNames, index=uniqNames)

    confusion_matrix_plot = f"{output_prefix}_confusion_matrix.png"
    create_confusion_matrix_plot(cm_df, uniqNames, confusion_matrix_plot)
    results['confusion_matrix_csv_path'] = confusion_matrix_plot

    # Multiclass ROC/AUC computation and plotting
    def compute_multiclass_roc_auc(y_true, y_proba, n_classes):
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        
        auc_scores = {}
        fpr_dict, tpr_dict = {}, {}
        
        for i in range(n_classes):
            # Use actual probabilities for class i
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            auc_scores[i] = roc_auc
            fpr_dict[i] = fpr
            tpr_dict[i] = tpr
        
        return auc_scores, fpr_dict, tpr_dict, y_true_bin

    auc_scores, fpr_dict, tpr_dict, y_true_binarized = compute_multiclass_roc_auc(y_holdout, y_pred_proba, n_classes)
    sorted_auc_scores = sorted(auc_scores.items(), key=lambda x: x[1], reverse=True)

    def export_roc_plot(y_true_bin, y_pred_proba, label_hash, n_classes, auc_scores, output_path):
        plt.figure(figsize=(14, 10))
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2, label=f'{label_hash[i]} (AUC={roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC curves plot saved: {output_path}")

    roc_curves_plot = f"{output_prefix}_roc_curves.png"
    export_roc_plot(y_true_binarized, y_pred_proba, lableHash, n_classes, auc_scores, roc_curves_plot)
    results['roc_curves_plot_path'] = roc_curves_plot

    def export_auc_table(sorted_auc_scores, label_hash, output_path):
        rankings_data = []
        for rank, (class_index, auc_score) in enumerate(sorted_auc_scores, 1):
            if class_index in label_hash:
                if auc_score >= 0.9:
                    performance = 'Excellent'
                elif auc_score >= 0.8:
                    performance = 'Good'
                elif auc_score >= 0.7:
                    performance = 'Fair'
                else:
                    performance = 'Poor'
                rankings_data.append({
                    'Rank': rank,
                    'Class': label_hash[class_index],
                    'AUC_Score': round(auc_score, 3),
                    'Performance': performance
                })
        rankings_df = pd.DataFrame(rankings_data)
        rankings_df.to_csv(output_path, index=False)
        print(f"AUC rankings table saved: {output_path}")
        return rankings_df

    auc_table_path = f"{output_prefix}_auc_rankings.csv"
    auc_df = export_auc_table(sorted_auc_scores, lableHash, auc_table_path)
    results['auc_rankings_csv_path'] = auc_table_path

    # Prepare results data
    results.update({
        'max_auc': {'class_index': sorted_auc_scores[0][0], 'class_name': lableHash[sorted_auc_scores[0][0]], 'auc': float(sorted_auc_scores[0][1])},
        'min_auc': {'class_index': sorted_auc_scores[-1][0], 'class_name': lableHash[sorted_auc_scores[-1][0]], 'auc': float(sorted_auc_scores[-1][1])},
        'n_classes': int(n_classes),
        'total_samples': int(len(y_holdout)),
        'class_names': uniqNames.tolist(),
        'class_counts': counts.tolist(),
        'auc_scores': [{'class_index': int(idx), 'class_name': lableHash[idx], 'auc': float(score)} 
                      for idx, score in sorted_auc_scores]
    })
    
    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate XGBoost model on holdout data and generate PDF report.")
    parser.add_argument('--classColumn', required=True, help='Name of the classified column')
    parser.add_argument('--leEncoderFile', required=True, help='Path to label encoder .npy file')
    parser.add_argument('--model_pickle', required=True, help='Path to trained model pickle')
    parser.add_argument('--holdoutDataframe', required=True, help='Path to holdout dataframe pickle')
    parser.add_argument('--select_features_csv', required=True, help='Path to selected features CSV')
    args = parser.parse_args()

    classColumn = args.classColumn
    leEncoderFile = args.leEncoderFile

    with open(args.model_pickle, 'rb') as file:
        xgbMdl = pickle.load(file)

    with open(args.select_features_csv, 'r') as file:
        next(file)  # Skip header
        featureList = [line.strip() for line in file if line.strip()]
    if 'level_0' in featureList:
        featureList.remove('level_0')
    featureList.append(classColumn)

    prefix = f"holdoutEval_{os.path.splitext(os.path.basename(args.model_pickle))[0]}"

    #Read the directly use. Filtering to featureList already done upstream
    focusData = pd.read_pickle(args.holdoutDataframe)

    # Generate evaluation data and plots
    results = check_holdout(focusData, xgbMdl, classColumn, leEncoderFile, prefix)
    
    # Add metadata
    results['metadata'] = {
        'model_file': args.model_pickle,
        'encoder_file': args.leEncoderFile,
        'features_file': args.select_features_csv,
        'feature_count': len(featureList) - 1
    }
    
    # Save results as JSON
    with open(f"{prefix}_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Still create original CSV for compatibility
    performanceDF = pd.DataFrame({
        "Model": args.model_pickle, 
        "Accuracy": results['accuracy'], 
        "Max AUC": results['max_auc']['auc'], 
        "Min AUC": results['min_auc']['auc']
    }, index=[0])
    
    csv_path = f"holdout_{os.path.splitext(os.path.basename(args.model_pickle))[0]}.csv"
    performanceDF.to_csv(csv_path, index=False)
    print(f"Performance CSV saved: {csv_path}")