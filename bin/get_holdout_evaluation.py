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

import pickle
import xgboost as xgb

############################ PLOT AND TABLE GENERATION ############################

def create_class_distribution_plot(unique_names, counts, output_path):
    """Create class distribution bar chart and save to file"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create horizontal bar chart
    bars = ax.barh(unique_names, counts, color='steelblue', alpha=0.7)
    ax.set_xlabel('Number of Samples')
    ax.set_title('Class Distribution in Holdout Dataset')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        width = bar.get_width()
        ax.text(width + max(counts)*0.01, bar.get_y() + bar.get_height()/2, 
               f'{count}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Class distribution plot saved: {output_path}")

def create_confusion_matrix_plot(cm_df, class_names, output_path):
    """Create confusion matrix heatmap and save to file"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names,
               ax=ax, cbar_kws={'label': 'Number of Predictions'})
    
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual Class')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Confusion matrix plot saved: {output_path}")

def create_roc_curves_plot(y_true_binarized, y_pred_scores, label_hash, n_classes, auc_scores, output_path):
    """Create ROC curves for all classes and save to file"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if len(auc_scores) == 0:
        ax.text(0.5, 0.5, 'No ROC curves available\nCheck class distribution', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    else:
        # Sort by AUC score (descending)
        sorted_auc_scores = sorted(auc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Use distinct colors for all classes
        colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_auc_scores)))
        
        for (class_index, roc_auc), color in zip(sorted_auc_scores, colors):
            if class_index in label_hash:
                fpr, tpr, _ = roc_curve(y_true_binarized[:, class_index], y_pred_scores[:, class_index])
                ax.plot(fpr, tpr, color=color, lw=2, 
                       label=f'{label_hash[class_index]} (AUC = {roc_auc:.3f})')
    
    # Add reference line
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves')
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
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

    X_holdout = toCheckDF[list(toCheckDF.select_dtypes(include=[np.number]).columns.values)]
    X_holdout = X_holdout[xgbM.feature_names]

    le = preprocessing.LabelEncoder()
    le.classes_ = np.load(leEncoderFile, allow_pickle=True)
    print(le.classes_.tolist())
    y_holdout = le.transform(toCheckDF[classColumn])

    # Make predictions
    dmatrix = xgb.DMatrix(X_holdout)
    y_pred_proba = xgbM.predict(dmatrix)

    """Plot ROC curve for binary or multiclass classification."""
    # Get unique values and their counts
    unique_values, counts = np.unique(y_holdout, return_counts=True)
    n_classes = len(unique_values)
    print(n_classes)
    uniqNames = le.inverse_transform(unique_values)

    accuracy = accuracy_score(y_holdout, y_pred_proba)
    results['accuracy'] = float(accuracy)
    f1 = f1_score(y_holdout, y_pred_proba, average='weighted')
    results['f1_score'] = float(f1)
    results['class_imbalance_detected'] = bool(detect_class_imbalance(counts))
    
    # Create class distribution plot
    class_distribution_plot = f"{output_prefix}_class_distribution.png"
    create_class_distribution_plot(uniqNames, counts, class_distribution_plot)
    results['class_distribution_plot_path'] = class_distribution_plot

    lableHash = dict(zip(unique_values, uniqNames))

    # Calculate confusion matrix
    cm = confusion_matrix(y_holdout, y_pred_proba)
    cm_df = pd.DataFrame(cm, columns=uniqNames, index=uniqNames)

    confusion_matrix_plot = f"{output_prefix}_confusion_matrix.png"
    create_confusion_matrix_plot(cm_df, uniqNames, confusion_matrix_plot)
    results['confusion_matrix_csv_path'] = confusion_matrix_plot

    # Multiclass case
    y_true_binarized = label_binarize(y_holdout, classes=np.arange(n_classes))
    y_pred_binarized = label_binarize(y_pred_proba, classes=np.arange(n_classes))

    #fpr = dict()
    #tpr = dict()
    #roc_auc = dict()

    #for i in range(n_classes):
    #   fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_binarized[:, i])
    #   roc_auc[i] = auc(fpr[i], tpr[i])

    #plt.figure()
    #colors = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'yellow']
    #for i in range(n_classes):
    #   plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
    #   label=f'{lableHash[i]} (a={roc_auc[i]:.2f})')

    #plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.title('Receiver Operating Characteristic on Holdout')
    #plt.legend(loc="lower right")
    #plt.show()
    #plt.savefig("auc_curve_multiclass.png", dpi=300, bbox_inches='tight')

    # Compute AUC for each class
    auc_scores = {}
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_pred_binarized[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[i] = roc_auc

    # Rank AUC scores
    sorted_auc_scores = sorted(auc_scores.items(), key=lambda x: x[1], reverse=True)

    # Plot AUC for each class
    roc_curves_plot = f"{output_prefix}_roc_curves.png"
    create_roc_curves_plot(y_true_binarized, y_pred_binarized, lableHash, n_classes, auc_scores, roc_curves_plot)
    results['roc_curves_plot_path'] = roc_curves_plot

    # Create and save AUC rankings table
    auc_table_path = f"{output_prefix}_auc_rankings.csv"
    auc_df = create_auc_rankings_table(sorted_auc_scores, lableHash, auc_table_path)
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

    myData = pd.read_pickle(args.holdoutDataframe)
    with open(args.select_features_csv, 'r') as file:
        next(file)  # Skip header
        featureList = [line.strip() for line in file if line.strip()]
    if 'level_0' in featureList:
        featureList.remove('level_0')
    featureList.append(classColumn)
    focusData = myData[featureList]
    focusData = focusData.loc[:, ~focusData.columns.duplicated()]

    with open(args.model_pickle, 'rb') as file:
        xgbMdl = pickle.load(file)

    prefix = f"holdoutEval_{os.path.splitext(os.path.basename(args.model_pickle))[0]}"

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
    
    csv_path = f"holdout_{prefix}.csv"
    performanceDF.to_csv(csv_path, index=False)
    print(f"Performance CSV saved: {csv_path}")