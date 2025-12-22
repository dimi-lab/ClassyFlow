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


def create_confusion_matrix_div(cm_df, class_names, output_path):
    """Create Plotly confusion matrix and save div to file"""
    import numpy as np
    import plotly.graph_objects as go
    
    n_classes = len(class_names)
    values = cm_df.values
    
    vmax = values.max() if values.max() > 0 else 1
    use_log = vmax > 100
    
    if use_log:
        color_values = np.log10(np.maximum(values, 0.1))
        colorbar_title = 'Count (log₁₀)'
    else:
        color_values = values
        colorbar_title = 'Count'
    
    if n_classes <= 25:
        text = [[str(v) if v > 0 else '' for v in row] for row in values]
        texttemplate = '%{text}'
    else:
        text = None
        texttemplate = None
    
    fig = go.Figure(data=go.Heatmap(
        z=color_values,
        x=class_names,
        y=class_names,
        text=text,
        texttemplate=texttemplate,
        textfont={'size': max(8, min(14, 16 - n_classes // 3))},
        colorscale='Blues',
        hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>' if text else
                      'Actual: %{y}<br>Predicted: %{x}<extra></extra>',
        colorbar=dict(title=colorbar_title, thickness=15),
    ))
    
    fig_width = max(700, n_classes * 50)
    fig_height = max(600, n_classes * 40)
    max_label_len = max(len(str(name)) for name in class_names)
    bottom_margin = max(120, max_label_len * 6)
    
    fig.update_layout(
    title=dict(text='Confusion Matrix', x=0.5, font=dict(size=16)),
    xaxis=dict(
        title=dict(text='Predicted', standoff=20),
        tickangle=-45,
        tickfont=dict(size=max(8, min(12, 14 - n_classes // 4))),
        side='bottom'
    ),
    yaxis=dict(
        title='Actual',
        tickfont=dict(size=max(8, min(12, 14 - n_classes // 4))),
        autorange='reversed'
    ),
    margin=dict(l=100, r=40, t=60, b=bottom_margin),
    plot_bgcolor='white',
    autosize=True
)
    
    div_html = fig.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})
    
    with open(output_path, 'w') as f:
        f.write(div_html)
    
    print(f"Confusion matrix div saved: {output_path}")


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

    confusion_matrix_plot = f"{output_prefix}_confusion_matrix.html"
    create_confusion_matrix_div(cm_df, uniqNames, confusion_matrix_plot)
    results['confusion_matrix_html_path'] = confusion_matrix_plot

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
        """Create interactive ROC curves using Plotly"""
        import numpy as np
        import plotly.graph_objects as go
        from sklearn.metrics import roc_curve, auc
        
        fig = go.Figure()
        
        # Generate colors
        colors = [f'hsl({int(i * 360 / n_classes)}, 70%, 50%)' for i in range(n_classes)]
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'{label_hash[i]} (AUC={roc_auc:.2f})',
                line=dict(color=colors[i], width=2),
                hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra>%{fullData.name}</extra>'
            ))
        
        # Random classifier diagonal
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='black', width=2, dash='dash'),
            opacity=0.5
        ))
        
        fig.update_layout(
            title=dict(text='Receiver Operating Characteristic (ROC) Curves', x=0.5, font=dict(size=16)),
            xaxis=dict(
                title='False Positive Rate',
                range=[0, 1],
                gridcolor='lightgray',
                gridwidth=1
            ),
            yaxis=dict(
                title='True Positive Rate',
                range=[0, 1.05],
                gridcolor='lightgray',
                gridwidth=1
            ),
            legend=dict(
                x=1.02,
                y=0.5,
                yanchor='middle',
                font=dict(size=10)
            ),
            margin=dict(l=60, r=200, t=60, b=60),
            plot_bgcolor='white',
            autosize=True
        )
        
        div_html = fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={'responsive': True}
        )
        
        with open(output_path, 'w') as f:
            f.write(div_html)
        
        print(f"ROC curves plot saved: {output_path}")

    roc_curves_plot = f"{output_prefix}_roc_curves.html"
    export_roc_plot(y_true_binarized, y_pred_proba, lableHash, n_classes, auc_scores, roc_curves_plot)
    results['roc_curves_plot_path'] = roc_curves_plot

    def export_pr_plot(y_true_bin, y_pred_proba, label_hash, n_classes, output_path):
        """Create interactive Precision-Recall curves using Plotly"""
        import numpy as np
        import plotly.graph_objects as go
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        fig = go.Figure()
        
        colors = [f'hsl({int(i * 360 / n_classes)}, 70%, 50%)' for i in range(n_classes)]
        
        ap_scores = {}
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
            ap = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
            ap_scores[i] = ap
            
            fig.add_trace(go.Scatter(
                x=recall,
                y=precision,
                mode='lines',
                name=f'{label_hash[i]} (AP={ap:.2f})',
                line=dict(color=colors[i], width=2),
                hovertemplate='Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra>%{fullData.name}</extra>'
            ))
        
        fig.update_layout(
            title=dict(text='Precision-Recall Curves', x=0.5, font=dict(size=16)),
            xaxis=dict(
                title='Recall',
                range=[0, 1],
                gridcolor='lightgray',
                gridwidth=1
            ),
            yaxis=dict(
                title='Precision',
                range=[0, 1.05],
                gridcolor='lightgray',
                gridwidth=1
            ),
            legend=dict(
                x=1.02,
                y=0.5,
                yanchor='middle',
                font=dict(size=10)
            ),
            margin=dict(l=60, r=200, t=60, b=60),
            plot_bgcolor='white',
            autosize=True
        )
        
        div_html = fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={'responsive': True}
        )
        
        with open(output_path, 'w') as f:
            f.write(div_html)
        
        print(f"PR curves plot saved: {output_path}")
        
        return ap_scores
    
    pr_curves_plot = f"{output_prefix}_pr_curves.html"
    ap_scores = export_pr_plot(y_true_binarized, y_pred_proba, lableHash, n_classes, pr_curves_plot)
    results['pr_curves_plot_path'] = pr_curves_plot

    # def export_auc_table(sorted_auc_scores, label_hash, output_path):
    #     rankings_data = []
    #     for rank, (class_index, auc_score) in enumerate(sorted_auc_scores, 1):
    #         if class_index in label_hash:
    #             if auc_score >= 0.9:
    #                 performance = 'Excellent'
    #             elif auc_score >= 0.8:
    #                 performance = 'Good'
    #             elif auc_score >= 0.7:
    #                 performance = 'Fair'
    #             else:
    #                 performance = 'Poor'
    #             rankings_data.append({
    #                 'Rank': rank,
    #                 'Class': label_hash[class_index],
    #                 'AUC_Score': round(auc_score, 3),
    #                 'Performance': performance
    #             })
    #     rankings_df = pd.DataFrame(rankings_data)
    #     rankings_df.to_csv(output_path, index=False)
    #     print(f"AUC rankings table saved: {output_path}")
    #     return rankings_df

    # auc_table_path = f"{output_prefix}_auc_rankings.csv"
    # auc_df = export_auc_table(sorted_auc_scores, lableHash, auc_table_path)
    # results['auc_rankings_csv_path'] = auc_table_path

    def export_auc_table(sorted_auc_scores, ap_scores, label_hash, class_prevalence, output_path):
        """Export AUC and AP scores with context-aware performance labels"""
        rankings_data = []
        for rank, (class_index, auc_score) in enumerate(sorted_auc_scores, 1):
            if class_index in label_hash:
                ap = ap_scores.get(class_index, None)
                prevalence = class_prevalence.get(class_index, 0)
                
                # AUC performance (fixed thresholds)
                if auc_score >= 0.9:
                    auc_perf = 'Excellent'
                elif auc_score >= 0.8:
                    auc_perf = 'Good'
                elif auc_score >= 0.7:
                    auc_perf = 'Fair'
                else:
                    auc_perf = 'Poor'
                
                # AP performance (relative to baseline)
                # AP baseline for random classifier = prevalence
                if ap is not None and prevalence > 0:
                    ap_lift = ap / prevalence  # How much better than random
                    if ap_lift >= 10:
                        ap_perf = 'Excellent'
                    elif ap_lift >= 5:
                        ap_perf = 'Good'
                    elif ap_lift >= 2:
                        ap_perf = 'Fair'
                    else:
                        ap_perf = 'Poor'
                else:
                    ap_perf = 'N/A'
                
                rankings_data.append({
                    'Rank': rank,
                    'Class': label_hash[class_index],
                    'Prevalence': round(prevalence, 3),
                    'AUC': round(auc_score, 3),
                    'AUC_Performance': auc_perf,
                    'AP': round(ap, 3) if ap else None,
                    'AP_Performance': ap_perf
                })

        rankings_df = pd.DataFrame(rankings_data)
        rankings_df.to_csv(output_path, index=False)
        print(f"AUC/AP rankings table saved: {output_path}")
        return rankings_df

    class_prevalence = {i: count / len(y_holdout) for i, count in zip(unique_values, counts)}

    auc_table_path = f"{output_prefix}_auc_rankings.csv"
    auc_df = export_auc_table(sorted_auc_scores, ap_scores, lableHash, class_prevalence, auc_table_path)
    
    # Prepare results data
    results.update({
        'max_auc': {'class_index': sorted_auc_scores[0][0], 'class_name': lableHash[sorted_auc_scores[0][0]], 'auc': float(sorted_auc_scores[0][1])},
        'min_auc': {'class_index': sorted_auc_scores[-1][0], 'class_name': lableHash[sorted_auc_scores[-1][0]], 'auc': float(sorted_auc_scores[-1][1])},
        'n_classes': int(n_classes),
        'total_samples': int(len(y_holdout)),
        'class_names': uniqNames.tolist(),
        'class_counts': counts.tolist(),
        'auc_scores': [{'class_index': int(idx), 'class_name': lableHash[idx], 'auc': float(score)} 
                      for idx, score in sorted_auc_scores],
        'ap_scores': [{'class_name': lableHash[k], 'ap': float(v)} for k, v in ap_scores.items()]
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