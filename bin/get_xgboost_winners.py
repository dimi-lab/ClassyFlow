#!/usr/bin/env python3

import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn import preprocessing
import xgboost as xgb
from datetime import datetime
import json
import os

def plot_parameter_search(df, top_models, output_path):
    """Create parameter search boxplot using Plotly"""
    import plotly.graph_objects as go

    best_color = "#e74c3c"
    second_color = "#f39c12"
    
    df = df.copy()
    df['combination'] = df.apply(lambda row: f"max_depth={int(row['max_depth'])}, eta={row['eta']}", axis=1)
    
    # Sort by max_depth, then eta
    df_sorted = df.sort_values(['max_depth', 'eta'])
    ordered_combs = df_sorted['combination'].unique().tolist()
    
    # Get top 2 combinations
    top_2_combs = [f"max_depth={int(row['max_depth'])}, eta={row['eta']}" for _, row in top_models.head(2).iterrows()]
    max_comb = top_2_combs[0]
    second_max_comb = top_2_combs[1] if len(top_2_combs) > 1 else None
    
    # Track where max_depth changes
    max_depths_ordered = [df_sorted[df_sorted['combination'] == comb]['max_depth'].iloc[0] for comb in ordered_combs]
    
    fig = go.Figure()
    
    for comb in ordered_combs:
        y_values = df[df['combination'] == comb]['testf']
        
        if comb == max_comb:
            color = best_color
        elif comb == second_max_comb:
            color = second_color
        else:
            color = '#bdc3c7'
        
        fig.add_trace(go.Box(
            y=y_values, 
            name=comb,
            marker_color=color,
            line_color=color,
            hoverinfo='name',
            showlegend=False
        ))

    # --- Custom legend entries ---
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(size=12, color=best_color),
        name="Best model",
        showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(size=12, color=second_color),
        name="Second-best model",
        showlegend=True
    ))
    
    # Vertical lines where max_depth changes
    shapes = []
    for i in range(1, len(max_depths_ordered)):
        if max_depths_ordered[i] != max_depths_ordered[i-1]:
            shapes.append(dict(
                type='line',
                x0=i - 0.5,
                x1=i - 0.5,
                y0=0,
                y1=1,
                yref='paper',
                line=dict(color='gray', width=1, dash='dash')
            ))
    
    fig.update_layout(
        title='XGBoost Parameter Search Results',
        xaxis_title='Parameter Combinations',
        yaxis_title='Test Accuracy',
        xaxis_tickangle=-45,
        shapes=shapes,
        hoverlabel=dict(namelength=-1),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    div_html = fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={'responsive': True}
    )

    with open(output_path, 'w') as f:
        f.write(div_html)

    print(f"Parameter search plot saved: {output_path}")

def plot_class_distribution(unique, counts, output_path):
    """Create improved class distribution plot and save to file"""
    # Sort by counts (descending order)
    sorted_indices = np.argsort(counts)[::-1]
    sorted_unique = [unique[i] for i in sorted_indices]
    sorted_counts = counts[sorted_indices]
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create horizontal bar chart with gradient colors
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sorted_unique)))
    bars = ax.barh(sorted_unique, sorted_counts, color=colors, alpha=0.8, 
                   edgecolor='white', linewidth=1.5)
    
    # Styling to match theme
    ax.set_xlabel('Number of Samples', fontsize=14, fontweight='bold', color='#2c3e50')
    ax.set_ylabel('Cell Types', fontsize=14, fontweight='bold', color='#2c3e50')
    ax.set_title('Class Distribution in Training Data', fontsize=16, fontweight='bold', 
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
    ax.tick_params(axis='both', which='major', labelsize=12, colors='#2c3e50')
    
    # Background color
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Class distribution plot saved: {output_path}")

def plot_train_vs_test(df, top_models, output_path="xgbWinners_train_vs_test.png"):
    """Plot training vs test accuracy to identify overfitting"""
    # Calculate means per combination
    summary = df.groupby(['max_depth', 'eta']).agg({
        'Training': 'mean',
        'testf': 'mean'
    }).reset_index()
    
    # Identify top model combinations
    top_combos = set(zip(top_models['max_depth'], top_models['eta']))
    summary['is_top'] = summary.apply(
        lambda row: (row['max_depth'], row['eta']) in top_combos, axis=1
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all models
    ax.scatter(summary[~summary['is_top']]['Training'], 
               summary[~summary['is_top']]['testf'],
               alpha=0.6, s=100, c='#bdc3c7', label='Other Models')
    
    # Highlight top models
    ax.scatter(summary[summary['is_top']]['Training'], 
               summary[summary['is_top']]['testf'],
               alpha=0.9, s=200, c='#e74c3c', 
               edgecolors='black', linewidths=2, label='Top Models')
    
    # Add diagonal line (perfect agreement)
    min_val = min(summary['Training'].min(), summary['testf'].min())
    max_val = max(summary['Training'].max(), summary['testf'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'k--', alpha=0.5, linewidth=2, label='Perfect Agreement')
    
    # Add annotations for top models
    for _, row in summary[summary['is_top']].iterrows():
        ax.annotate(f"depth={row['max_depth']}\neta={row['eta']}", 
                   (row['Training'], row['testf']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, bbox=dict(boxstyle='round,pad=0.5', 
                                        fc='yellow', alpha=0.7))
    
    ax.set_xlabel('Training Accuracy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Training vs Test Accuracy\n(Points below diagonal = overfitting)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Format as percentages
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y * 100)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y * 100)))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Train vs test plot saved: {output_path}")


def plot_cv_stability(df, top_models, output_path="xgbWinners_cv_stability.png"):
    """Plot test accuracy across CV folds to assess stability"""
    # Filter to top model combinations
    top_combos = set(zip(top_models['max_depth'], top_models['eta']))
    df['combination'] = df.apply(
        lambda row: f"depth={row['max_depth']}, eta={row['eta']}", axis=1
    )
    df['is_top'] = df.apply(
        lambda row: (row['max_depth'], row['eta']) in top_combos, axis=1
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot CV folds for top models
    top_data = df[df['is_top']]
    
    for combo in top_data['combination'].unique():
        combo_data = top_data[top_data['combination'] == combo]
        # Sort by CV fold before plotting
        combo_data = combo_data.sort_values('cv')
        ax.plot(combo_data['cv'], combo_data['testf'], 
               marker='o', markersize=8, linewidth=2, label=combo)
    
    ax.set_xlabel('CV Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Validation Stability for Top Models', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as percentages
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y * 100)))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"CV stability plot saved: {output_path}")

def make_a_new_model(toTrainDF, classColumn, cpu_jobs, 
                     model_performance_table):
    """Train XGBoost models and save outputs as separate files"""  

    results = {
        'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Filter classes based on threshold
    class_counts = toTrainDF[classColumn].value_counts()
    print(f"Original class counts: {dict(class_counts)}")

    label_counts = pd.DataFrame({
    'Cell Type': class_counts.index,
    'Number of annotations': class_counts.values
    })

    classes_summary_path = "xgbWinners_classes_summary.csv"
    label_counts.to_csv(classes_summary_path, index=False)
    print(f"Classes summary table saved: {classes_summary_path}")
    results['classes_summary_path'] = classes_summary_path


    # Prepare features
    X = toTrainDF.select_dtypes(include=[np.number])
    X = X.loc[:, ~X.columns.duplicated()]

    le = preprocessing.LabelEncoder()
    y_Encode = le.fit_transform(toTrainDF[classColumn])
    unique, counts = np.unique(y_Encode, return_counts=True)

    encoder_path = "classes.npy"
    np.save(encoder_path, le.classes_)
    print(f"Label encoder saved: {encoder_path}")

    # Create class distribution plot
    class_dist_path = "xgbWinners_class_distribution.png"
    plot_class_distribution(unique, counts, class_dist_path)
    results['class_distribution_plot_path'] = class_dist_path
    
    num_round = 200
    xgboostParams = pd.read_csv(model_performance_table)

    #Create summary table
    xgboostParams['Training'] = xgboostParams['Training'].str.rstrip('%').astype(float) / 100
    summary_table = xgboostParams.groupby(['max_depth', 'eta']).agg(
        cv=('cv', lambda x: len(x.unique())),
        Training_mean=('Training', 'mean'),
        Training_std=('Training', 'std'),
        Test_mean=('testf', 'mean'),
        Test_std=('testf', 'std')
    ).reset_index()

    if len(summary_table) < 2:
        raise RuntimeError(
            f"Parameter search did not yield at least 2 unique parameter sets (found {len(summary_table)}). "
            "Multiple models cannot be compared. Please check your parameter search grid or input data."
        )

    param_table_path = "xgbWinners_parameter_summary.csv"
    summary_table.to_csv(param_table_path, index=False)
    print(f"Parameter summary table saved: {param_table_path}")
    results['parameter_summary_csv_path'] = param_table_path

    # Train and save top 2 models
    top_models = summary_table.sort_values('Test_mean', ascending=False).drop_duplicates(subset=['max_depth', 'eta']).head(2)
    print("Top Models:", top_models)

    for fname, (_, row) in zip(["XGBoost_Model_First.pkl", "XGBoost_Model_Second.pkl"], top_models.iterrows()):
        param = {
            'max_depth': int(row['max_depth']),
            'eta': row['eta'],
            'objective': 'multi:softprob',
            'n_jobs': cpu_jobs,
            'num_class': len(unique),
            'eval_metric': 'mlogloss'
        }
        
        #Creating dmatrix and saving the feature names/order
        dtrainAll = xgb.DMatrix(X, label=y_Encode, feature_names=X.columns.tolist())
        #Train the model
        bst = xgb.train(param, dtrainAll, num_round)
        #Save
        pickle.dump(bst, open(fname, "wb"))
        print(f"Model saved: {fname}")

    #Plot param search 
    param_plot_path = "xgbWinners_parameter_search.html"
    plot_parameter_search(xgboostParams, top_models, param_plot_path)
    results['parameter_search_plot_path'] = param_plot_path

    # Experimental plots
    plot_train_vs_test(xgboostParams, top_models)
    plot_cv_stability(xgboostParams, top_models)

    return results

def main():
    parser = argparse.ArgumentParser(description="Train and select XGBoost models based on parameter search results.")
    parser.add_argument('--classColumn', required=True, help='Name of the classified column')
    parser.add_argument('--cpu_jobs', type=int, default=16, help='Number of CPU jobs to use')
    parser.add_argument('--model_performance_table', required=True, help='CSV with model performance')
    parser.add_argument('--trainingDataframe', required=True, help='Path to training dataframe pickle')
    args = parser.parse_args()

    focusData = pd.read_pickle(args.trainingDataframe)

    training_results = make_a_new_model(
        focusData,
        args.classColumn,
        args.cpu_jobs,
        args.model_performance_table
    )

    # Save comprehensive results as JSON
    json_path = "xgbWinners_results.json"
    with open(json_path, 'w') as f:
        json.dump(training_results, f, indent=2)


if __name__ == "__main__":
    main()