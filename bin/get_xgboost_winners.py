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
    """Create parameter search boxplot and save to file"""
    df['combination'] = df.apply(lambda row: f"max_depth={row['max_depth']}, eta={row['eta']}", axis=1)

    # Sort combinations by max_depth first, then eta
    df_sorted = df.sort_values(['max_depth', 'eta'])
    ordered_combs = df_sorted['combination'].unique().tolist()

    # Get top 2 combinations from provided top_models
    top_2_combs = [f"max_depth={int(row['max_depth'])}, eta={row['eta']}" for _, row in top_models.iterrows()]
    max_comb = top_2_combs[0]
    second_max_comb = top_2_combs[1] if len(top_2_combs) > 1 else None

    # Create better color palette
    color_palette = ['#bdc3c7'] * len(ordered_combs)  # Light gray for others
    color_palette[ordered_combs.index(max_comb)] = '#e74c3c'  # Red for best
    if second_max_comb:
        color_palette[ordered_combs.index(second_max_comb)] = '#f39c12'  # Orange for second

    fig, ax = plt.subplots(figsize=(18, 10))

    # Create boxplot with improved styling and specified order
    box_plot = sns.boxplot(
        x='combination', y='testf', hue='combination', data=df,
        order=ordered_combs,
        palette=color_palette, legend=False, 
        flierprops={'markerfacecolor':'#95a5a6', 'markeredgecolor':'#7f8c8d', 'markersize': 6},
        boxprops={'alpha': 0.8, 'linewidth': 1.5},
        whiskerprops={'linewidth': 1.5, 'color': '#2c3e50'},
        capprops={'linewidth': 1.5, 'color': '#2c3e50'},
        medianprops={'linewidth': 2, 'color': '#2c3e50'},
        ax=ax
    )

    # Find positions where max_depth changes and draw vertical lines
    max_depths = df_sorted.groupby('combination')['max_depth'].first()
    max_depths_ordered = [max_depths[comb] for comb in ordered_combs]

    for i in range(1, len(max_depths_ordered)):
        if max_depths_ordered[i] != max_depths_ordered[i-1]:
            ax.axvline(x=i-0.5, color='#34495e', linestyle='--', linewidth=2, alpha=0.6)

    # Styling
    ax.set_ylim(df['testf'].min() - 0.01, df['testf'].max() + 0.01)
    yticks = ax.get_yticks()
    ax.set_yticklabels(['{:.0f}%'.format(y * 100) for y in yticks])

    ax.set_xlabel('Parameter Combinations', fontsize=14, fontweight='bold', color='#2c3e50')
    ax.set_ylabel('Test Accuracy', fontsize=14, fontweight='bold', color='#2c3e50')
    ax.set_title('XGBoost Parameter Search Results', fontsize=16, fontweight='bold', 
                    color='#2c3e50', pad=20)

    # Grid and axis styling
    ax.grid(True, alpha=0.3, axis='y', linestyle='-', color='#bdc3c7')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')

    # Rotate x-axis labels
    plt.xticks(rotation=35, ha='right', fontsize=11, color='#2c3e50')
    ax.tick_params(axis='y', which='major', labelsize=12, colors='#2c3e50')

    # Background styling
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')

    # Add legend for color coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', alpha=0.8, label='Best Model'),
    ]
    if second_max_comb:
        legend_elements.append(
            Patch(facecolor='#f39c12', alpha=0.8, label='Second Best Model')
        )
    legend_elements.append(
        Patch(facecolor='#bdc3c7', alpha=0.8, label='Other Models')
    )
    ax.legend(handles=legend_elements, loc='upper left', frameon=True, 
                fancybox=True, shadow=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
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
    param_plot_path = "xgbWinners_parameter_search.png"
    plot_parameter_search(xgboostParams, top_models, param_plot_path)
    results['parameter_search_plot_path'] = param_plot_path

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