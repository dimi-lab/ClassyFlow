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

def plot_parameter_search(df, output_path):
    """Create parameter search boxplot and save to file"""
    df['combination'] = df.apply(lambda row: f"max_depth={row['max_depth']}, eta={row['eta']}", axis=1)
    mean_values = df.groupby('combination')['testf'].mean()
    max_comb = mean_values.idxmax()
    second_max_comb = mean_values.nlargest(2).idxmin()
    unique_combs = df['combination'].unique().tolist()
    color_palette = ['grey'] * len(unique_combs)
    color_palette[unique_combs.index(max_comb)] = 'red'
    color_palette[unique_combs.index(second_max_comb)] = 'orange'

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x='combination', y='testf', hue='combination', data=df,
        palette=color_palette, legend=False, flierprops={'markerfacecolor':'grey'}
    )
    plt.ylim(df['testf'].min() - 0.01, df['testf'].max() + 0.01)
    yticks = plt.gca().get_yticks()
    plt.gca().set_yticks(yticks)
    plt.gca().set_yticklabels(['{:.0f}%'.format(y * 100) for y in yticks])
    plt.xlabel('Combinations of Parameters')
    plt.ylabel('Test Accuracy')
    plt.title('Boxplot of XGB Training')
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Parameter search plot saved: {output_path}")

def plot_class_distribution(unique, counts, output_path):
    """Create class distribution plot and save to file"""
    plt.figure(figsize=(8, 6))
    plt.barh(unique, counts, color='steelblue', alpha=0.7)
    plt.xlabel('Number of Samples')
    plt.ylabel('Class Labels')
    plt.title('Class Distribution in Training Data')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, count in enumerate(counts):
        plt.text(count + max(counts)*0.01, i, f'{count}', 
                ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Class distribution plot saved: {output_path}")

def make_a_new_model(toTrainDF, classColumn, cpu_jobs, mim_class_label_threshold, 
                     model_performance_table):
    """Train XGBoost models and save outputs as separate files"""  

    results = {
        'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'class_column': classColumn,
        'cpu_jobs': cpu_jobs,
        'min_class_threshold': mim_class_label_threshold
    }

    # Filter classes based on threshold
    class_counts = toTrainDF[classColumn].value_counts()
    print(f"Original class counts: {dict(class_counts)}")

    classes_to_keep = class_counts[class_counts > mim_class_label_threshold].index
    original_shape = toTrainDF.shape
    toTrainDF = toTrainDF[toTrainDF[classColumn].isin(classes_to_keep)]
    filtered_shape = toTrainDF.shape

    print(f"Data shape: {original_shape} -> {filtered_shape} (filtered by threshold)")

    # Store class filtering info
    results['original_data_shape'] = list(original_shape)
    results['filtered_data_shape'] = list(filtered_shape)
    results['classes_removed'] = {k: int(v) for k, v in class_counts[class_counts <= mim_class_label_threshold].to_dict().items()}
    results['classes_kept'] = {k: int(v) for k, v in class_counts[class_counts > mim_class_label_threshold].to_dict().items()}

    # Prepare features
    X = toTrainDF.select_dtypes(include=[np.number])
    X = X.loc[:, ~X.columns.duplicated()]

    results['feature_count'] = X.shape[1]
    results['feature_names'] = X.columns.tolist()

    le = preprocessing.LabelEncoder()
    y_Encode = le.fit_transform(toTrainDF[classColumn])
    unique, counts = np.unique(y_Encode, return_counts=True)

    encoder_path = "classes.npy"
    np.save(encoder_path, le.classes_)
    print(f"Label encoder saved: {encoder_path}")
    results['label_encoder_path'] = encoder_path
    results['class_mapping'] = {int(k): str(v) for k, v in zip(unique.tolist(), le.classes_.tolist())}

    # Create class distribution plot
    class_dist_path = "xgbWinners_class_distribution.png"
    plot_class_distribution(unique, counts, class_dist_path)
    results['class_distribution_plot_path'] = class_dist_path
    results['final_class_counts'] = {int(k): int(v) for k, v in zip(unique.tolist(), counts.tolist())}
    
    num_round = 200

    xgboostParams = pd.read_csv(model_performance_table)
    results['parameter_search_data'] = xgboostParams.to_dict('records')

    param_plot_path = "xgbWinners_parameter_search.png"
    plot_parameter_search(xgboostParams, param_plot_path)
    results['parameter_search_plot_path'] = param_plot_path

    #Create summary table
    xgboostParams['Training'] = xgboostParams['Training'].str.rstrip('%').astype(float) / 100
    summary_table = xgboostParams.groupby(['max_depth', 'eta']).agg(
        cv=('cv', lambda x: len(x.unique())),
        Training_mean=('Training', 'mean'),
        Training_std=('Training', 'std'),
        Test_mean=('testf', 'mean'),
        Test_std=('testf', 'std')
    ).reset_index()

    # --- Add this to handle parameter search being too small ---
    if len(summary_table) < 2:
        raise RuntimeError(
            f"Parameter search did not yield at least 2 unique parameter sets (found {len(summary_table)}). "
            "Multiple models cannot be compared. Please check your parameter search grid or input data."
        )
    # -----------------------------------------------------------

    param_table_path = "xgbWinners_parameter_summary.csv"
    summary_table.to_csv(param_table_path, index=False)
    print(f"Parameter summary table saved: {param_table_path}")
    results['parameter_summary_csv_path'] = param_table_path

    # Train and save top 2 models
    top_models = summary_table.sort_values('Test_mean', ascending=False).drop_duplicates(subset=['max_depth', 'eta']).head(2)
    print("Top Models:", top_models)

    model_info = []
    model_paths = []

    #for fname, (_, row) in zip(["XGBoost_Model_First.pkl", "XGBoost_Model_Second.pkl"], top_models.iterrows()):
    for i, (fname, (_, row)) in enumerate(zip(["XGBoost_Model_First.pkl", "XGBoost_Model_Second.pkl"], 
                                             top_models.iterrows())):
        param = {
            'max_depth': int(row['max_depth']),
            'eta': row['eta'],
            'objective': 'multi:softmax',
            'n_jobs': cpu_jobs,
            'num_class': len(unique),
            'eval_metric': 'mlogloss'
        }
        dtrainAll = xgb.DMatrix(X, label=y_Encode)
        bst = xgb.train(param, dtrainAll, num_round)
        pickle.dump(bst, open(fname, "wb"))
        print(f"Model saved: {fname}")

        # Store model info
        model_info.append({
            'rank': i + 1,
            'filename': fname,
            'max_depth': int(row['max_depth']),
            'eta': float(row['eta']),
            'test_mean': float(row['Test_mean']),
            'test_std': float(row['Test_std']),
            'training_mean': float(row['Training_mean']),
            'training_std': float(row['Training_std']),
            'cv_folds': int(row['cv']),
            'num_rounds': num_round,
            'parameters': param
        })
        model_paths.append(fname)

    results['best_model'] = model_info[0] if model_info else None
    results['second_best_model'] = model_info[1] if len(model_info) > 1 else None

        # Training summary
    results['training_summary'] = {
        'total_samples': filtered_shape[0],
        'total_features': X.shape[1],
        'num_classes': len(unique),
        'training_rounds': num_round,
        'models_trained': len(model_info),
        'best_test_accuracy': float(top_models.iloc[0]['Test_mean']) if not top_models.empty else None
    }


def main():
    parser = argparse.ArgumentParser(description="Train and select XGBoost models based on parameter search results.")
    parser.add_argument('--classColumn', required=True, help='Name of the classified column')
    parser.add_argument('--cpu_jobs', type=int, default=16, help='Number of CPU jobs to use')
    parser.add_argument('--mim_class_label_threshold', type=int, required=True, help='Minimum label count')
    parser.add_argument('--model_performance_table', required=True, help='CSV with model performance')
    parser.add_argument('--trainingDataframe', required=True, help='Path to training dataframe pickle')
    parser.add_argument('--select_features_csv', required=True, help='Path to selected features CSV')
    args = parser.parse_args()

    myData = pd.read_pickle(args.trainingDataframe)
    with open(args.select_features_csv, 'r') as file:
        next(file)
        featureList = [line.strip() for line in file if line.strip()]

    if 'level_0' in featureList:
        featureList.remove('level_0')
    featureList.append(args.classColumn)
    focusData = myData[featureList]

    training_results = make_a_new_model(
        focusData,
        args.classColumn,
        args.cpu_jobs,
        args.mim_class_label_threshold,
        args.model_performance_table
    )

    # Save comprehensive results as JSON
    json_path = "xgbWinners_results.json"
    with open(json_path, 'w') as f:
        json.dump(training_results, f, indent=2)


if __name__ == "__main__":
    main()