#!/usr/bin/env python3

import os, sys, csv, time
import argparse
import pandas as pd
import json

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


### Static Variables: File Formatting
batchColumn = 'Batch'

def stratified_split(df, stratify_cols, holdout_frac=0.01, min_count=3):
    """
    Create stratified train/holdout split indicator.
    Returns a Series aligned with the original dataframe index.
    """
    split_indicator = pd.Series('train', index=df.index, name='split')
    grouped = df.groupby(stratify_cols)
    
    for name, group in grouped:
        if len(group) >= min_count:
            holdout_indices = group.sample(
                frac=holdout_frac, 
                random_state=42
            ).index
            split_indicator.loc[holdout_indices] = 'holdout'
    
    return split_indicator


def gather_annotations(pickle_files, classColumn, holdoutFraction, cellTypeNegative, minimumHoldoutThreshold):
    dataframes = []
    results = {
        'holdout_fraction': holdoutFraction,
        'negative_classes': ", ".join(cellTypeNegative),
        'min_holdout_thresh': minimumHoldoutThreshold
    }

    # Load and combine data
    for file in pickle_files:
        print(f"Getting...{file}")
        if file.endswith('.pkl'):
            df = pd.read_pickle(file)
            dataframe_name = os.path.basename(file).replace('.pkl','').replace('merged_dataframe_','')
        else:
            df = pd.read_csv(file, sep='\t', low_memory=False)
            dataframe_name = os.path.basename(file).replace('.tsv','').replace('boxcox_transformed_','')
        df[batchColumn] = dataframe_name
        dataframes.append(df)
    
    merged_df = pd.concat(dataframes, ignore_index=True)
    results['total_cells'] = merged_df.shape[0]
    
    # Clean data
    merged_df[classColumn] = merged_df[classColumn].str.strip()
    merged_df = merged_df.dropna(subset=[classColumn])
    # Remove empty strings and negative classes
    merged_df = merged_df.loc[~merged_df[classColumn].isin(cellTypeNegative)]
    merged_df = merged_df.reset_index(drop=True)  # Clean reset
    results['total_annotated_cells'] = merged_df.shape[0]

    # Create split indicator
    split_indicator = stratified_split(
        merged_df, 
        [batchColumn, classColumn], 
        holdout_frac=holdoutFraction, 
        min_count=minimumHoldoutThreshold
    )
    
    # Add split column to dataframe
    merged_df['split'] = split_indicator
    
    # Split data
    holdout_df = merged_df[merged_df['split'] == 'holdout'].copy()
    train_df = merged_df[merged_df['split'] == 'train'].copy()
    
    # Remove split column before saving
    holdout_df = holdout_df.drop('split', axis=1)
    train_df = train_df.drop('split', axis=1)
    
    print(f"holdout_df {holdout_df.shape}")
    print(f"train_df {train_df.shape}")
    
    # Validation checks
    assert len(holdout_df) + len(train_df) == len(merged_df), "Split doesn't account for all data"
    
    # Only keep cell types that appear in holdout
    holdout_cell_types = set(holdout_df[classColumn].unique())
    train_df = train_df[train_df[classColumn].isin(holdout_cell_types)]
    
    results['total_holdout'] = len(holdout_df)
    results['total_training'] = len(train_df)
    
    # Create frequency table
    ct = merged_df[classColumn].value_counts()
    pt = merged_df[classColumn].value_counts(normalize=True).mul(100).round(2).astype(str) + '%'
    hd = holdout_df[classColumn].value_counts()
    
    freq_table = pd.concat([ct, pt, hd], axis=1, keys=['counts', '%', 'holdout']).fillna("Not Used")
    freq_table = freq_table.reset_index().rename(columns={'index': classColumn})
    
    # Export tables and lists
    freq_table.to_html('cell_count_table.html', index=False)
    freq_table.to_csv('cell_count_table.csv', index=False)
    
    # Save cell types that made it to holdout
    kept_cell_types = holdout_df[classColumn].unique()
    with open("celltypes.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for cell_type in sorted(kept_cell_types):
            writer.writerow([cell_type])
    
    # Save final dataframes
    holdout_df.to_pickle('holdout_dataframe.pkl')
    train_df.to_pickle('training_dataframe.pkl')
    
    # Save results
    with open('training_split_report.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return train_df, holdout_df, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split annotations for training and holdout sets.")
    parser.add_argument('--classColumn', required=True, help='Name of the classified column')
    parser.add_argument('--holdoutFraction', type=float, required=True, help='Fraction of data to hold out')
    parser.add_argument('--cellTypeNegative', required=True, help='Comma-separated list of negative class values')
    parser.add_argument('--minimunHoldoutThreshold', type=int, required=True, help='Minimum label count for holdout')
    parser.add_argument('--pickle_files', required=True, help='Space-separated list of pickle files')
    args = parser.parse_args()

    classColumn = args.classColumn
    holdoutFraction = args.holdoutFraction
    cellTypeNegative = args.cellTypeNegative.split(",")
    cellTypeNegative.append("")
    minimunHoldoutThreshold = args.minimunHoldoutThreshold
    pickle_files = args.pickle_files.split(' ')

    gather_annotations(pickle_files, classColumn, holdoutFraction, cellTypeNegative, minimunHoldoutThreshold)