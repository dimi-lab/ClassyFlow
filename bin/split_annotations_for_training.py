#!/usr/bin/env python3

import os, sys, csv, time
import argparse
import pandas as pd
import json

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


### Static Variables: File Formatting
batchColumn = 'Batch'


def stratified_split(df, stratify_cols, holdout_frac, min_count):    
    if df.empty:
        return pd.Series(dtype='object', name='split')
    if not (0 < holdout_frac < 1):
        raise ValueError("holdout_frac must be between 0 and 1")
    if min_count < 1:
        raise ValueError("min_count must be >= 1")
    
    missing_cols = [col for col in stratify_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
    
    group_sizes = df.groupby(stratify_cols).size()
    valid_groups = group_sizes[group_sizes >= min_count].index
    
    if len(stratify_cols) == 1:
        group_keys = df[stratify_cols[0]]
    else:
        group_keys = df[stratify_cols].apply(tuple, axis=1)
    
    mask = group_keys.isin(valid_groups)
    valid_indices = df.index[mask]
    
    def sample_group(group_df):
        n_holdout = int(len(group_df) * holdout_frac)
        if n_holdout == 0:
            return pd.Series(False, index=group_df.index)
        sampled_indices = group_df.sample(n=n_holdout, random_state=42).index
        holdout_series = pd.Series(False, index=group_df.index)
        holdout_series.loc[sampled_indices] = True
        return holdout_series
    
    holdout_mask = (
        df.loc[valid_indices]
        .groupby(group_keys[valid_indices])
        .apply(sample_group)
    )
    
    if isinstance(holdout_mask.index, pd.MultiIndex):
        holdout_indices = holdout_mask[holdout_mask].index.get_level_values(-1)
    else:
        holdout_indices = holdout_mask[holdout_mask].index
    
    split_indicator = pd.Series('Not Used', index=df.index, name='split')
    split_indicator.loc[valid_indices] = 'train'
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
    del dataframes
    del df
    
    #Gather some metrics before any filtering
    results['total_num_cells'] = len(merged_df)
    results['total_num_batches'] = merged_df[batchColumn].nunique()
    results['total_num_rois'] = merged_df["Image"].nunique()

    # Clean data and remove unlabelled rows
    merged_df[classColumn] = merged_df[classColumn].str.strip()
    merged_df = merged_df.dropna(subset=[classColumn])

    # Remove empty strings and negative classes
    merged_df = merged_df.loc[~merged_df[classColumn].isin(cellTypeNegative)]
    merged_df = merged_df.reset_index(drop=True)  # Clean reset
    results['total_num_annotated_cells'] = len(merged_df)
    results['total_num_cell_types'] = merged_df[classColumn].nunique()

    # Create split indicator
    split_indicator = stratified_split(
        merged_df, 
        [batchColumn, classColumn], 
        holdout_frac=holdoutFraction, 
        min_count=minimumHoldoutThreshold
    )
    
    # Add split column to dataframe
    merged_df['split'] = split_indicator

    ct = merged_df[classColumn].value_counts().rename('Label Count')
    pt = merged_df[classColumn].value_counts(normalize=True).mul(100).round(2).rename('Label Percent').astype(str) + '%'
    grouped = (merged_df.groupby([classColumn, 'split']).size().unstack(fill_value=0).rename_axis(None, axis=1))
    grouped.rename(columns={'train': 'Training Set Count', 'holdout': 'Holdout Set Count'}, inplace=True)
    
     # Combine everything
    freq_table = pd.concat([ct, pt, grouped], axis=1).reset_index()
    freq_table.rename(columns={classColumn: 'Label'}, inplace=True)

    # Ensure consistent column order
    for col in ['Training labels', 'Holdout labels', 'Not Used']:
        if col not in freq_table.columns:
            freq_table[col] = 0

    #Keep only columns we want 
    freq_table = freq_table[['Label', 'Label Count', 'Label Percent', 'Training Set Count', 'Holdout Set Count']]
    
    freq_table.to_csv('cell_count_table.csv', index=False)

    # Save cell types that made it to holdout
    kept_cell_types = merged_df[merged_df['split'] != 'Not Used'][classColumn].unique()
    with open("celltypes.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for cell_type in sorted(kept_cell_types):
            writer.writerow([cell_type])
    
    # Split data
    holdout_df = merged_df[merged_df['split'] == 'holdout'].copy()
    train_df = merged_df[merged_df['split'] == 'train'].copy()
    del merged_df

    # Remove split column before saving
    holdout_df = holdout_df.drop('split', axis=1)
    train_df = train_df.drop('split', axis=1)
    
    # Validation checks
    assert holdout_df[classColumn].nunique() == train_df[classColumn].nunique(), "Training and holdout data have different number of classes!!!"
    
    results['total_holdout'] = len(holdout_df)
    results['total_training'] = len(train_df)
    
    # Save final dataframes
    holdout_df.to_pickle('holdout_dataframe.pkl')
    train_df.to_pickle('training_dataframe.pkl')
    
    # Save results
    with open('training_split_report.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    
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