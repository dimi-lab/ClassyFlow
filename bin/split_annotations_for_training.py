#!/usr/bin/env python3

import os, sys, csv, time
import argparse
import pandas as pd
import json
import gc
import warnings
import concurrent.futures
warnings.filterwarnings("ignore", category=DeprecationWarning)

batchColumn = 'Batch'
excludeErronousNanString = "nan"

def stratified_split(df, stratify_cols, holdout_frac, min_count, min_count_col):    
    if df.empty:
        return pd.Series(dtype='object', name='split')
    if not (0 < holdout_frac < 1):
        raise ValueError("holdout_frac must be between 0 and 1")
    if min_count < 1:
        raise ValueError("min_count must be >= 1")
    
    missing_cols = [col for col in stratify_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
    
    if min_count_col not in df.columns:
        raise ValueError(f"min_count_col '{min_count_col}' not found in DataFrame")
    
    # Filter based on overall counts in min_count_col
    class_counts = df[min_count_col].value_counts()
    valid_classes = class_counts[class_counts >= min_count].index
    
    print(f"Classes with >= {min_count} samples: {len(valid_classes)}/{len(class_counts)}")
    excluded_classes = class_counts[class_counts < min_count]
    if len(excluded_classes) > 0:
        print(f"Excluded classes (< {min_count} samples):")
        for cls, count in excluded_classes.items():
            print(f"  {cls}: {count} samples")
    
    # Keep only rows with valid classes
    valid_mask = df[min_count_col].isin(valid_classes)
    valid_indices = df.index[valid_mask]
    
    # Create grouping keys for stratification
    if len(stratify_cols) == 1:
        group_keys = df[stratify_cols[0]]
    else:
        group_keys = df[stratify_cols].apply(tuple, axis=1)
    
    def sample_group(group_df):
        n_holdout = int(len(group_df) * holdout_frac)
        if n_holdout == 0:
            # If group is too small for holdout, put all in train
            return pd.Series(False, index=group_df.index)
        sampled_indices = group_df.sample(n=n_holdout, random_state=42).index
        holdout_series = pd.Series(False, index=group_df.index)
        holdout_series.loc[sampled_indices] = True
        return holdout_series
    
    # Stratified sampling on valid data only
    holdout_mask = (
        df.loc[valid_indices]
        .groupby(group_keys[valid_indices], group_keys=False)
        .apply(sample_group, include_groups=False)
    )
    
    holdout_indices = holdout_mask[holdout_mask].index
    
    # Create split indicator
    split_indicator = pd.Series('Not Used', index=df.index, name='split')
    split_indicator.loc[valid_indices] = 'train'
    split_indicator.loc[holdout_indices] = 'holdout'
    
    return split_indicator

def process_file(file, classColumn, cellTypeNegative, excludeErronousNanString, batchColumn):
    print(f"Getting...{file}")
    if file.endswith('.pkl'):
        df = pd.read_pickle(file)
        dataframe_name = os.path.basename(file).replace('.pkl','').replace('merged_dataframe_','')
    else:
        df = pd.read_csv(file, sep='\t', low_memory=False)
        dataframe_name = os.path.basename(file).replace('.tsv','').replace('boxcox_transformed_','')
    df[batchColumn] = dataframe_name

    # Filter out negative/unwanted classes and drop NA in classColumn
    df[classColumn] = df[classColumn].astype(str).str.strip()
    df = df.dropna(subset=[classColumn])
    df = df.loc[~df[classColumn].isin(cellTypeNegative)]
    df = df[df[classColumn] != excludeErronousNanString]
    print(f"After filtering erroneous 'nan' strings: {df.shape[0]} rows")
    print(f"Value counts for '{classColumn}' after filtering:\n{df[classColumn].value_counts()}")
    return df

def gather_annotations(pickle_files, classColumn, holdoutFraction, cellTypeNegative, minimumHoldoutThreshold):
    dataframes = []
    results = {
        'holdout_fraction': holdoutFraction,
        'negative_classes': ", ".join(cellTypeNegative),
        'min_holdout_thresh': minimumHoldoutThreshold
    }

    # Parallel file processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_file, file, classColumn, cellTypeNegative, excludeErronousNanString, batchColumn
            )
            for file in pickle_files
        ]
        for future in concurrent.futures.as_completed(futures):
            df = future.result()
            dataframes.append(df)
            gc.collect()

    merged_df = pd.concat(dataframes, ignore_index=True)
    print(f"[MERGE CHECK] After merging: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
    del dataframes
    gc.collect()
    
    # Gather some metrics before any further filtering
    results['total_num_cells'] = len(merged_df)
    results['total_num_batches'] = merged_df[batchColumn].nunique()
    results['total_num_rois'] = merged_df["Image"].nunique()
    results['total_num_annotated_cells'] = len(merged_df)
    results['total_num_cell_types'] = merged_df[classColumn].nunique()

    # Create split indicator
    split_indicator = stratified_split(
        merged_df, 
        [batchColumn, classColumn], 
        holdout_frac=holdoutFraction, 
        min_count=minimumHoldoutThreshold,
        min_count_col=classColumn
    )
    
    # Add split column to dataframe
    merged_df['split'] = split_indicator
    merged_df = merged_df.copy()  # Defragments the DataFrame in memory

    # --- Add check here ---
    print(f"Unique '{classColumn}' values: {merged_df[classColumn].nunique()} = [{', '.join(merged_df[classColumn].unique())}]")
    print(f"Total rows: {merged_df.shape[0]}, Total columns: {merged_df.shape[1]}")
    # --- End check ---

    ct = merged_df[classColumn].value_counts().rename('Label Count')
    pt = merged_df[classColumn].value_counts(normalize=True).mul(100).round(2).rename('Label Percent').astype(str) + '%'
    grouped = (merged_df.groupby([classColumn, 'split']).size().unstack(fill_value=0).rename_axis(None, axis=1))
    grouped.rename(columns={'train': 'Training Set Count', 'holdout': 'Holdout Set Count'}, inplace=True)
    
    freq_table = pd.concat([ct, pt, grouped], axis=1).reset_index()
    freq_table.rename(columns={classColumn: 'Label'}, inplace=True)

    for col in ['Training labels', 'Holdout labels', 'Not Used']:
        if col not in freq_table.columns:
            freq_table[col] = 0

    freq_table = freq_table[['Label', 'Label Count', 'Label Percent', 'Training Set Count', 'Holdout Set Count']]
    freq_table.to_csv('cell_count_table.csv', index=False)

    # Create and save per-batch label count crosstab
    per_batch_label_count = pd.crosstab(merged_df[classColumn], merged_df[batchColumn])
    per_batch_label_count.to_csv('per_batch_label_count.csv')


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
    gc.collect()

    # Remove split column before saving
    holdout_df = holdout_df.drop('split', axis=1)
    train_df = train_df.drop('split', axis=1)
    
    if holdout_df[classColumn].nunique() != train_df[classColumn].nunique():
        print("\n[ERROR] Training and holdout data have different number of classes!!!")
        print(f"Unique classes in training set ({train_df[classColumn].nunique()}): {sorted(train_df[classColumn].unique())}")
        print(f"Unique classes in holdout set ({holdout_df[classColumn].nunique()}): {sorted(holdout_df[classColumn].unique())}")
        missing_in_train = set(holdout_df[classColumn].unique()) - set(train_df[classColumn].unique())
        missing_in_holdout = set(train_df[classColumn].unique()) - set(holdout_df[classColumn].unique())
        if missing_in_train:
            print(f"Classes in holdout but not in train: {sorted(missing_in_train)}")
        if missing_in_holdout:
            print(f"Classes in train but not in holdout: {sorted(missing_in_holdout)}")
        raise AssertionError("Training and holdout data have different number of classes!!!")
    
    results['total_holdout'] = len(holdout_df)
    results['total_training'] = len(train_df)
    
    holdout_df.to_pickle('holdout_dataframe.pkl')
    train_df.to_pickle('training_dataframe.pkl')
    
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
