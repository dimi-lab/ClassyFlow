#!/usr/bin/env python3

import sys, os
import argparse
import pandas as pd
import re
import random
import string

def merge_tab_delimited_files(directory_path, excld, slide_by_prefix, folder_is_slide, input_extension, input_delimiter, batchID, target_size):
    # List all files in the directory
    files = [f for f in os.listdir(directory_path) if f.endswith(input_extension)]

    def load_selected_columns(file_path, chunk_size=40000, excld_regex=None):
        selected_columns = []
        all_chunks = []
        # Read the first chunk to identify columns to exclude
        for first_chunk in pd.read_csv(file_path, chunksize=1, low_memory=False, sep=input_delimiter, dtype=str):
            columns = first_chunk.columns
            if excld_regex:
                exclude_regex = re.compile(excld_regex, re.IGNORECASE)
                selected_columns = [col for col in columns if not exclude_regex.search(col)]
            else:
                selected_columns = list(columns)
            break
        # Process chunks efficiently
        with pd.read_csv(file_path, usecols=selected_columns, chunksize=chunk_size, low_memory=False, sep=input_delimiter, dtype=str) as reader:
            for chunk in reader:
                all_chunks.append(chunk)
        allLnData = pd.concat(all_chunks, axis=0, ignore_index=True)
        return allLnData

    def get_chunk_size(file_path):
        size_bytes = os.path.getsize(file_path)
        if size_bytes < 10 * 1024 * 1024:  # less than 10MB
            return 5000
        elif size_bytes < 1024 * 1024 * 1024:  # less than 1GB
            return 40000
        else:
            return 100000

    dataframes = []
    for file in files:
        file_path = os.path.join(directory_path, file)
        excld_regex = excld if excld != '' else None
        chunk_size = get_chunk_size(file_path)
        df = load_selected_columns(file_path, chunk_size=chunk_size, excld_regex=excld_regex)
        if slide_by_prefix:
            df['Slide'] = [e.split('_')[0] for e in df['Image'].tolist() ]
        elif folder_is_slide:
            df['Slide'] = directory_path
        else:
            df['Slide'] = file

        if folder_is_slide:
            df['Image'] = directory_path+'-'+df['Image']
        dataframes.append(df)

    # Concatenate all DataFrames
    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df = merged_df.reset_index(drop=True)

    ## Throw Error if Quant Files are empty.
    if merged_df.shape[0] == 0:
        sys.exit("Merged Input Files result in EMPTY data table: {}".format(directory_path))

    # If target_size is set and merged_df is larger, split into multiple files (round robin)
    if target_size and merged_df.shape[0] > target_size:
        n_splits = (merged_df.shape[0] + target_size - 1) // target_size
        rand_suffixes = [''.join(random.choices(string.ascii_letters + string.digits, k=5)) for _ in range(n_splits)]
        out_paths = [f'merged_dataframe_{batchID}-{suffix}.pkl' for suffix in rand_suffixes]
        split_dfs = [[] for _ in range(n_splits)]

        # Assign each row to a split in round robin fashion
        for idx, row in merged_df.iterrows():
            split_idx = idx % n_splits
            split_dfs[split_idx].append(row)

        for i, rows in enumerate(split_dfs):
            split_df = pd.DataFrame(rows, columns=merged_df.columns)
            split_df.to_pickle(out_paths[i])
            print(f"[INFO] Saved split {i+1}/{n_splits} with {split_df.shape[0]} rows to {out_paths[i]}")
    else:
        merged_df.to_pickle(f'merged_dataframe_{batchID}-00000.pkl')
        print(f"[INFO] Saved merged dataframe to merged_dataframe_{batchID}-00000.pkl")

    # Print basename and number of columns for each file
    for file in files:
        file_path = os.path.join(directory_path, file)
        try:
            header = pd.read_csv(file_path, nrows=0, sep=input_delimiter, dtype=str).columns
            print(f"[INFO] {os.path.basename(file_path)} ({directory_path}): {len(header)} columns")
        except Exception as e:
            print(f"[WARNING] Could not read columns from {file_path}: {e}")

    # Check that all files have the same columns before loading data
    header_sets = {}
    for file in files:
        file_path = os.path.join(directory_path, file)
        try:
            header = pd.read_csv(file_path, nrows=0, sep=input_delimiter, dtype=str).columns.tolist()
            header_sets[file] = header
        except Exception as e:
            print(f"[ERROR] Could not read header from {file_path}: {e}")
            sys.exit(1)
    # Compare all header lists to the first file's header
    ref_file = files[0]
    ref_header = header_sets[ref_file]
    mismatch = False
    for fname, header in header_sets.items():
        if header != ref_header:
            mismatch = True
            missing = sorted(set(ref_header) - set(header))
            extra = sorted(set(header) - set(ref_header))
            print(f"[ERROR] File '{fname}' has different columns than '{ref_file}'.")
            if missing:
                print(f"  Missing columns: {missing}")
            if extra:
                print(f"  Extra columns: {extra}")
    if mismatch:
        sys.exit("[ERROR] Not all files have identical columns. Please fix the input files.")

    sts = ["Min", "Max", "Median", "Mean", "Std.Dev.", "Variance"]

    for col in merged_df.columns:
        if any(s in col for s in sts):
            # Check if column is numeric
            if not pd.api.types.is_numeric_dtype(merged_df[col]):
                print(f"[WARNING] Column '{col}' should be numeric but is {merged_df[col].dtype}. Attempting to convert.")
                merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
                n_nans = merged_df[col].isna().sum()
                if n_nans > 0:
                    print(f"[INFO] Filled {n_nans} NaN values in '{col}' with 0 after conversion.")
                    merged_df[col] = merged_df[col].fillna(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge tab-delimited files in a directory.")
    parser.add_argument('directory_path', help='Path to the directory containing files to merge')
    parser.add_argument('excludingString', help='String to exclude columns by regex (can be empty string)')
    parser.add_argument('--slide_by_prefix', action='store_true', help='Set this flag if slide is determined by prefix')
    parser.add_argument('--folder_is_slide', action='store_true', help='Set this flag if folder is the slide')
    parser.add_argument('--input_extension', default='.txt', help='File extension to look for (default: .txt)')
    parser.add_argument('--input_delimiter', default='\t', help='Delimiter for input files (default: tab)')
    parser.add_argument('--batchID', default='batch', help='Batch ID for output pickle file name')
    parser.add_argument('--target_size', type=int, default=1000000, help='Target number of rows per output file (default: 8000)')
    args = parser.parse_args()

    if args.input_delimiter == '\\t':
        args.input_delimiter = '\t'
    print(f"Using delimiter: >{repr(args.input_delimiter)}<")

    merge_tab_delimited_files(
        args.directory_path,
        args.excludingString,
        args.slide_by_prefix,
        args.folder_is_slide,
        args.input_extension,
        args.input_delimiter,
        args.batchID,
        args.target_size
    )

