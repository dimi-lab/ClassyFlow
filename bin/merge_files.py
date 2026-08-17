#!/usr/bin/env python3

import sys, os
import argparse
import pandas as pd
import re
import random
import string
import numpy as np

def merge_tab_delimited_files(directory_path, excld, slide_by_prefix, folder_is_slide, input_extension, input_delimiter, batchID, target_size):
    # List all files in the directory
    files = [f for f in os.listdir(directory_path) if f.endswith(input_extension)]
    assert len(files) > 0, f"No files with extension '{input_extension}' found in directory: {directory_path}"

    def load_selected_columns(file_path, chunk_size=40000, excld_regex=None):
        selected_columns = []
        all_chunks = []
        sts = ["Min", "Max", "Median", "Mean", "Std.Dev.", "Variance"]
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
                # Convert stat columns to numeric during chunk loading
                for col in chunk.columns:
                    if any(s in col for s in sts):
                        chunk[col] = pd.to_numeric(chunk[col], errors='coerce').fillna(0)
                all_chunks.append(chunk)
        allLnData = pd.concat(all_chunks, axis=0, ignore_index=True)
        # Assert file is not empty
        assert allLnData.shape[0] > 0, f"Input file '{file_path}' is empty after loading."
        # Assert no duplicate columns
        assert allLnData.columns.duplicated().sum() == 0, f"Duplicate columns found in file '{file_path}': {allLnData.columns[allLnData.columns.duplicated()].tolist()}"
        # Assert key columns present
        for key_col in ['Image']:
            assert key_col in allLnData.columns, f"Required column '{key_col}' missing in file '{file_path}'"
        return allLnData

    def get_chunk_size(file_path):
        size_bytes = os.path.getsize(file_path)
        if size_bytes < 10 * 1024 * 1024:  # less than 10MB
            return 5000
        elif size_bytes < 1024 * 1024 * 1024:  # less than 1GB
            return 40000
        else:
            return 100000

    # Read headers first to determine reference column order
    header_sets = {}
    for file in files:
        file_path = os.path.join(directory_path, file)
        try:
            header = pd.read_csv(file_path, nrows=0, sep=input_delimiter, dtype=str).columns.tolist()
            header_sets[file] = header
        except Exception as e:
            print(f"[ERROR] Could not read header from {file_path}: {e}")
            sys.exit(1)
    ref_file = files[0]
    ref_header = header_sets[ref_file]

    dataframes = []
    for file in files:
        file_path = os.path.join(directory_path, file)
        excld_regex = excld if excld != '' else None
        chunk_size = get_chunk_size(file_path)
        df = load_selected_columns(file_path, chunk_size=chunk_size, excld_regex=excld_regex)
        # Assert consistent data types for each column
        if len(dataframes) > 0:
            prev_df = dataframes[0]
            for col in df.columns:
                if col in prev_df.columns:
                    if df[col].dtype != prev_df[col].dtype:
                        # If both are numeric, cast both to float and warn
                        if (pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(prev_df[col])):
                            print(f"[WARNING] Column '{col}' has inconsistent numeric dtypes between files: {df[col].dtype} vs {prev_df[col].dtype}. Casting both to float.")
                            df[col] = df[col].astype(float)
                            prev_df[col] = prev_df[col].astype(float)
                        else:
                            raise AssertionError(f"Column '{col}' has inconsistent dtype between files: {df[col].dtype} vs {prev_df[col].dtype}")
        # Reorder columns if possible
        if list(df.columns) != ref_header and len(df.columns) == len(ref_header):
            try:
                df = df.reindex(columns=ref_header)
                print(f"[INFO] Reordered columns in '{file}' to match '{ref_file}'.")
            except Exception as e:
                print(f"[WARNING] Could not reorder columns in '{file}': {e}")
        if slide_by_prefix:
            slide_series = df['Image'].str.split('_').str[0]
        elif folder_is_slide:
            slide_series = pd.Series(directory_path, index=df.index)
        else:
            slide_series = pd.Series(file, index=df.index)

        # Build all new/modified columns at once via pd.concat to avoid
        # repeated frame.insert calls that cause DataFrame fragmentation.
        if folder_is_slide:
            image_series = directory_path + '-' + df['Image']
            df = pd.concat(
                [df.drop(columns=['Image']),
                 pd.DataFrame({'Image': image_series, 'Slide': slide_series}, index=df.index)],
                axis=1
            )
        else:
            df = pd.concat(
                [df, pd.DataFrame({'Slide': slide_series}, index=df.index)],
                axis=1
            )
            df['Image'] = directory_path+'-'+df['Image']
        df = df.copy()

        dataframes.append(df)

    # Concatenate all DataFrames
    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df = merged_df.reset_index(drop=True)

    #Adding batchID to track throughout the pipeline
    merged_df["original_batchID"] = batchID
    
    # Assert merged DataFrame is not empty
    assert merged_df.shape[0] > 0, f"Merged Input Files result in EMPTY data table: {directory_path}"
    # Assert no duplicate columns in merged DataFrame
    assert merged_df.columns.duplicated().sum() == 0, f"Duplicate columns found in merged DataFrame: {merged_df.columns[merged_df.columns.duplicated()].tolist()}"
    # Warn if merged DataFrame has fewer columns than expected (may be due to excludingString)
    expected_cols = len(ref_header)
    if merged_df.shape[1] < expected_cols:
        print(f"[WARNING] Merged DataFrame has fewer columns ({merged_df.shape[1]}) than expected ({expected_cols}). This may be due to the excludingString filter.")
    # Assert key columns present
    for key_col in ['Image', 'Slide']:
        assert key_col in merged_df.columns, f"Required column '{key_col}' missing in merged DataFrame"
    # Assert unique image origin mapping
    assert merged_df['Image'].notna().all(), "Some rows in merged DataFrame have missing 'Image' values"

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
            missing = sorted(set(ref_header) - set(header))
            extra = sorted(set(header) - set(ref_header))
            if not missing and not extra:
                print(f"[WARNING] File '{fname}' columns are out of order compared to '{ref_file}'. Reordering columns.")
                # Fix column order in the corresponding dataframe
                for i, file in enumerate(files):
                    if file == fname:
                        dataframes[i] = dataframes[i].reindex(columns=ref_header)
                        break
            else:
                mismatch = True
                print(f"[ERROR] File '{fname}' has different columns than '{ref_file}'.")
                print(f"  Missing = {missing}  & Extra = {extra}")
                if missing:
                    if len(missing) > 5:
                        print(f" '{fname}' Missing columns (showing first 5 of {len(missing)}): {missing[:5]}")
                    else:
                        print(f" '{fname}' Missing columns: {missing}")
                if extra:
                    print(f" '{fname}' Extra columns: {extra}")
    if mismatch:
        sys.exit("[ERROR] Not all files have identical columns. Please fix the input files.")

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

