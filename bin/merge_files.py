#!/usr/bin/env python3

import sys, os
import argparse
import pandas as pd

def merge_tab_delimited_files(directory_path, excld, slide_by_prefix, folder_is_slide, input_extension, input_delimiter, batchID):
    # List all files in the directory
    files = [f for f in os.listdir(directory_path) if f.endswith(input_extension)]

    # Read and concatenate all tab-delimited files into a single DataFrame
    dataframes = []
    for file in files:
        file_path = os.path.join(directory_path, file)
        df = pd.read_csv(file_path, sep=input_delimiter, low_memory=False)
        #print(file_path) - Used to debug, misformatted files.
        #print(df.columns.tolist())
        if excld != '':
            df = df[df.columns.drop(list(df.filter(regex='('+excld+')')))]
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
    # merged_df = merged_df.sample(n=5000)  # remove this after testing
    merged_df = merged_df.reset_index()

    ## Throw Error if Quant Files are empty.
    if merged_df.shape[0] == 0:
        sys.exit("Merged Input Files result in EMPTY data table: {}".format(directory_path))

    # Save the merged DataFrame as a pickle file
    merged_df.to_pickle(f'merged_dataframe_{batchID}.pkl')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge tab-delimited files in a directory.")
    parser.add_argument('directory_path', help='Path to the directory containing files to merge')
    parser.add_argument('excludingString', help='String to exclude columns by regex (can be empty string)')
    parser.add_argument('--slide_by_prefix', action='store_true', help='Set this flag if slide is determined by prefix')
    parser.add_argument('--folder_is_slide', action='store_true', help='Set this flag if folder is the slide')
    parser.add_argument('--input_extension', default='.txt', help='File extension to look for (default: .txt)')
    parser.add_argument('--input_delimiter', default='\t', help='Delimiter for input files (default: tab)')
    parser.add_argument('--batchID', default='batch', help='Batch ID for output pickle file name')

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
        args.batchID
    )

