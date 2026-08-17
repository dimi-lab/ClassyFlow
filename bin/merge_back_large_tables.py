#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import gc

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Merge large tab-delimited tables efficiently.")
    parser.add_argument('--input_files', nargs='+', required=True, help='List of input TSV files to merge')
    parser.add_argument('--output_file', required=True, help='Output TSV file name')
    return parser.parse_args()

def validate_originalBatchID(input_files):
    batch_ids = {}

    for file in input_files:
        df = pd.read_csv(file, sep='\t')
        if 'original_batchID' not in df.columns:
            del df
            gc.collect()
            raise ValueError(f"Column 'original_batchID' not found in {file}")
        unique_ids = df['original_batchID'].unique()
        if len(unique_ids) != 1:
            del df
            gc.collect()
            raise ValueError(f"File {file} contains {len(unique_ids)} unique batch IDs: {unique_ids.tolist()}."
                             f"Expected exatly 1 batch ID")
        batch_ids[file] = unique_ids[0]
        del df
        gc.collect()

    #Check that all files being merged have the exact same original batchID
    unique_batch_ids = set(batch_ids.values())
    if len(unique_batch_ids) != 1:
        error_msg = "Input files have different original_BatchID values:\n"
        for file, batch_id in batch_ids.items():
            error_msg += f"   {os.path.basename(file)}: {batch_id}\n"
        raise ValueError(error_msg)
    
    print(f"All files have passed batchID validation and are ok to merge!")

def main():
    args = parse_args()
    input_files = args.input_files
    output_file = args.output_file

    if not input_files:
        raise ValueError("No input files provided.")

    # Validate that all files have the same original Batch ID (using pandas, but only for validation)
    validate_originalBatchID(input_files)

    # Write header from the first file
    with open(input_files[0], 'r') as fin:
        header = fin.readline()

    with open(output_file, 'w') as fout:
        fout.write(header)
        # Append all files, skipping header for each, line by line (low memory)
        for file in input_files:
            with open(file, 'r') as fin:
                next(fin)  # skip header
                for line in fin:
                    fout.write(line)

if __name__ == "__main__":
    main()