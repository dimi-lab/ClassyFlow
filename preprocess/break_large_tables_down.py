import argparse
import pandas as pd
import re
import os

# Function to filter and load specific columns
def load_selected_columns(file_path, chunk_size=100000):
    """
    Load specific columns from a large file in chunks and combine them into a single DataFrame.
    Parameters:
    - file_path (str): Path to the large file.
    - chunk_size (int): Number of rows per chunk.

    Returns:
    - pd.DataFrame: Combined DataFrame with selected columns.
    """
    selected_columns = []  # Store the matching columns
    all_chunks = []        # Store the processed chunks
    
    # Read the first chunk to identify matching columns
    for first_chunk in pd.read_csv(file_path, chunksize=1, low_memory=False, sep='\t'):
        columns = first_chunk.columns
        regex = re.compile(r"(Centroid|Mean|_Pred)|^Image$|^Name$|^CellTypes$|^SiteLocation", re.IGNORECASE)
        selected_columns = [col for col in columns if regex.search(col)]
        break  # We only need to examine the columns once

    print(f"Selected columns: {selected_columns}")
 
    # Process chunks efficiently and write to a new file
    with pd.read_csv(file_path, usecols=selected_columns, chunksize=chunk_size, low_memory=True, sep='\t') as reader:
        for i, chunk in enumerate(reader):
            all_chunks.append(chunk)
            print(f"Processed chunk {i + 1}")
        
    # Combine all chunks into a single DataFrame
    allLnData = pd.concat(all_chunks, axis=0, ignore_index=True)
    return allLnData

def split_round_robin(df, target_size, base_path):
    n_rows = df.shape[0]
    n_splits = (n_rows + target_size - 1) // target_size  # Ceiling division
    print(f"Splitting {n_rows} rows into {n_splits} files (target {target_size} rows per file)")
    split_dfs = [[] for _ in range(n_splits)]
    for idx, row in df.iterrows():
        split_dfs[idx % n_splits].append(row)
    out_paths = []
    for i, rows in enumerate(split_dfs):
        split_df = pd.DataFrame(rows, columns=df.columns)
        out_path = f"{base_path}_split{i+1}.tsv"
        split_df.to_csv(out_path, sep='\t', index=False)
        print(f"Output file {out_path}: {split_df.shape[0]} rows, {split_df.shape[1]} columns")
        out_paths.append(out_path)
    return out_paths

def main():
    parser = argparse.ArgumentParser(description="Split a large table into round-robin smaller tables.")
    parser.add_argument("input_file", help="Path to the input TSV file")
    parser.add_argument("--target_size", type=int, default=10000, help="Target number of rows per output file (default: 10000)")
    args = parser.parse_args()

    print(f"Reading: {args.input_file}")
    df = load_selected_columns(args.input_file)
    print(f"Input file shape: {df.shape[0]} rows, {df.shape[1]} columns")

    base, ext = os.path.splitext(args.input_file)
    split_round_robin(df, args.target_size, base)

if __name__ == "__main__":
    main()

