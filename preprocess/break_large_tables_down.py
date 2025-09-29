import argparse
import pandas as pd
import os

def split_round_robin_stream(input_file, target_size):
    # First, count total rows (excluding header)
    with open(input_file) as f:
        total_rows = sum(1 for _ in f) - 1
    n_splits = (total_rows + target_size - 1) // target_size
    print(f"Splitting {total_rows} rows into {n_splits} files (target {target_size} rows per file)")

    # Prepare output file handles and write headers
    base, ext = os.path.splitext(input_file)
    out_paths = [f"{base}_split{i+1}.tsv" for i in range(n_splits)]
    out_files = [open(path, 'w') for path in out_paths]

    # Read header
    with open(input_file) as f:
        header = f.readline()
        for out in out_files:
            out.write(header)

    # Stream rows and write round-robin
    row_counts = [0] * n_splits
    global_row_idx = 0
    for chunk in pd.read_csv(input_file, sep='\t', chunksize=10000, low_memory=False):
        for _, row in chunk.iterrows():
            split_idx = global_row_idx % n_splits
            out_files[split_idx].write('\t'.join(map(str, row.values)) + '\n')
            row_counts[split_idx] += 1
            global_row_idx += 1

    for out in out_files:
        out.close()

    for i, path in enumerate(out_paths):
        print(f"Output file {path}: {row_counts[i]} rows (plus header)")

def main():
    parser = argparse.ArgumentParser(description="Split a large table into round-robin smaller tables (memory efficient).")
    parser.add_argument("-i", "--input_file", help="Path to the input TSV file", required=True)
    parser.add_argument("--target_size", type=int, default=20000, help="Target number of rows per output file (default: 20000)")
    args = parser.parse_args()

    split_round_robin_stream(args.input_file, args.target_size)

if __name__ == "__main__":
    main()

