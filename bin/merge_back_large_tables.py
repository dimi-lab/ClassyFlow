import argparse
import os

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Merge large tab-delimited tables efficiently.")
    parser.add_argument('--input_files', nargs='+', required=True, help='List of input TSV files to merge')
    parser.add_argument('--output_file', required=True, help='Output TSV file name')
    return parser.parse_args()

def write_header(input_file, output_file):
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        header = fin.readline()
        fout.write(header)

def append_file(input_file, output_file):
    with open(input_file, 'r') as fin, open(output_file, 'a') as fout:
        # Skip header
        next(fin)
        for line in fin:
            fout.write(line)

def main():
    args = parse_args()
    input_files = args.input_files
    output_file = args.output_file

    if not input_files:
        raise ValueError("No input files provided.")

    # Write header from the first file
    write_header(input_files[0], output_file)

    # Append all files, skipping header for each
    for file in input_files:
        append_file(file, output_file)

if __name__ == "__main__":
    main()