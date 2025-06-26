#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Calculate bin density for cell predictions.")
    parser.add_argument('--input_tsv', required=True, help='Input TSV file with cell predictions')
    parser.add_argument('--bin_size', type=float, default=120, help='Bin size for density calculation (default: 120)')
    parser.add_argument('--density_cutoff', type=int, default=3, help='Density cutoff for low density (default: 3)')
    args = parser.parse_args()

    df = pd.read_csv(args.input_tsv, sep='\t')

    # Calculate bin indices
    df['binX'] = np.floor(df['Centroid X µm'] / args.bin_size).astype(int)
    df['binY'] = np.floor(df['Centroid Y µm'] / args.bin_size).astype(int)

    # Count cells in each bin
    bincounts = df.groupby(['binX', 'binY']).size().reset_index(name='bin_density')

    # Merge bin density back to main dataframe
    df = df.merge(bincounts, on=['binX', 'binY'], how='left')

    # Mark low bin density
    df['low_bin_density'] = df['bin_density'] <= args.density_cutoff

    # Optionally, print summary stats
    n_lowdensity = (df['bin_density'] <= args.density_cutoff).sum()
    pct_lowdensity = n_lowdensity / len(df)
    print(f"Low density cells: {n_lowdensity} ({pct_lowdensity:.1%})")

    # Write output
    output_tsv = args.input_tsv.replace('_PRED.tsv', '_qPRED.tsv')
    df.to_csv(output_tsv, sep='\t', index=False)

if __name__ == "__main__":
    main()