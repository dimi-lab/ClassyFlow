#!/usr/bin/env python3

import argparse
import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

def parse_args():
    
    return parser.parse_args()

def read_prediction_files(input_dir, file_pattern="*qPRED.tsv"):
    """Read all prediction files and combine into a single dataframe"""
    prediction_files = glob.glob(os.path.join(input_dir, file_pattern))
    
    if not prediction_files:
        raise ValueError(f"No files found matching pattern {file_pattern} in {input_dir}")
    
    all_data = []
    
    for file_path in prediction_files:
        try:
            df = pd.read_csv(file_path, sep='\t')
            
            # Extract sample name from filename
            sample_name = Path(file_path).stem.replace('.ome.tiff_qPRED', '')
            df['Sample'] = sample_name
            
            all_data.append(df)
            
            print(f"Loaded {len(df):,} cells from {sample_name}")
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No valid prediction files could be read")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df["Sample"] = combined_df["Image"].str.replace('.ome.tif', '')

    return combined_df

def create_abundance_plot(df, output_file):
    """Create stacked bar plot with simplified dynamic sizing"""
    # Calculate dataset characteristics for sizing
    n_samples = df['Sample'].nunique()
    n_cell_types = df['CellTypePrediction'].nunique()
    
    # Simple dynamic sizing
    plot_width = max(10, min(20, 8 + n_samples * 0.5))
    plot_height = max(6, min(12, 6 + n_cell_types * 0.2))
    
    # Calculate proportions and pivot
    proportions_list = []
    for sample in sorted(df['Sample'].unique()):
        sample_df = df[df['Sample'] == sample]
        proportions = sample_df['CellTypePrediction'].value_counts(normalize=True) * 100
        for cell_type, percentage in proportions.items():
            proportions_list.append({
                'Sample': sample,
                'CellType': cell_type,
                'Percentage': percentage
            })
    
    proportions_df = pd.DataFrame(proportions_list)
    pivot_df = proportions_df.pivot(index='Sample', columns='CellType', values='Percentage').fillna(0)
    print(pivot_df)
    # Get overall cell type order (most abundant first)
    overall_abundance = df['CellTypePrediction'].value_counts()
    cell_type_order = overall_abundance.index.tolist()
    
    # Reorder columns to match abundance order
    pivot_df = pivot_df.reindex(columns=cell_type_order, fill_value=0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(plot_width, plot_height), dpi=300)
    
    # Generate colors
    colors = sns.color_palette("Set2", n_cell_types)
    
    # Create stacked bars in order
    bottom = np.zeros(len(pivot_df))
    for i, cell_type in enumerate(pivot_df.columns):
        ax.bar(
            range(len(pivot_df)), 
            pivot_df[cell_type], 
            bottom=bottom,
            label=cell_type,
            color=colors[i],
            alpha=0.85,
            edgecolor='white',
            linewidth=0.8
        )
        bottom += pivot_df[cell_type]
        
    # Styling
    ax.set_title('Predicted Cell Type Composition by Sample', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Sample', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Cells (%)', fontsize=12, fontweight='bold')
    
    # X-axis labels
    ax.set_xticks(range(len(pivot_df)))
    rotation = 45 if n_samples <= 15 else 70
    ax.set_xticklabels(pivot_df.index, rotation=rotation, ha='right')
    
    # Y-axis
    ax.set_ylim(0, 100)
    
    # Add sample counts above bars
    sample_counts = df.groupby('Sample').size()
    for i, sample in enumerate(pivot_df.index):
        ax.text(i, 102, f'n={sample_counts[sample]:,}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Legend (matching bar order - bottom to top)
    handles, labels = ax.get_legend_handles_labels()
    # Reverse to match visual stacking order (bottom to top)
    legend = ax.legend(
        reversed(handles), reversed(labels),
        bbox_to_anchor=(1.05, 1), 
        loc='upper left',
        frameon=True,
        title='Cell Type',
        title_fontsize=11,
        fontsize=10
    )
    
    # Clean styling
    ax.grid(True, alpha=0.3)
    sns.despine(top=True, right=True)
    ax.set_facecolor('#fafafa')
    
    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')


def generate_abundance_results(df):

    results = {
        'total_predicted_cells': len(df),
        'most_common_prediction': df['CellTypePrediction'].value_counts().index[0],
        'most_rare_prediction': df['CellTypePrediction'].value_counts().index[-1],
        'total_low_density_cells': df['low_bin_density'].sum(),
        'abundance_plot': "prediction_abundance_plot.png"
    }

    # Create abundance plot
    print("Creating abundance visualization...")
    create_abundance_plot(df, results["abundance_plot"])

    #Save json
    with open("abundance_metrics.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

def main():
    parser = argparse.ArgumentParser(description="Generate summary statistics from cell type prediction files")
    parser.add_argument('--input_dir', required=True, help='Directory containing prediction TSV files')
    args = parser.parse_args()
    
    # Read all prediction files
    print(f"Reading prediction files from {args.input_dir}...")
    df = read_prediction_files(args.input_dir)

    # Generate abundance data
    results = generate_abundance_results(df)
    
    
if __name__ == "__main__":
    main()