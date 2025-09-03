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

def calculate_per_sample_stats(df):
    """Calculate detailed statistics for each sample/ROI"""
    per_sample_stats = []
    
    for sample in sorted(df['Sample'].unique()):
        sample_df = df[df['Sample'] == sample]
        
        # Get cell type counts and percentages
        cell_type_counts = sample_df['CellTypePrediction'].value_counts()
        cell_type_percentages = sample_df['CellTypePrediction'].value_counts(normalize=True) * 100
        
        # Get most common and second most common classes
        most_common_classes = cell_type_counts.head(2)
        
        sample_stats = {
            'sample_name': sample,
            'total_cells': len(sample_df),
            'unique_classes': sample_df['CellTypePrediction'].nunique(),
            'most_common_class': most_common_classes.index[0] if len(most_common_classes) > 0 else None,
            'most_common_percentage': cell_type_percentages.iloc[0] if len(cell_type_percentages) > 0 else 0,
            'second_common_class': most_common_classes.index[1] if len(most_common_classes) > 1 else None,
            'second_common_percentage': cell_type_percentages.iloc[1] if len(cell_type_percentages) > 1 else 0,
            'cell_type_distribution': dict(cell_type_counts),
            'percentage_distribution': dict(cell_type_percentages)
        }
        
        # Add low density cell count if available
        if 'low_bin_density' in sample_df.columns:
            sample_stats['low_density_cells'] = sample_df['low_bin_density'].sum()
        
        per_sample_stats.append(sample_stats)
    
    return per_sample_stats

def create_abundance_plot(df, output_file):
    """Create stacked bar plot with simplified dynamic sizing"""
    # Calculate dataset characteristics for sizing
    n_samples = df['Sample'].nunique()
    n_cell_types = df['CellTypePrediction'].nunique()
    
    # Enhanced dynamic sizing for large datasets
    plot_width = max(12, min(30, 10 + n_samples * 0.4))
    plot_height = max(8, min(12, 7 + n_cell_types * 0.15))
    
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
    
    # Create plot with standardized styling
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(plot_width, plot_height), dpi=300)
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.linewidth': 1
    })
    
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
        
    # Standardized styling
    ax.set_title('Predicted Cell Type Composition by Sample', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Sample', fontsize=11, fontweight='bold')
    ax.set_ylabel('Percentage of Cells (%)', fontsize=11, fontweight='bold')
    
    # X-axis labels - dynamic rotation based on sample count
    ax.set_xticks(range(len(pivot_df)))
    if n_samples <= 10:
        rotation = 30
        fontsize = 10
    elif n_samples <= 25:
        rotation = 45
        fontsize = 9
    else:
        rotation = 70
        fontsize = 8
    ax.set_xticklabels(pivot_df.index, rotation=rotation, ha='right', fontsize=fontsize)
    
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
    """Generate overall abundance results and per-sample statistics"""
    
    # Calculate per-sample statistics
    per_sample_stats = calculate_per_sample_stats(df)
    
    # Get cell type counts and percentages
    cell_type_counts = df['CellTypePrediction'].value_counts()
    cell_type_percentages = (cell_type_counts / len(df) * 100).round(1)
    
    # Most abundant types (top 5)
    most_abundant = []
    for cell_type, percentage in cell_type_percentages.head(5).items():
        most_abundant.append({
            "name": str(cell_type),
            "percentage": float(percentage)
        })
    
    # Rarest cell type
    rarest_type = {
        "name": str(cell_type_percentages.index[-1]),
        "percentage": float(cell_type_percentages.iloc[-1])
    }
    
    # Overall results
    results = {
        'total_predicted_cells': len(df),
        'total_samples': df['Sample'].nunique(),
        'most_common_prediction': df['CellTypePrediction'].value_counts().index[0],
        'most_rare_prediction': df['CellTypePrediction'].value_counts().index[-1],
        'abundance_plot': "prediction_abundance_plot.png",
        'per_sample_statistics': per_sample_stats
    }
    
    # Summary metrics for final report
    summary_metrics = {
        'identified_cell_types': df['CellTypePrediction'].nunique(),
        'most_abundant_types': most_abundant,
        'rarest_cell_type': rarest_type,
        'composition_chart': "prediction_abundance_plot.png"
    }
    
    # Add low density cells if available
    if 'low_bin_density' in df.columns:
        results['total_low_density_cells'] = df['low_bin_density'].sum()
    
    # Create abundance plot
    print("Creating abundance visualization...")
    create_abundance_plot(df, results["abundance_plot"])

    # Save overall results
    with open("abundance_metrics.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save summary metrics for final report
    with open("summary_metrics.json", 'w') as f:
        json.dump(summary_metrics, f, indent=2, default=str)
    print("Saved summary metrics to summary_metrics.json")
    
    # Save individual per-sample JSON files for template processing
    for sample_stats in per_sample_stats:
        sample_name = sample_stats['sample_name']
        sample_file = f"{sample_name}_summary_stats.json"
        with open(sample_file, 'w') as f:
            json.dump(sample_stats, f, indent=2, default=str)
        print(f"Saved detailed statistics for {sample_name} to {sample_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Generate summary statistics from cell type prediction files")
    parser.add_argument('--input_dir', required=True, help='Directory containing prediction TSV files')
    args = parser.parse_args()
    
    # Read all prediction files
    print(f"Reading prediction files from {args.input_dir}...")
    df = read_prediction_files(args.input_dir)

    # Generate abundance data and per-sample statistics
    results = generate_abundance_results(df)
    
    print(f"\nSummary:")
    print(f"- Total cells predicted: {results['total_predicted_cells']:,}")
    print(f"- Total samples/ROIs: {results['total_samples']}")
    print(f"- Most common cell type: {results['most_common_prediction']}")
    print(f"- Generated abundance plot: {results['abundance_plot']}")
    print(f"- Per-sample statistics saved for {len(results['per_sample_statistics'])} samples")
    
if __name__ == "__main__":
    main()