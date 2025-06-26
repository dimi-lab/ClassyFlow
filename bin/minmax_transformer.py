#!/usr/bin/env python3

import sys, os, time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import argparse

# Set style for professional plots
plt.style.use('default')
sns.set_palette("husl")

def calculate_cv_metrics(df, df_transformed, batchName):
    """Calculate coefficient of variation for each marker by slide"""
    cv_data = []
    
    # Get mean columns
    mean_cols = [col for col in df.columns if 'Mean' in col]
    
    for col in mean_cols:
        for slide in df['Slide'].unique():
            slide_data_orig = df[df['Slide'] == slide][col]
            slide_data_trans = df_transformed[df_transformed['Slide'] == slide][col]
            
            # Calculate CV (std/mean) if mean > 0
            cv_orig = slide_data_orig.std() / slide_data_orig.mean() if slide_data_orig.mean() > 0 else np.nan
            cv_trans = slide_data_trans.std() / slide_data_trans.mean() if slide_data_trans.mean() > 0 else np.nan
            
            cv_data.append({
                'marker': col.replace('Cell: ', '').replace(': Mean', ''),
                'slide': slide,
                'cv_original': cv_orig,
                'cv_transformed': cv_trans,
                'cv_improvement': cv_orig - cv_trans if not (np.isnan(cv_orig) or np.isnan(cv_trans)) else np.nan
            })
    
    cv_df = pd.DataFrame(cv_data)
    return cv_df

def create_cv_heatmap(cv_df, batchName, plot_type='improvement'):
    """Create CV heatmap showing improvement or transformed values"""
    
    if plot_type == 'improvement':
        pivot_data = cv_df.pivot(index='marker', columns='slide', values='cv_improvement')
        title = 'CV Improvement by Marker and Slide (Original - Transformed)'
        filename = f'cv_improvement_heatmap_{batchName}.png'
        cmap = 'RdYlGn'  # Red = worse, Green = better
    else:
        pivot_data = cv_df.pivot(index='marker', columns='slide', values='cv_transformed')
        title = 'Coefficient of Variation After Transformation'
        filename = f'cv_transformed_heatmap_{batchName}.png'
        cmap = 'YlOrRd_r'  # Lower CV = better
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap=cmap, center=0 if plot_type == 'improvement' else None,
                cbar_kws={'label': 'CV Improvement' if plot_type == 'improvement' else 'CV'})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Slide', fontsize=12)
    plt.ylabel('Marker', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def get_worst_performing_markers(cv_df, n_markers=5):
    """Identify markers with worst CV performance"""
    marker_performance = cv_df.groupby('marker').agg({
        'cv_improvement': 'mean',
        'cv_transformed': 'mean'
    }).reset_index()
    
    # Sort by smallest improvement (or negative improvement)
    worst_markers = marker_performance.nsmallest(n_markers, 'cv_improvement')['marker'].tolist()
    return worst_markers

def create_slide_boxplots(df, df_transformed, filename, plotFraction):
    """Create before/after boxplots by slide"""
    
    # Sample data for plotting (preserve original logic)
    smTble_orig = df.groupby('Slide', group_keys=False).apply(lambda x: x.sample(frac=plotFraction))
    smTble_trans = df_transformed.groupby('Slide', group_keys=False).apply(lambda x: x.sample(frac=plotFraction))
    
    # Filter for mean/median columns
    df_batching_orig = smTble_orig.filter(regex='(Mean|Median|Slide)', axis=1)
    df_batching_trans = smTble_trans.filter(regex='(Mean|Median|Slide)', axis=1)
    
    # Melt for plotting
    df_melted_orig = pd.melt(df_batching_orig, id_vars=["Slide"])
    df_melted_trans = pd.melt(df_batching_trans, id_vars=["Slide"])
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Original values
    sns.boxplot(x='Slide', y='value', data=df_melted_orig, ax=ax1, 
                color="#CD7F32", showfliers=False)
    ax1.set_title('Combined Marker Distribution (Original Values)', fontsize=14, fontweight='bold')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Transformed values
    sns.boxplot(x='Slide', y='value', data=df_melted_trans, ax=ax2, 
                color="#50C878", showfliers=False)
    ax2.set_title('Combined Marker Distribution (MinMax Transformed)', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def create_transformation_examples(df, df_transformed, filename, nucMark, n_examples=5):
    """Create transformation scatter plots for representative markers"""
    
    mean_cols = [col for col in df.columns if 'Mean' in col]
    nuc_col = next((col for col in mean_cols if nucMark in col), mean_cols[0])
    
    # Select every nth marker to get good representation
    step = max(1, len(mean_cols) // n_examples)
    example_markers = mean_cols[::step][:n_examples]
    
    # Ensure nucleus marker is included
    if nuc_col not in example_markers:
        example_markers[0] = nuc_col
    
    fig, axes = plt.subplots(1, n_examples, figsize=(20, 4))
    if n_examples == 1:
        axes = [axes]
    
    for i, marker in enumerate(example_markers):
        # Create scatter plot
        axes[i].scatter(df[marker], df_transformed[marker], alpha=0.6, s=1)
        
        # Add diagonal line
        max_orig = df[marker].max()
        max_trans = df_transformed[marker].max()
        axes[i].plot([0, max_orig], [0, max_trans], 'r--', alpha=0.7, linewidth=2)
        
        # Formatting
        marker_name = marker.replace('Cell: ', '').replace(': Mean', '')
        axes[i].set_title(f'MinMax Transform: {marker_name}', fontweight='bold')
        axes[i].set_xlabel('Original Value')
        axes[i].set_ylabel('Transformed Value')
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Transformation Examples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def create_distribution_comparison(df, df_transformed, worst_markers, filename):
    """Create density plots for worst performing markers"""
    
    n_markers = len(worst_markers)
    fig, axes = plt.subplots(n_markers, 1, figsize=(12, 3*n_markers))
    if n_markers == 1:
        axes = [axes]
    
    for i, marker in enumerate(worst_markers):
        # Find the actual column name
        marker_col = next((col for col in df.columns if marker in col and 'Mean' in col), None)
        if marker_col is None:
            continue
            
        # Create density plots
        axes[i].hist(df[marker_col], bins=50, alpha=0.7, density=True, 
                    label='Original', color='#CD7F32')
        axes[i].hist(df_transformed[marker_col], bins=50, alpha=0.7, density=True, 
                    label='MinMax Transformed', color='#50C878')
        
        axes[i].set_title(f'{marker} Distribution Comparison', fontweight='bold')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Density')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Distribution Comparison: Worst Performing Markers', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def collect_and_transform(df, batchName, qTyp, nucMark, plotFraction):
    """Main transformation and plotting function"""
    
    # Clean image names (preserve original logic)
    df['Image'] = [e.replace('.ome.tiff', '') for e in df['Image'].tolist()]
    
    # Apply MinMax transformation (preserve original logic)
    scaler = MinMaxScaler(feature_range=(-2, 2))
    
    # grab just quant fields
    imgMets = df.filter(regex='(Min|Max|Median|Mean|StdDev)', axis=1)
    df_norm = pd.DataFrame(scaler.fit_transform(imgMets), columns=imgMets.columns)

    df_a = df[df.columns.difference(imgMets.columns)]
    bcDf = pd.concat([df_a.reset_index(drop=True), df_norm], axis=1).fillna(0)
    
    # Calculate CV metrics
    cv_df = calculate_cv_metrics(df, bcDf, batchName)
    
    # Get worst performing markers
    worst_markers = get_worst_performing_markers(cv_df, n_markers=5)
    
    # Generate plots
    results = {
        'slide_boxplots': f'minmax_slide_boxplots_{batchName}.png',
        'cv_heatmap': f'minmax_cv_improvement_heatmap_{batchName}.png',
        'transformation_examples': f'minmax_transformation_examples_{batchName}.png',
        'distribution_comparison': f'minmax_distribution_comparison_{batchName}.png'
    }
    
    # 1. Slide boxplots
    create_slide_boxplots(df, bcDf, results['slide_boxplots'], plotFraction)
    
    # 2. CV heatmap
    create_cv_heatmap(cv_df, results['cv_heatmap'], plot_type='improvement')
    
    # 3. Transformation examples
    create_transformation_examples(df, bcDf, results['transformation_examples'], nucMark)
    
    # 4. Distribution comparison
    create_distribution_comparison(df, bcDf, worst_markers, results['distribution_comparison'])
    
    # Save transformation results
    bcDf.to_csv(f"minmax_transformed_{batchName}.tsv", sep="\t")
    
    # Calculate summary metrics
    minmax_summary = {
        'feature_range': '(-2, 2)',
        'total_features_transformed': len(imgMets.columns),
        'scaling_stats': {
            'mean_scale_factor': (df_norm.max() - df_norm.min()).mean(),
            'min_transformed_value': df_norm.min().min(),
            'max_transformed_value': df_norm.max().max()
        }
    }
    
    results.update({
        'transformation_type': 'minmax',
        'batch_name': batchName,
        'total_markers': len([col for col in df.columns if 'Mean' in col]),
        'total_slides': df['Slide'].nunique(),
        'total_cells': len(df),
        'worst_performing_markers': worst_markers,
        'minmax_metrics': minmax_summary,
        'cv_metrics': {
            'mean_cv_improvement': cv_df['cv_improvement'].mean(),
            'median_cv_improvement': cv_df['cv_improvement'].median(),
            'markers_improved': (cv_df['cv_improvement'] > 0).sum(),
            'markers_worsened': (cv_df['cv_improvement'] < 0).sum()
        },
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    })
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MinMax transform quantification tables and generate QC plots.")
    parser.add_argument('--pickleTable', required=True, help='Input pickle file')
    parser.add_argument('--batchID', required=True, help='Batch ID for output file naming')
    parser.add_argument('--quantType', required=True, help='QuPath object type (e.g., CellObject)')
    parser.add_argument('--nucMark', required=True, help='Nucleus marker name (e.g., DAPI)')
    parser.add_argument('--plotFraction', type=float, default=0.25, help='Fraction of data to plot for QC (default: 0.25)')

    args = parser.parse_args()

    myData = pd.read_pickle(args.pickleTable)
    myFileIdx = args.batchID
    quantType = args.quantType
    nucMark = args.nucMark
    plotFraction = args.plotFraction
            
    metrics = collect_and_transform(myData, myFileIdx, quantType, nucMark, plotFraction)
    
    # Save metrics to JSON
    with open(f'minmax_results_{args.batchID}.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)