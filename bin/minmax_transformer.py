#!/usr/bin/env python3

import sys, os, time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import argparse
from scipy.stats import pearsonr
from jinja2 import Template
from pathlib import Path

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

def get_worst_performing_markers(cv_df, n_markers=5):
    """Identify markers with worst CV performance"""
    marker_performance = cv_df.groupby('marker').agg({
        'cv_improvement': 'mean',
        'cv_transformed': 'mean'
    }).reset_index()
    
    # Sort by smallest improvement (or negative improvement)
    worst_markers = marker_performance.nsmallest(n_markers, 'cv_improvement')['marker'].tolist()
    return worst_markers

def create_all_transformation_plots_html(df, df_transformed, batchName, transformation_type='MinMax'):
    """Create individual plots for all markers and generate single HTML with dropdown"""
    
    # Create plots directory
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    
    # Define metric types to process
    metric_types = ['Min', 'Max', 'Median', 'Mean', 'Std', 'Variance', 'Area']
    
    # Store all plot data organized by metric type
    all_metrics_data = {}
    
    for metric_type in metric_types:
        # Get columns for this metric type
        if metric_type == 'Std':
            metric_cols = [col for col in df.columns if 'Std' in col]
        else:
            metric_cols = [col for col in df.columns if f': {metric_type}' in col or f' {metric_type}' in col]
        
        if not metric_cols:
            print(f"No columns found for metric type: {metric_type}")
            continue
        
        print(f"Processing {len(metric_cols)} columns for {metric_type} metric...")
        
        # Store plot information for this metric type
        plot_data = []
        
        # Create individual plots for each marker
        for i, marker_col in enumerate(metric_cols):
            # Calculate correlation
            orig_values = df[marker_col].dropna()
            trans_values = df_transformed[marker_col].dropna()
            
            # Ensure same length arrays for correlation
            min_len = min(len(orig_values), len(trans_values))
            if min_len > 0:
                correlation, _ = pearsonr(orig_values[:min_len], trans_values[:min_len])
            else:
                correlation = 0
            
            # Create figure
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # Scatter plot
            ax.scatter(df[marker_col], df_transformed[marker_col], alpha=0.6, s=1, color='blue')
            
            # Add diagonal reference line
            max_orig = df[marker_col].max()
            max_trans = df_transformed[marker_col].max()
            ax.plot([0, max_orig], [0, max_trans], 'r--', alpha=0.7, linewidth=2, label='Identity Line')
            
            # Formatting
            ax.set_title(f'{transformation_type} Transform: {marker_col}', fontweight='bold', fontsize=12)
            ax.set_xlabel('Original Value', fontsize=10)
            ax.set_ylabel('Transformed Value', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add text box with stats
            textstr = f'Range: (-2, 2)\nr = {correlation:.3f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            
            # Save plot with metric type in filename
            safe_marker_name = marker_col.replace("/", "_").replace(" ", "_").replace(":", "")
            plot_filename = f'minmax_{batchName}_{metric_type.lower()}_{i:03d}_{safe_marker_name}.png'
            plot_path = plots_dir / plot_filename
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Store plot information
            plot_data.append({
                'marker_name': marker_col,
                'filename': str(plot_path),
                'correlation': correlation
            })
        
        if plot_data:
            all_metrics_data[metric_type] = {
                'plots': plot_data,
                'total': len(metric_cols)
            }
    
    # Generate single HTML with dropdown
    generate_combined_html(all_metrics_data, batchName, transformation_type)
    
    print(f"Individual plots saved in: {plots_dir}/")
    
    return all_metrics_data

def generate_combined_html(all_metrics_data, batchName, transformation_type):
    """Generate a single HTML file with dropdown to switch between metric types"""
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{transformation_type} Transformation - {batchName}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 10px;
        }}
        .controls {{
            text-align: center;
            margin: 20px 0;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-selector {{
            padding: 10px 20px;
            font-size: 16px;
            border: 2px solid #2196F3;
            border-radius: 5px;
            background-color: white;
            cursor: pointer;
        }}
        .metric-selector:hover {{
            background-color: #f0f0f0;
        }}
        .metadata {{
            text-align: center;
            color: #666;
            margin-bottom: 20px;
        }}
        .stats-summary {{
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: none;
        }}
        .stats-summary.active {{
            display: block;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }}
        .stat-item {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2196F3;
        }}
        .stat-label {{
            font-size: 12px;
            color: #666;
        }}
        .plots-container {{
            display: none;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .plots-container.active {{
            display: grid;
        }}
        .plot-item {{
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .plot-item h3 {{
            margin-top: 0;
            color: #333;
            font-size: 14px;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }}
        .plot-item img {{
            width: 100%;
            height: auto;
        }}
        .plot-stats {{
            display: flex;
            justify-content: space-between;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #eee;
            font-size: 12px;
            color: #666;
        }}
        .good-correlation {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .poor-correlation {{
            color: #f44336;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <h1>{transformation_type} Transformation Results</h1>
    <div class="metadata">
        <p><strong>Batch:</strong> {batchName} | <strong>Generated:</strong> {time.strftime("%Y-%m-%d %H:%M:%S")} | <strong>Range:</strong> (-2, 2)</p>
    </div>
    
    <div class="controls">
        <label for="metricSelector" style="margin-right: 10px; font-weight: bold;">Select Metric Type:</label>
        <select id="metricSelector" class="metric-selector" onchange="showMetric(this.value)">
            <option value="">-- Select a Metric --</option>
"""
    
    # Add options for each metric type
    for metric_type in all_metrics_data.keys():
        html_content += f'            <option value="{metric_type}">{metric_type} ({all_metrics_data[metric_type]["total"]} markers)</option>\n'
    
    html_content += """        </select>
    </div>
"""
    
    # Add content for each metric type
    for metric_type, metric_data in all_metrics_data.items():
        html_content += f"""
    <div class="stats-summary" id="stats-{metric_type}">
        <h2>{metric_type} Transformation Summary</h2>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-value">{metric_data['total']}</div>
                <div class="stat-label">Total Markers</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">(-2, 2)</div>
                <div class="stat-label">Feature Range</div>
            </div>
        </div>
    </div>
    
    <div class="plots-container" id="plots-{metric_type}">
"""
        
        for plot in metric_data['plots']:
            correlation_class = ''
            if plot['correlation'] > 0.8:
                correlation_class = 'good-correlation'
            elif plot['correlation'] < 0.5:
                correlation_class = 'poor-correlation'
            
            html_content += f"""        <div class="plot-item">
            <h3>{plot['marker_name']}</h3>
            <img src="{plot['filename']}" alt="{plot['marker_name']} transformation">
            <div class="plot-stats">
                <span>Range: (-2, 2)</span>
                <span class="{correlation_class}">Correlation: {plot['correlation']:.3f}</span>
            </div>
        </div>
"""
        
        html_content += "    </div>\n"
    
    # Add JavaScript for switching between metrics
    html_content += """
    <script>
        function showMetric(metricType) {
            // Hide all stats and plots
            const allStats = document.querySelectorAll('.stats-summary');
            const allPlots = document.querySelectorAll('.plots-container');
            
            allStats.forEach(el => el.classList.remove('active'));
            allPlots.forEach(el => el.classList.remove('active'));
            
            // Show selected metric
            if (metricType) {
                const statsEl = document.getElementById('stats-' + metricType);
                const plotsEl = document.getElementById('plots-' + metricType);
                if (statsEl) statsEl.classList.add('active');
                if (plotsEl) plotsEl.classList.add('active');
            }
        }
        
        // Show first metric by default
        window.onload = function() {
            const selector = document.getElementById('metricSelector');
            if (selector.options.length > 1) {
                selector.selectedIndex = 1;
                showMetric(selector.options[1].value);
            }
        }
    </script>
</body>
</html>"""
    
    # Save HTML file
    html_filename = f'minmax_all_plots_{batchName}.html'
    with open(html_filename, 'w') as f:
        f.write(html_content)
    
    print(f"Combined HTML report generated: {html_filename}")

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

    # Create HTML with all transformation plots for all metric types
    all_plot_data = create_all_transformation_plots_html(df, bcDf, batchName, 'MinMax')
    
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
    
    results = {
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
        'all_transformation_plots': all_plot_data,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
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