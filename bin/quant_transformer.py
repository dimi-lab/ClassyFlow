#!/usr/bin/env python3

import sys, os, time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pprint import pprint
from scipy.stats import pearsonr, boxcox
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from jinja2 import Template
from pathlib import Path
import base64
from io import BytesIO

# Set style for professional plots
plt.style.use('default')
sns.set_palette("husl")

# =============================================================================
# SHARED UTILITY FUNCTIONS
# =============================================================================

def calculate_cv_metrics(df, df_transformed, batchName, targetFeature=None):
    """Calculate coefficient of variation for each marker by slide"""
    cv_data = []
    
    # Get mean columns - adjust based on targetFeature if provided
    if targetFeature and 'boxcox' in targetFeature.lower():
        # For boxcox, use the targetFeature parameter
        mean_cols = [col for col in df.columns if targetFeature in col]
    else:
        # For other methods, use 'Mean' columns
        mean_cols = [col for col in df.columns if 'Mean' in col]
    
    for col in mean_cols:
        for slide in df['Slide'].unique():
            slide_data_orig = df[df['Slide'] == slide][col]
            slide_data_trans = df_transformed[df_transformed['Slide'] == slide][col]
            
            # Calculate CV (std/mean) if mean > 0
            cv_orig = slide_data_orig.std() / slide_data_orig.mean() if slide_data_orig.mean() > 0 else np.nan
            cv_trans = slide_data_trans.std() / slide_data_trans.mean() if slide_data_trans.mean() > 0 else np.nan
            
            if targetFeature and 'boxcox' in targetFeature.lower():
                marker_name = col.replace(targetFeature, '')
            else:
                marker_name = col.replace('Cell: ', '').replace(': Mean', '')
            
            cv_data.append({
                'marker': marker_name,
                'slide': slide,
                'cv_original': cv_orig,
                'cv_transformed': cv_trans,
                'cv_improvement': cv_orig - cv_trans if not (np.isnan(cv_orig) or np.isnan(cv_trans)) else np.nan
            })
    
    cv_df = pd.DataFrame(cv_data)
    return cv_df

def get_worst_performing_markers(cv_df, n_markers=5):
    """Identify markers with worst CV performance"""
    if cv_df.empty:
        return []
        
    marker_performance = cv_df.groupby('marker').agg({
        'cv_improvement': 'mean',
        'cv_transformed': 'mean'
    }).reset_index()
    
    # Sort by smallest improvement (or negative improvement)
    worst_markers = marker_performance.nsmallest(n_markers, 'cv_improvement')['marker'].tolist()
    return worst_markers

def create_transformation_plots_for_method(df, df_transformed, batchName, transformation_type, **kwargs):
    """Create transformation plots based on method type"""
    
    if transformation_type == 'Log':
        return create_log_plots_html(df, df_transformed, batchName, transformation_type)
    elif transformation_type == 'Quantile':
        quantile_split = kwargs.get('quantile_split', 100)
        return create_quantile_plots_html(df, df_transformed, batchName, transformation_type, quantile_split)
    elif transformation_type == 'MinMax':
        return create_minmax_plots_html(df, df_transformed, batchName, transformation_type)
    elif transformation_type == 'Box-Cox':
        bxcx_metrics = kwargs.get('boxcox_metrics')
        target_features = kwargs.get('target_features', 'Cell: Mean')
        return create_boxcox_plots_html(df, df_transformed, bxcx_metrics, batchName, target_features, transformation_type)
    else:
        return {}

def create_quantile_plots_html(df, df_transformed, batchName, transformation_type, quantile_split):
    """Create plots for quantile transformation - simplified for completion"""
    # Return empty dict for now - can be implemented fully if needed
    print(f"Quantile HTML generation placeholder - quantiles: {quantile_split}")
    return {}

def create_minmax_plots_html(df, df_transformed, batchName, transformation_type):
    """Create plots for minmax transformation - simplified for completion"""
    # Return empty dict for now - can be implemented fully if needed
    print(f"MinMax HTML generation placeholder")
    return {}

def create_boxcox_plots_html(df, df_transformed, bxcx_metrics, batchName, target_features, transformation_type):
    """Create plots for boxcox transformation - simplified for completion"""
    # Return empty dict for now - can be implemented fully if needed
    print(f"Box-Cox HTML generation placeholder - target features: {target_features}")
    return {}

# =============================================================================
# METHOD-SPECIFIC PLOT FUNCTIONS
# =============================================================================

def create_log_plots_html(df, df_transformed, batchName, transformation_type):
    """Create plots for log transformation"""
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
            
            # Create standardized figure
            plt.style.use('default')
            fig, ax = plt.subplots(figsize=(8, 5))
            plt.rcParams.update({
                'font.size': 10,
                'font.family': 'sans-serif',
                'axes.linewidth': 1
            })
            
            # Scatter plot
            ax.scatter(df[marker_col], df_transformed[marker_col], alpha=0.6, s=1, color='blue')
            
            # Add diagonal reference line
            max_orig = df[marker_col].max()
            max_trans = df_transformed[marker_col].max()
            ax.plot([0, max_orig], [0, max_trans], 'r--', alpha=0.7, linewidth=2, label='Identity Line')
            
            # Formatting
            ax.set_title(f'{transformation_type} Transform: {marker_col}', fontweight='bold', fontsize=11)
            ax.set_xlabel('Original Value', fontsize=10)
            ax.set_ylabel('Transformed Value', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add text box with stats
            textstr = f'log(x + 1)\nr = {correlation:.3f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            
            # Convert plot to base64 instead of saving to file
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode('utf-8')
            
            # Store plot information
            plot_data.append({
                'marker_name': marker_col,
                'img_b64': img_b64,
                'correlation': correlation
            })
        
        if plot_data:
            all_metrics_data[metric_type] = {
                'plots': plot_data,
                'total': len(metric_cols)
            }
    
    # Generate single HTML with dropdown
    generate_log_html(all_metrics_data, batchName, transformation_type)
    
    return all_metrics_data

def generate_log_html(all_metrics_data, batchName, transformation_type):
    """Generate HTML for log transformation"""
    
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
        <p><strong>Batch:</strong> {batchName} | <strong>Generated:</strong> {time.strftime("%Y-%m-%d %H:%M:%S")} | <strong>Transform:</strong> log(x + 1)</p>
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
                <div class="stat-value">log(x+1)</div>
                <div class="stat-label">Transform Function</div>
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
            <img src="data:image/png;base64,{plot['img_b64']}" alt="{plot['marker_name']} transformation">
            <div class="plot-stats">
                <span>Transform: log(x+1)</span>
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
    html_filename = f'log_all_plots_{batchName}.html'
    with open(html_filename, 'w') as f:
        f.write(html_content)
    
    print(f"Combined HTML report generated: {html_filename}")

# =============================================================================
# METHOD-SPECIFIC TRANSFORMATION FUNCTIONS
# =============================================================================

def apply_log_transform(df, batchName, plotFraction, quantType, nucMark):
    """Apply log transformation with original logic"""
    
    # Clean image names
    df['Image'] = [e.replace('.ome.tiff', '') for e in df['Image'].tolist()]
    
    # Apply log transformation to numeric columns
    bcDf = df.copy(deep=True)
    numeric_cols = bcDf.select_dtypes(include=[np.number]).columns
    pprint(numeric_cols)
    numeric_cols = [x for x in numeric_cols if "Centroid" not in x]
    bcDf[numeric_cols] = bcDf[numeric_cols].apply(lambda x: np.log(x + 1))
    
    # Calculate CV metrics
    cv_df = calculate_cv_metrics(df, bcDf, batchName)
    
    # Get worst performing markers
    worst_markers = get_worst_performing_markers(cv_df, n_markers=5)
    
    # Create HTML with all transformation plots for all metric types
    all_plot_data = create_transformation_plots_for_method(df, bcDf, batchName, 'Log')
    
    # Save transformation results
    bcDf.to_csv(f"log_transformed_{batchName}.tsv", sep="\t")
    
    # Calculate summary metrics
    results = {
        'transformation_type': 'log',
        'batch_name': batchName,
        'total_markers': len([col for col in df.columns if 'Mean' in col]),
        'total_slides': df['Slide'].nunique(),
        'total_cells': len(df),
        'worst_performing_markers': worst_markers,
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

def apply_quantile_transform(df, batchName, quantType, nucMark, plotFraction, quantileSplit):
    """Apply quantile transformation with original logic"""
    
    # Clean image names
    df['Image'] = [e.replace('.ome.tiff', '') for e in df['Image'].tolist()]
    
    # Apply quantile transformation (preserve original logic)
    scaler = QuantileTransformer(n_quantiles=quantileSplit, random_state=0)
    imgMets = df.filter(regex='(Min|Max|Median|Mean|Std*|Variance|Area)', axis=1)
    df_norm = pd.DataFrame(scaler.fit_transform(imgMets), columns=imgMets.columns)
    df_a = df[df.columns.difference(imgMets.columns)]
    bcDf = pd.concat([df_a.reset_index(drop=True), df_norm], axis=1).fillna(0)
    
    # Calculate CV metrics
    cv_df = calculate_cv_metrics(df, bcDf, batchName)
    
    # Get worst performing markers
    worst_markers = get_worst_performing_markers(cv_df, n_markers=5)
    
    # Create HTML with all transformation plots for all metric types
    all_plot_data = create_transformation_plots_for_method(df, bcDf, batchName, 'Quantile', quantile_split=quantileSplit)
    
    # Save transformation results
    bcDf.to_csv(f"quantile_transformed_{batchName}.tsv", sep="\t")
    
    # Calculate summary metrics
    results = {
        'transformation_type': 'quantile',
        'batch_name': batchName,
        'quantile_splits': quantileSplit,
        'total_markers': len([col for col in df.columns if 'Mean' in col]),
        'total_slides': df['Slide'].nunique(),
        'total_cells': len(df),
        'worst_performing_markers': worst_markers,
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

def apply_minmax_transform(df, batchName, quantType, nucMark, plotFraction):
    """Apply MinMax transformation with original logic"""
    
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
    all_plot_data = create_transformation_plots_for_method(df, bcDf, batchName, 'MinMax')
    
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

def get_max_value(df):
    """Helper function from original boxcox code"""
    values = df.values.flatten()
    filtered_values = values[np.isfinite(values)]
    if filtered_values.size > 0:
        max_value = np.max(filtered_values)
    else:
        max_value = 65535
    return max_value

def apply_boxcox_transform(df, batchName, quantType, nucMark, plotFraction, targetFeature):
    """Apply Box-Cox transformation with original logic"""
    
    # Clean image names
    df['Image'] = df['Image'].str.replace('.ome.tiff', '', regex=False)
    
    # Apply Box-Cox transformation (preserve original logic)
    metrics = []
    bcDf = df.fillna(0).copy()
    stat_cols = list(bcDf.filter(regex='(Min|Max|Median|Mean|Std*|Variance|Area)'))
    
    for fld in stat_cols:
        # Skip columns that are entirely empty (all NaN or zero length)
        col_values = bcDf[fld].dropna()
        if col_values.empty:
            bcDf[fld] = np.nan
            mxLambda = 'SkippedEmpty'
            preMu = np.nan
            metrics.append([fld, preMu, mxLambda, np.nan, np.nan, np.nan])
            continue
        preMu = bcDf[fld].mean()
        try:
            nArr, mxLambda = boxcox(bcDf[fld].add(1).values)
            bcDf[fld] = nArr
            mxLambda = f"{mxLambda:.3f}"
        except Exception:
            bcDf[fld] = 0
            mxLambda = 'Failed'
        metrics.append([fld, preMu, mxLambda, bcDf[fld].mean(), bcDf[fld].min(), bcDf[fld].max()])

    bxcxMetrics = pd.DataFrame(metrics, columns=['Feature', 'Pre_Mean', 'Lambda', 'Post_Mean', 'Post_Min', 'Post_Max'])
    bxcxMetrics.to_csv(f"BoxCoxRecord_{batchName}.csv", index=False)

    # Calculate CV metrics
    cv_df = calculate_cv_metrics(df, bcDf, batchName, targetFeature)
    
    # Get worst performing markers
    worst_markers = get_worst_performing_markers(cv_df, n_markers=5)
    
    # Create HTML with transformation plots for selected features
    all_plot_data = create_transformation_plots_for_method(
        df, bcDf, batchName, 'Box-Cox', 
        boxcox_metrics=bxcxMetrics,
        target_features=targetFeature
    )
    
    # Save transformation results
    bcDf.to_csv(f"boxcox_transformed_{batchName}.tsv", sep="\t", index=False)
    
    # Calculate summary metrics
    boxcox_summary = {
        'total_features': len(bxcxMetrics),
        'successful_transforms': len(bxcxMetrics[bxcxMetrics['Lambda'] != 'Failed']),
        'failed_transforms': len(bxcxMetrics[bxcxMetrics['Lambda'] == 'Failed']),
        'lambda_stats': {
            'mean_lambda': bxcxMetrics[bxcxMetrics['Lambda'] != 'Failed']['Lambda'].astype(float).mean(),
            'median_lambda': bxcxMetrics[bxcxMetrics['Lambda'] != 'Failed']['Lambda'].astype(float).median()
        }
    }
    
    results = {
        'transformation_type': 'boxcox',
        'batch_name': batchName,
        'total_markers': len([col for col in df.columns if 'Mean' in col]),
        'total_slides': df['Slide'].nunique(),
        'total_cells': len(df),
        'worst_performing_markers': worst_markers,
        'boxcox_metrics': boxcox_summary,
        'cv_metrics': {
            'mean_cv_improvement': cv_df['cv_improvement'].mean(),
            'median_cv_improvement': cv_df['cv_improvement'].median(),
            'markers_improved': (cv_df['cv_improvement'] > 0).sum(),
            'markers_worsened': (cv_df['cv_improvement'] < 0).sum()
        },
        'all_transformation_plots': all_plot_data,
        'target_features': targetFeature,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return results

# =============================================================================
# ARGUMENT PARSER AND MAIN FUNCTION
# =============================================================================

def main():
    """Main function with unified argument parser and method dispatcher"""
    parser = argparse.ArgumentParser(description="Apply normalization transformations to quantification data.")
    
    # Common arguments for all methods
    parser.add_argument('--pickleTable', required=True, help='Input pickle file')
    parser.add_argument('--batchID', required=True, help='Batch ID for output file naming')
    parser.add_argument('--quantType', required=True, help='QuPath object type (e.g., CellObject)')
    parser.add_argument('--nucMark', required=True, help='Nucleus marker name (e.g., DAPI)')
    parser.add_argument('--plotFraction', type=float, default=0.25, help='Fraction of data to plot for QC (default: 0.25)')
    parser.add_argument('--method', required=True, choices=['log', 'quantile', 'minmax', 'boxcox'],
                       help='Transformation method to apply')
    
    # Method-specific arguments
    parser.add_argument('--quantileSplit', type=int, 
                       help='Number of quantiles for QuantileTransformer (required for quantile method)')
    parser.add_argument('--target-feature', dest='targetFeature', default='Cell: Mean',
                       help='Comma-separated list of column name patterns to plot for boxcox method (default: "Cell: Mean")')
    
    args = parser.parse_args()
    
    # Validate method-specific arguments
    if args.method == 'quantile' and args.quantileSplit is None:
        parser.error("--quantileSplit is required when using quantile method")
    
    # Load data
    print(f"Loading data from {args.pickleTable}...")
    myData = pd.read_pickle(args.pickleTable)
    
    # Dispatch to appropriate transformation method
    print(f"Applying {args.method} transformation...")
    
    if args.method == 'log':
        metrics = apply_log_transform(myData, args.batchID, args.plotFraction, args.quantType, args.nucMark)
    elif args.method == 'quantile':
        metrics = apply_quantile_transform(myData, args.batchID, args.quantType, args.nucMark, args.plotFraction, args.quantileSplit)
    elif args.method == 'minmax':
        metrics = apply_minmax_transform(myData, args.batchID, args.quantType, args.nucMark, args.plotFraction)
    elif args.method == 'boxcox':
        metrics = apply_boxcox_transform(myData, args.batchID, args.quantType, args.nucMark, args.plotFraction, args.targetFeature)
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    # Save metrics to JSON
    output_json = f'{args.method}_results_{args.batchID}.json'
    with open(output_json, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    print(f"\nTransformation complete!")
    print(f"- Transformed data: {args.method}_transformed_{args.batchID}.tsv")
    print(f"- Metrics: {output_json}")
    print(f"- HTML report: {args.method}_all_plots_{args.batchID}.html")
    if args.method == 'boxcox':
        print(f"- Box-Cox record: BoxCoxRecord_{args.batchID}.csv")

if __name__ == "__main__":
    main()