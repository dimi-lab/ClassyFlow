#!/usr/bin/env python3

import sys, os, time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from scipy.stats import boxcox
from scipy.stats import pearsonr
from jinja2 import Template
from pathlib import Path
import base64
from io import BytesIO

# Set style for professional plots
plt.style.use('default')
sns.set_palette("husl")

def calculate_cv_metrics(df, df_transformed, batchName, targetFeature):
    """Calculate coefficient of variation for each marker by slide"""
    cv_data = []
    
    # Get mean columns
    mean_cols = [col for col in df.columns if targetFeature in col]
    
    for col in mean_cols:
        for slide in df['Slide'].unique():
            slide_data_orig = df[df['Slide'] == slide][col]
            slide_data_trans = df_transformed[df_transformed['Slide'] == slide][col]
            
            # Calculate CV (std/mean) if mean > 0
            cv_orig = slide_data_orig.std() / slide_data_orig.mean() if slide_data_orig.mean() > 0 else np.nan
            cv_trans = slide_data_trans.std() / slide_data_trans.mean() if slide_data_trans.mean() > 0 else np.nan
            
            cv_data.append({
                'marker': col.replace(targetFeature, ''),
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

def create_all_transformation_plots_html(df, df_transformed, bxcxMetrics, batchName, target_features=None, transformation_type='Box-Cox'):
    """
    Create individual plots for selected markers and generate single HTML with dropdown
    
    Args:
        df: Original dataframe
        df_transformed: Transformed dataframe
        bxcxMetrics: Box-Cox metrics dataframe
        batchName: Name of the batch
        target_features: List of strings to match column names (e.g., ['Cell: Mean', 'Cell: Max'])
        transformation_type: Type of transformation
    """
    
    # Parse comma-separated target features
    target_list = [f.strip() for f in target_features.split(',')]
    
    # Find all columns that match any of the target features
    target_cols = []
    for target in target_list:
        matching_cols = [col for col in df.columns if target in col]
        target_cols.extend(matching_cols)
    
    # Remove duplicates while preserving order
    target_cols = list(dict.fromkeys(target_cols))
    
    if not target_cols:
        print(f"Error: No columns found matching any of the target features: {target_features}")
        return {}
    
    print(f"Total columns to plot: {len(target_cols)}")
    
    # Group columns by feature type for organization
    feature_groups = {}
    for col in target_cols:
        last_field = col.split(': ')[-1]
        group = last_field.strip().title()
        
        if group not in feature_groups:
            feature_groups[group] = []
        feature_groups[group].append(col)
    
    # Store all plot data organized by group
    all_metrics_data = {}
    
    for group_name, group_cols in feature_groups.items():
        if not group_cols:
            continue
            
        print(f"Processing {len(group_cols)} columns for {group_name} group...")
        
        # Store plot information for this group
        plot_data = []
        
        # Create individual plots for each marker
        for i, marker_col in enumerate(group_cols):
            # Get lambda value from metrics
            lambda_row = bxcxMetrics[bxcxMetrics['Feature'] == marker_col]
            if not lambda_row.empty:
                lambda_value = lambda_row.iloc[0]['Lambda']
            else:
                lambda_value = 'N/A'
            
            # Calculate correlation
            # Align both Series by index and drop NaNs together
            paired = pd.concat([df[marker_col], df_transformed[marker_col]], axis=1, join='inner').dropna()

            if len(paired) > 0:
                correlation, _ = pearsonr(paired.iloc[:, 0], paired.iloc[:, 1])
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
            
            # Standardized formatting
            ax.set_title(f'{transformation_type} Transform: {marker_col}', fontweight='bold', fontsize=11)
            ax.set_xlabel('Original Value', fontsize=10)
            ax.set_ylabel('Transformed Value', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add text box with stats
            textstr = f'λ = {lambda_value}\nr = {correlation:.3f}'
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
                'lambda_value': lambda_value,
                'correlation': correlation
            })
        
        if plot_data:
            # Count successful and failed transforms for this group
            group_metrics = bxcxMetrics[bxcxMetrics['Feature'].isin(group_cols)]
            successful = len(group_metrics[~group_metrics['Lambda'].isin(['Failed', 'SkippedEmpty'])])
            failed = len(group_metrics[group_metrics['Lambda'].isin(['Failed', 'SkippedEmpty'])])
            
            all_metrics_data[group_name] = {
                'plots': plot_data,
                'successful': successful,
                'failed': failed,
                'total': len(group_cols)
            }
    
    # Generate single HTML with dropdown
    if all_metrics_data:
        generate_combined_html(all_metrics_data, batchName, transformation_type)
    
    return all_metrics_data

def generate_combined_html(all_metrics_data, batchName, transformation_type):
    """Generate a single HTML file with dropdown to switch between feature groups"""
    
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
            grid-template-columns: repeat(3, 1fr);
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
        <p><strong>Batch:</strong> {batchName} | <strong>Generated:</strong> {time.strftime("%Y-%m-%d %H:%M")}</p>
    </div>
    
    <div class="controls">
        <label for="metricSelector" style="margin-right: 10px; font-weight: bold;">Select Feature Group:</label>
        <select id="metricSelector" class="metric-selector" onchange="showMetric(this.value)">
            <option value="">-- Select a Feature Group --</option>
"""
    
    # Add options for each feature group
    for group_name in all_metrics_data.keys():
        html_content += f'            <option value="{group_name}">{group_name} ({all_metrics_data[group_name]["total"]} features)</option>\n'
    
    html_content += """        </select>
    </div>
"""
    
    # Add content for each feature group
    for group_name, group_data in all_metrics_data.items():
        html_content += f"""
    <div class="stats-summary" id="stats-{group_name.replace(' ', '_')}">
        <h2>{group_name} Transformation Summary</h2>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-value">{group_data['total']}</div>
                <div class="stat-label">Total Features</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{group_data['successful']}</div>
                <div class="stat-label">Successful Transforms</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{group_data['failed']}</div>
                <div class="stat-label">Failed Transforms</div>
            </div>
        </div>
    </div>
    
    <div class="plots-container" id="plots-{group_name.replace(' ', '_')}">
"""
        
        for plot in group_data['plots']:
            correlation_class = ''
            if plot['correlation'] > 0.8:
                correlation_class = 'good-correlation'
            elif plot['correlation'] < 0.5:
                correlation_class = 'poor-correlation'
            
            html_content += f"""        <div class="plot-item">
            <h3>{plot['marker_name']}</h3>
            <img src="data:image/png;base64,{plot['img_b64']}" alt="{plot['marker_name']} transformation">
            <div class="plot-stats">
                <span>Lambda: {plot['lambda_value']}</span>
                <span class="{correlation_class}">Correlation: {plot['correlation']:.3f}</span>
            </div>
        </div>
"""
        
        html_content += "    </div>\n"
    
    # Add JavaScript for switching between groups
    html_content += """
    <script>
        function showMetric(groupName) {
            // Hide all stats and plots
            const allStats = document.querySelectorAll('.stats-summary');
            const allPlots = document.querySelectorAll('.plots-container');
            
            allStats.forEach(el => el.classList.remove('active'));
            allPlots.forEach(el => el.classList.remove('active'));
            
            // Show selected group
            if (groupName) {
                const safeGroupName = groupName.replace(' ', '_');
                const statsEl = document.getElementById('stats-' + safeGroupName);
                const plotsEl = document.getElementById('plots-' + safeGroupName);
                if (statsEl) statsEl.classList.add('active');
                if (plotsEl) plotsEl.classList.add('active');
            }
        }
        
        // Show first group by default
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
    html_filename = f'boxcox_all_plots_{batchName}.html'
    with open(html_filename, 'w') as f:
        f.write(html_content)
    
    print(f"Combined HTML report generated: {html_filename}")

def get_max_value(df):
    """Helper function from original code"""
    values = df.values.flatten()
    filtered_values = values[np.isfinite(values)]
    if filtered_values.size > 0:
        max_value = np.max(filtered_values)
    else:
        max_value = 65535
    return max_value

def collect_and_transform(df, batchName, quantType, nucMark, plotFraction, targetFeature=None):
    """Main transformation and plotting function"""
    
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
    bxcxMetrics.to_csv(f"BoxCoxRecord_{myFileIdx}.csv", index=False)

    # Calculate CV metrics
    cv_df = calculate_cv_metrics(df, bcDf, batchName, targetFeature)
    
    # Get worst performing markers
    worst_markers = get_worst_performing_markers(cv_df, n_markers=5)
    
    # Create HTML with transformation plots for selected features
    all_plot_data = create_all_transformation_plots_html(
        df, bcDf, bxcxMetrics, batchName, 
        target_features=targetFeature,
        transformation_type='Box-Cox'
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Box-Cox transform quantification tables and generate QC plots.")
    parser.add_argument('--pickleTable', required=True, help='Input pickle file')
    parser.add_argument('--batchID', required=True, help='Batch ID for output file naming')
    parser.add_argument('--quantType', required=True, help='QuPath object type (e.g., CellObject)')
    parser.add_argument('--nucMark', required=True, help='Nucleus marker name (e.g., DAPI)')
    parser.add_argument('--plotFraction', type=float, default=0.25, help='Fraction of data to plot for QC (default: 0.25)')
    parser.add_argument('--target-feature', dest='targetFeature', default='Cell: Mean',
                       help='Comma-separated list of column name patterns to plot (e.g., "Cell: Mean,Cell: Max")')
    
    args = parser.parse_args()
    myData = pd.read_pickle(args.pickleTable)
    myFileIdx = args.batchID
    quantType = args.quantType
    nucMark = args.nucMark
    plotFraction = args.plotFraction
    targetFeature = args.targetFeature

    metrics = collect_and_transform(myData, myFileIdx, quantType, nucMark, plotFraction, targetFeature)
    # Save metrics to JSON
    with open(f'boxcox_results_{args.batchID}.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)