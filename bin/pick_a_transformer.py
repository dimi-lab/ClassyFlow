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
from pathlib import Path
import base64
from io import BytesIO

# Set style for professional plots
plt.style.use('default')
sns.set_palette("husl")

# =============================================================================
# CONSTANTS
# =============================================================================

PLOT_CONFIG = {
    'figsize': (8, 5),
    'dpi': 300,
    'alpha': 0.6,
    'point_size': 1,
    'font_size': 10,
    'title_size': 11,
    'linewidth': 1
}

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{transformation_type} Transformation - {batch_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        h1 {{ color: #333; text-align: center; margin-bottom: 10px; }}
        .controls {{ text-align: center; margin: 20px 0; background-color: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-selector {{ padding: 10px 20px; font-size: 16px; border: 2px solid #2196F3; border-radius: 5px; background-color: white; cursor: pointer; }}
        .metric-selector:hover {{ background-color: #f0f0f0; }}
        .metadata {{ text-align: center; color: #666; margin-bottom: 20px; }}
        .stats-summary {{ background-color: white; padding: 15px; border-radius: 5px; margin-bottom: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: none; }}
        .stats-summary.active {{ display: block; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }}
        .stat-item {{ text-align: center; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
        .stat-label {{ font-size: 12px; color: #666; }}
        .plots-container {{ display: none; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 20px; margin-top: 20px; }}
        .plots-container.active {{ display: grid; }}
        .plot-item {{ background-color: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .plot-item h3 {{ margin-top: 0; color: #333; font-size: 14px; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
        .plot-item img {{ width: 100%; height: auto; }}
        .plot-stats {{ display: flex; justify-content: space-between; margin-top: 10px; padding-top: 10px; border-top: 1px solid #eee; font-size: 12px; color: #666; }}
        .good-correlation {{ color: #4CAF50; font-weight: bold; }}
        .poor-correlation {{ color: #f44336; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>{transformation_type} Transformation Results</h1>
    <div class="metadata">
        <p><strong>Batch:</strong> {batch_name} | <strong>Generated:</strong> {timestamp} | <strong>Transform:</strong> {transform_function}</p>
    </div>
    
    <div class="controls">
        <label for="metricSelector" style="margin-right: 10px; font-weight: bold;">Select Feature Group:</label>
        <select id="metricSelector" class="metric-selector" onchange="showMetric(this.value)">
            <option value="">-- Select a Feature Group --</option>
            {group_options}
        </select>
    </div>
    
    {group_content}
    
    <script>
        function showMetric(groupName) {{
            const allStats = document.querySelectorAll('.stats-summary');
            const allPlots = document.querySelectorAll('.plots-container');
            
            allStats.forEach(el => el.classList.remove('active'));
            allPlots.forEach(el => el.classList.remove('active'));
            
            if (groupName) {{
                const safeGroupName = groupName.replace(/[ :]/g, '_');
                const statsEl = document.getElementById('stats-' + safeGroupName);
                const plotsEl = document.getElementById('plots-' + safeGroupName);
                if (statsEl) statsEl.classList.add('active');
                if (plotsEl) plotsEl.classList.add('active');
            }}
        }}
        
        window.onload = function() {{
            const selector = document.getElementById('metricSelector');
            if (selector.options.length > 1) {{
                selector.selectedIndex = 1;
                showMetric(selector.options[1].value);
            }}
        }}
    </script>
</body>
</html>"""

# =============================================================================
# SHARED UTILITY FUNCTIONS
# =============================================================================

def clean_image_names(df):
    """Clean image names by removing .ome.tiff extension"""
    df = df.copy()
    df['Image'] = df['Image'].str.replace('.ome.tiff', '', regex=False)
    return df

def get_target_columns(df, target_feature):
    """Get columns matching the target feature patterns"""
    target_list = [f.strip() for f in target_feature.split(',')]
    target_cols = []
    
    for target in target_list:
        matching_cols = [col for col in df.columns if target in col]
        target_cols.extend(matching_cols)
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(target_cols))

def organize_columns_by_groups(target_cols):
    """Organize columns into groups based on their suffix"""
    feature_groups = {}
    
    for col in target_cols:
        if ': ' in col:
            group = col.split(': ')[-1].strip().title()
        else:
            group = 'Other'
        
        if group not in feature_groups:
            feature_groups[group] = []
        feature_groups[group].append(col)
    
    return feature_groups

def calculate_cv_metrics(df_original, df_transformed, target_cols):
    """Calculate coefficient of variation for each marker by slide"""
    cv_data = []
    
    for col in target_cols:
        for slide in df_original['Slide'].unique():
            slide_data_orig = df_original[df_original['Slide'] == slide][col]
            slide_data_trans = df_transformed[df_transformed['Slide'] == slide][col]
            
            # Calculate CV (std/mean) if mean > 0
            cv_orig = slide_data_orig.std() / slide_data_orig.mean() if slide_data_orig.mean() > 0 else np.nan
            cv_trans = slide_data_trans.std() / slide_data_trans.mean() if slide_data_trans.mean() > 0 else np.nan
            
            marker_name = col.replace('Cell: ', '').replace(': Mean', '')
            
            cv_data.append({
                'marker': marker_name,
                'slide': slide,
                'cv_original': cv_orig,
                'cv_transformed': cv_trans,
                'cv_improvement': cv_orig - cv_trans if not (np.isnan(cv_orig) or np.isnan(cv_trans)) else np.nan
            })
    
    return pd.DataFrame(cv_data)

def get_worst_performing_markers(cv_df, n_markers=5):
    """Identify markers with worst CV performance"""
    if cv_df.empty:
        return []
        
    marker_performance = cv_df.groupby('marker').agg({
        'cv_improvement': 'mean',
        'cv_transformed': 'mean'
    }).reset_index()
    
    worst_markers = marker_performance.nsmallest(n_markers, 'cv_improvement')['marker'].tolist()
    return worst_markers

def create_plot_for_column(df_original, df_transformed, col, transformation_type, param_info):
    """Create a single transformation plot for a column"""
    # Calculate correlation
    paired = pd.concat([df_original[col], df_transformed[col]], axis=1, join='inner').dropna()
    
    # Check for sufficient data for correlation
    if paired.shape[0] < 2:
        print(f"[WARNING] Skipping correlation for column '{col}': not enough valid data points (n={paired.shape[0]}). Branch: insufficient value variability.")
        correlation = float('nan')
    else:
        try:
            correlation, _ = pearsonr(paired.iloc[:, 0], paired.iloc[:, 1])
        except TypeError as e:
            print(f"\n[ERROR] TypeError while computing pearsonr for column '{col}': {e}")
            print(f"Column '{col}' original dtype: {df_original[col].dtype}")
            print(f"Column '{col}' transformed dtype: {df_transformed[col].dtype}")

            # Find and print problematic values in both columns
            orig_non_float = df_original[col][~df_original[col].apply(lambda x: isinstance(x, (int, float, float, complex)) or pd.isna(x))]
            trans_non_float = df_transformed[col][~df_transformed[col].apply(lambda x: isinstance(x, (int, float, float, complex)) or pd.isna(x))]

            if not orig_non_float.empty:
                print(f"\n[DEBUG] Non-numeric values in original column '{col}':")
                print(orig_non_float.head(10))
            if not trans_non_float.empty:
                print(f"\n[DEBUG] Non-numeric values in transformed column '{col}':")
                print(trans_non_float.head(10))

            # Show a sample of the paired data that caused the error
            print("\n[DEBUG] Sample of paired data (first 10 rows):")
            print(paired.head(10))

            # Optionally, show all unique non-numeric values
            print("\n[DEBUG] Unique non-numeric values in original column:")
            print(orig_non_float.unique())
            print("\n[DEBUG] Unique non-numeric values in transformed column:")
            print(trans_non_float.unique())

            correlation = float('nan')
            print(f"[WARNING] Skipping correlation for column '{col}' due to TypeError. Branch: type error.")
        except ValueError as e:
            print(f"[WARNING] Skipping correlation for column '{col}': {e}. Branch: insufficient value variability.")
            correlation = float('nan')
    
    # Create plot
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize'])
    plt.rcParams.update({
        'font.size': PLOT_CONFIG['font_size'],
        'font.family': 'sans-serif',
        'axes.linewidth': PLOT_CONFIG['linewidth']
    })
    
    # Scatter plot
    ax.scatter(df_original[col], df_transformed[col], 
              alpha=PLOT_CONFIG['alpha'], s=PLOT_CONFIG['point_size'], color='blue')
    
    # Add reference line
    max_orig = df_original[col].max()
    max_trans = df_transformed[col].max()
    if max_orig > 0 and max_trans > 0:
        ax.plot([0, max_orig], [0, max_trans], 'r--', alpha=0.7, linewidth=2)
    
    # Formatting
    ax.set_title(f'{transformation_type} Transform: {col}', fontweight='bold', fontsize=PLOT_CONFIG['title_size'])
    ax.set_xlabel('Original Value', fontsize=PLOT_CONFIG['font_size'])
    ax.set_ylabel('Transformed Value', fontsize=PLOT_CONFIG['font_size'])
    ax.grid(True, alpha=0.3)
    
    # Add stats box
    textstr = f'{param_info["name"]}: {param_info["value"]}\nr = {correlation if not np.isnan(correlation) else "N/A"}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Convert to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return {
        'marker_name': col,
        'img_b64': img_b64,
        'param_value': param_info['value'],
        'correlation': correlation
    }

def create_all_plots(df_original, df_transformed, target_cols, transformation_type, param_info_func):
    """Create plots for all target columns organized by groups"""
    feature_groups = organize_columns_by_groups(target_cols)
    all_plot_data = {}
    
    for group_name, group_cols in feature_groups.items():
        print(f"Processing {len(group_cols)} columns for {group_name} group...")
        
        plot_data = []
        for col in group_cols:
            param_info = param_info_func(col)
            plot = create_plot_for_column(df_original, df_transformed, col, transformation_type, param_info)
            plot_data.append(plot)
        
        all_plot_data[group_name] = {
            'plots': plot_data,
            'total': len(group_cols)
        }
    
    return all_plot_data

def generate_html_report(all_plot_data, batch_name, transformation_type, transform_function, method_specific_stats=None):
    """Generate HTML report for transformation results"""
    # Create group options
    group_options = []
    for group_name, group_data in all_plot_data.items():
        group_options.append(f'<option value="{group_name}">{group_name} ({group_data["total"]} features)</option>')
    
    # Create group content
    group_content = []
    for group_name, group_data in all_plot_data.items():
        safe_group_name = group_name.replace(' ', '_').replace(':', '_')
        
        # Stats section
        stats_items = [
            f'<div class="stat-item"><div class="stat-value">{group_data["total"]}</div><div class="stat-label">Total Features</div></div>',
            f'<div class="stat-item"><div class="stat-value">{transform_function}</div><div class="stat-label">Transform Function</div></div>'
        ]
        
        # Add method-specific stats
        if method_specific_stats and group_name in method_specific_stats:
            for stat_name, stat_value in method_specific_stats[group_name].items():
                stats_items.append(f'<div class="stat-item"><div class="stat-value">{stat_value}</div><div class="stat-label">{stat_name}</div></div>')
        
        stats_html = f'''
    <div class="stats-summary" id="stats-{safe_group_name}">
        <h2>{group_name} Transformation Summary</h2>
        <div class="stats-grid">
            {''.join(stats_items)}
        </div>
    </div>'''
        
        # Plots section
        plot_items = []
        for plot in group_data['plots']:
            correlation_class = 'good-correlation' if plot['correlation'] > 0.8 else 'poor-correlation' if plot['correlation'] < 0.5 else ''
            
            plot_items.append(f'''
        <div class="plot-item">
            <h3>{plot['marker_name']}</h3>
            <img src="data:image/png;base64,{plot['img_b64']}" alt="{plot['marker_name']} transformation">
            <div class="plot-stats">
                <span>Param: {plot['param_value']}</span>
                <span class="{correlation_class}">Correlation: {plot['correlation']:.3f}</span>
            </div>
        </div>''')
        
        plots_html = f'''
    <div class="plots-container" id="plots-{safe_group_name}">
        {''.join(plot_items)}
    </div>'''
        
        group_content.append(stats_html + plots_html)
    
    # Fill template
    html_content = HTML_TEMPLATE.format(
        transformation_type=transformation_type,
        batch_name=batch_name,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        transform_function=transform_function,
        group_options='\n'.join(group_options),
        group_content='\n'.join(group_content)
    )
    
    # Save HTML
    method_name = transformation_type.lower().replace('-', '').replace(' ', '')
    html_filename = f'{method_name}_all_plots_{batch_name}.html'
    with open(html_filename, 'w') as f:
        f.write(html_content)
    
    print(f"Combined HTML report generated: {html_filename}")
    return html_filename

def save_results(df_transformed, results, batch_name, method):
    """Save transformed data and results"""
    # Save transformed data
    output_file = f"{method}_transformed_{batch_name}.tsv"
    df_transformed.to_csv(output_file, sep="\t", index=False)
    
    # Save results JSON
    if results is not None:
        json_file = f'{method}_results_{batch_name}.json'
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)


# =============================================================================
# TRANSFORMATION METHODS
# =============================================================================

def apply_log_transform(df, batch_name, target_feature):
    """Apply log transformation"""
    target_cols = get_target_columns(df, target_feature)
    
    # Apply log transformation
    df_transformed = df.copy(deep=True)
    numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns
    numeric_cols = [x for x in numeric_cols if "Centroid" not in x]
    df_transformed[numeric_cols] = df_transformed[numeric_cols].apply(lambda x: np.log(x + 1))
    
    # Create plots
    param_info_func = lambda col: {'name': 'Transform', 'value': 'log(x+1)'}
    all_plot_data = create_all_plots(df, df_transformed, target_cols, 'Log', param_info_func)
    
    # Generate HTML
    html_file = generate_html_report(all_plot_data, batch_name, 'Log', 'log(x + 1)')
    
    # Calculate metrics
    cv_df = calculate_cv_metrics(df, df_transformed, target_cols)
    worst_markers = get_worst_performing_markers(cv_df)
    
    results = {
        'transformation_type': 'log',
        'batch_name': batch_name,
        'total_markers': len(target_cols),
        'total_slides': df['Slide'].nunique(),
        'total_cells': len(df),
        'worst_performing_markers': worst_markers,
        'cv_metrics': {
            'mean_cv_improvement': cv_df['cv_improvement'].mean(),
            'median_cv_improvement': cv_df['cv_improvement'].median(),
            'markers_improved': (cv_df['cv_improvement'] > 0).sum(),
            'markers_worsened': (cv_df['cv_improvement'] < 0).sum()
        },
        'html_report': html_file,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return df_transformed, results

def apply_quantile_transform(df, batch_name, target_feature, quantile_split):
    """Apply quantile transformation"""
    target_cols = get_target_columns(df, target_feature)
    
    # Apply quantile transformation
    scaler = QuantileTransformer(n_quantiles=quantile_split, random_state=0)
    imgMets = df.filter(regex='(Min|Max|Median|Mean|Std*|Variance|Area)', axis=1)
    df_norm = pd.DataFrame(scaler.fit_transform(imgMets), columns=imgMets.columns)
    df_norm = df_norm.fillna(0)
    df_a = df[df.columns.difference(imgMets.columns)]
    df_transformed = pd.concat([df_a.reset_index(drop=True), df_norm], axis=1)
        
    # Create plots
    param_info_func = lambda col: {'name': 'Quantiles', 'value': str(quantile_split)}
    all_plot_data = create_all_plots(df, df_transformed, target_cols, 'Quantile', param_info_func)
    
    # Generate HTML
    html_file = generate_html_report(all_plot_data, batch_name, 'Quantile', f'Quantile (n={quantile_split})')
    
    # Calculate metrics
    cv_df = calculate_cv_metrics(df, df_transformed, target_cols)
    worst_markers = get_worst_performing_markers(cv_df)
    
    results = {
        'transformation_type': 'quantile',
        'batch_name': batch_name,
        'quantile_splits': quantile_split,
        'total_markers': len(target_cols),
        'total_slides': df['Slide'].nunique(),
        'total_cells': len(df),
        'worst_performing_markers': worst_markers,
        'cv_metrics': {
            'mean_cv_improvement': cv_df['cv_improvement'].mean(),
            'median_cv_improvement': cv_df['cv_improvement'].median(),
            'markers_improved': (cv_df['cv_improvement'] > 0).sum(),
            'markers_worsened': (cv_df['cv_improvement'] < 0).sum()
        },
        'html_report': html_file,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return df_transformed, results

def apply_minmax_transform(df, batch_name, target_feature):
    """Apply MinMax transformation"""
    target_cols = get_target_columns(df, target_feature)
    
    # Apply MinMax transformation
    scaler = MinMaxScaler(feature_range=(-2, 2))
    imgMets = df.filter(regex='(Min|Max|Median|Mean|StdDev)', axis=1)
    df_norm = pd.DataFrame(scaler.fit_transform(imgMets), columns=imgMets.columns)
    df_norm = df_norm.fillna(0)
    df_a = df[df.columns.difference(imgMets.columns)]
    df_transformed = pd.concat([df_a.reset_index(drop=True), df_norm], axis=1)
    
    # Create plots
    param_info_func = lambda col: {'name': 'Range', 'value': '(-2, 2)'}
    all_plot_data = create_all_plots(df, df_transformed, target_cols, 'MinMax', param_info_func)
    
    # Generate HTML
    html_file = generate_html_report(all_plot_data, batch_name, 'MinMax', 'MinMax (-2, 2)')
    
    # Calculate metrics
    cv_df = calculate_cv_metrics(df, df_transformed, target_cols)
    worst_markers = get_worst_performing_markers(cv_df)
    
    results = {
        'transformation_type': 'minmax',
        'batch_name': batch_name,
        'total_markers': len(target_cols),
        'total_slides': df['Slide'].nunique(),
        'total_cells': len(df),
        'worst_performing_markers': worst_markers,
        'cv_metrics': {
            'mean_cv_improvement': cv_df['cv_improvement'].mean(),
            'median_cv_improvement': cv_df['cv_improvement'].median(),
            'markers_improved': (cv_df['cv_improvement'] > 0).sum(),
            'markers_worsened': (cv_df['cv_improvement'] < 0).sum()
        },
        'html_report': html_file,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return df_transformed, results

def apply_boxcox_transform(df, batch_name, target_feature):
    """Apply Box-Cox transformation"""
    target_cols = get_target_columns(df, target_feature)
    
    # Apply Box-Cox transformation
    metrics = []
    df_transformed = df.copy()
    lambda_values = {}
    stat_cols = list(df_transformed.filter(regex='(Min|Max|Median|Mean|Std*|Variance|Area)'))

    for col in stat_cols:
        # Coerce to numeric, skip column if any non-numeric values remain
        coerced = pd.to_numeric(df_transformed[col], errors='coerce')
        if coerced.isna().all():
            print(f"[WARNING] Skipping column '{col}' (all values non-numeric or NaN).")
            df_transformed[col] = np.nan
            lambda_val = 'SkippedEmpty'
            pre_mean = np.nan
            metrics.append([col, pre_mean, lambda_val, np.nan, np.nan, np.nan])
            continue
        if coerced.isna().sum() > 0:
            print(f"[WARNING] Column '{col}' contains some non-numeric values. These will be set to 0 for Box-Cox.")
            coerced = coerced.fillna(0)
        df_transformed[col] = coerced

        col_sum = df_transformed[col].sum()
        if col_sum == 0:
            df_transformed[col] = np.nan
            lambda_val = 'SkippedEmpty'
            pre_mean = np.nan
        else:
            pre_mean = df_transformed[col].mean()
            try:
                transformed_values, lambda_val = boxcox(df_transformed[col].add(1).values)
                df_transformed[col] = transformed_values
                lambda_values[col] = f"{lambda_val:.3f}"
            except Exception:
                df_transformed[col] = 0
                lambda_val = 'Failed'
                lambda_values[col] = 'Failed'

        metrics.append([col, pre_mean, lambda_val, df_transformed[col].mean(), 
                       df_transformed[col].min(), df_transformed[col].max()])
    
    # Save Box-Cox metrics
    boxcox_metrics = pd.DataFrame(metrics, columns=['Feature', 'Pre_Mean', 'Lambda', 'Post_Mean', 'Post_Min', 'Post_Max'])
    boxcox_metrics.to_csv(f"BoxCoxRecord_{batch_name}.csv", index=False)
    
    # Create plots with lambda values
    param_info_func = lambda col: {'name': 'Lambda', 'value': lambda_values.get(col, 'N/A')}
    all_plot_data = create_all_plots(df, df_transformed, target_cols, 'Box-Cox', param_info_func)
    
    # Add method-specific stats
    method_stats = {}
    for group_name, group_data in all_plot_data.items():
        group_cols = [plot['marker_name'] for plot in group_data['plots']]
        group_metrics = boxcox_metrics[boxcox_metrics['Feature'].isin(group_cols)]
        successful = len(group_metrics[~group_metrics['Lambda'].isin(['Failed', 'SkippedEmpty'])])
        failed = len(group_metrics[group_metrics['Lambda'].isin(['Failed', 'SkippedEmpty'])])
        method_stats[group_name] = {'Successful Transforms': successful, 'Failed Transforms': failed}
    
    # Generate HTML
    html_file = generate_html_report(all_plot_data, batch_name, 'Box-Cox', 'Box-Cox (λ varies)', method_stats)
    
    # Calculate metrics
    cv_df = calculate_cv_metrics(df, df_transformed, target_cols)
    worst_markers = get_worst_performing_markers(cv_df)
    
    results = {
        'transformation_type': 'boxcox',
        'batch_name': batch_name,
        'total_markers': len(target_cols),
        'total_slides': df['Slide'].nunique(),
        'total_cells': len(df),
        'worst_performing_markers': worst_markers,
        'boxcox_metrics': {
            'total_features': len(boxcox_metrics),
            'successful_transforms': len(boxcox_metrics[~boxcox_metrics['Lambda'].isin(['Failed', 'SkippedEmpty'])]),
            'failed_transforms': len(boxcox_metrics[boxcox_metrics['Lambda'].isin(['Failed', 'SkippedEmpty'])]),
            'mean_lambda': boxcox_metrics[~boxcox_metrics['Lambda'].isin(['Failed', 'SkippedEmpty'])]['Lambda'].astype(float).mean(),
            'median_lambda': boxcox_metrics[~boxcox_metrics['Lambda'].isin(['Failed', 'SkippedEmpty'])]['Lambda'].astype(float).median()
        },
        'cv_metrics': {
            'mean_cv_improvement': cv_df['cv_improvement'].mean(),
            'median_cv_improvement': cv_df['cv_improvement'].median(),
            'markers_improved': (cv_df['cv_improvement'] > 0).sum(),
            'markers_worsened': (cv_df['cv_improvement'] < 0).sum()
        },
        'html_report': html_file,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return df_transformed, results

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function with unified argument parser and method dispatcher"""
    parser = argparse.ArgumentParser(description="Apply normalization transformations to quantification data.")
    
    # Common arguments for all methods
    parser.add_argument('--pickleTable', required=True, help='Input pickle file')
    parser.add_argument('--batchID', required=True, help='Batch ID for output file naming')
    parser.add_argument('--method', required=True, choices=['log', 'quantile', 'minmax', 'boxcox', 'none'],
                       help='Transformation method to apply')
    parser.add_argument('--target-feature', dest='targetFeature', default='Cell: Mean',
                       help='Comma-separated list of column name patterns to target (required for all methods)')
    
    # Method-specific arguments
    parser.add_argument('--quantileSplit', type=int, 
                       help='Number of quantiles for QuantileTransformer (required for quantile method)')
    
    args = parser.parse_args()
    
    # Validate method-specific arguments
    if args.method == 'quantile' and args.quantileSplit is None:
        parser.error("--quantileSplit is required when using quantile method")
    
    # Load data
    print(f"Loading data from {args.pickleTable}...")
    df = pd.read_pickle(args.pickleTable)
    df = clean_image_names(df)
    
    # Convert statistical columns to numeric if they are not
    sts = ["Min", "Max", "Median", "Mean", "Std.Dev.", "Variance"]

    for col in df.columns:
        if any(s in col for s in sts):
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"[WARNING] Column '{col}' should be numeric but is {df[col].dtype}. Converting to numeric.")
                df[col] = pd.to_numeric(df[col], errors='coerce')
                n_nans = df[col].isna().sum()
                if n_nans > 0:
                    print(f"[INFO] Filled {n_nans} NaN values in '{col}' with 0 after conversion.")
                    df[col] = df[col].fillna(0)
    
    # Apply transformation
    print(f"Applying {args.method} transformation...")
    
    if args.method == 'log':
        df_transformed, results = apply_log_transform(df, args.batchID, args.targetFeature)
    elif args.method == 'quantile':
        df_transformed, results = apply_quantile_transform(df, args.batchID, args.targetFeature, args.quantileSplit)
    elif args.method == 'minmax':
        df_transformed, results = apply_minmax_transform(df, args.batchID, args.targetFeature)
    elif args.method == 'boxcox':
        df_transformed, results = apply_boxcox_transform(df, args.batchID, args.targetFeature)
    elif args.method == 'none':
        df_transformed = df
        results = "{}"   
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    # Save results
    save_results(df_transformed, results, args.batchID, args.method)
if __name__ == "__main__":
    main()