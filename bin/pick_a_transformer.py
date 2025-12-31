#!/usr/bin/env python3

import sys, os, time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from scipy.stats import pearsonr, boxcox, skew
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from pathlib import Path
import base64
from io import BytesIO

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
    df = df.copy()
    df['Image'] = df['Image'].str.replace('.ome.tiff', '', regex=False)
    return df


def get_target_columns(df, target_feature):
    target_list = [f.strip() for f in target_feature.split(',')]
    target_cols = []
    
    for target in target_list:
        matching_cols = [col for col in df.columns if target in col]
        target_cols.extend(matching_cols)
    
    return list(dict.fromkeys(target_cols))


def organize_columns_by_groups(target_cols):
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


def calculate_skewness_metrics(df_original, df_transformed, target_cols):
    skewness_data = []
    
    for col in target_cols:
        skew_before = skew(df_original[col].dropna())
        skew_after = skew(df_transformed[col].dropna())
        
        # Reduction ratio >1 means improvement (less skewed)
        if abs(skew_after) > 0.01:
            reduction_ratio = abs(skew_before) / abs(skew_after)
        else:
            reduction_ratio = abs(skew_before) / 0.01 if abs(skew_before) > 0.01 else 1.0
        
        skewness_data.append({
            'marker': col,
            'skewness_before': skew_before,
            'skewness_after': skew_after,
            'reduction_ratio': reduction_ratio
        })
    
    return pd.DataFrame(skewness_data)


def calculate_outlier_metrics(df_original, df_transformed, target_cols, threshold=3.0):
    outlier_data = []
    
    for col in target_cols:
        # Before transformation
        orig_values = df_original[col].dropna()
        orig_mean, orig_std = orig_values.mean(), orig_values.std()
        if orig_std > 0:
            outliers_before = ((orig_values - orig_mean).abs() > threshold * orig_std).sum()
        else:
            outliers_before = 0
        
        # After transformation
        trans_values = df_transformed[col].dropna()
        trans_mean, trans_std = trans_values.mean(), trans_values.std()
        if trans_std > 0:
            outliers_after = ((trans_values - trans_mean).abs() > threshold * trans_std).sum()
        else:
            outliers_after = 0
        
        # Reduction ratio >1 means improvement (fewer outliers)
        if outliers_after > 0:
            reduction_ratio = outliers_before / outliers_after
        else:
            reduction_ratio = outliers_before if outliers_before > 0 else 1.0
        
        outlier_data.append({
            'marker': col,
            'outliers_before': int(outliers_before),
            'outliers_after': int(outliers_after),
            'reduction_ratio': reduction_ratio
        })
    
    return pd.DataFrame(outlier_data)


def create_plot_for_column(df_original, df_transformed, col, transformation_type, param_info):
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize'])
    plt.rcParams.update({
        'font.size': PLOT_CONFIG['font_size'],
        'font.family': 'sans-serif',
        'axes.linewidth': PLOT_CONFIG['linewidth']
    })
    
    ax.scatter(df_original[col], df_transformed[col], 
              alpha=PLOT_CONFIG['alpha'], s=PLOT_CONFIG['point_size'], color='blue')
    
    max_orig = df_original[col].max()
    max_trans = df_transformed[col].max()
    if max_orig > 0 and max_trans > 0:
        ax.plot([0, max_orig], [0, max_trans], 'r--', alpha=0.7, linewidth=2)
    
    ax.set_title(f'{transformation_type} Transform: {col}', fontweight='bold', fontsize=PLOT_CONFIG['title_size'])
    ax.set_xlabel('Original Value', fontsize=PLOT_CONFIG['font_size'])
    ax.set_ylabel('Transformed Value', fontsize=PLOT_CONFIG['font_size'])
    ax.grid(True, alpha=0.3)
    
    textstr = f'{param_info["name"]}: {param_info["value"]}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return {
        'marker_name': col,
        'img_b64': img_b64,
        'param_value': param_info['value']
    }


def create_all_plots(df_original, df_transformed, target_cols, transformation_type, param_info_func):
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
    group_options = []
    for group_name, group_data in all_plot_data.items():
        group_options.append(f'<option value="{group_name}">{group_name} ({group_data["total"]} features)</option>')
    
    group_content = []
    for group_name, group_data in all_plot_data.items():
        safe_group_name = group_name.replace(' ', '_').replace(':', '_')
        
        stats_items = [
            f'<div class="stat-item"><div class="stat-value">{group_data["total"]}</div><div class="stat-label">Total Features</div></div>',
            f'<div class="stat-item"><div class="stat-value">{transform_function}</div><div class="stat-label">Transform Function</div></div>'
        ]
        
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
        
        plot_items = []
        for plot in group_data['plots']:
            plot_items.append(f'''
        <div class="plot-item">
            <h3>{plot['marker_name']}</h3>
            <img src="data:image/png;base64,{plot['img_b64']}" alt="{plot['marker_name']} transformation">
            <div class="plot-stats">
                <span>Param: {plot['param_value']}</span>
            </div>
        </div>''')
        
        plots_html = f'''
    <div class="plots-container" id="plots-{safe_group_name}">
        {''.join(plot_items)}
    </div>'''
        
        group_content.append(stats_html + plots_html)
    
    html_content = HTML_TEMPLATE.format(
        transformation_type=transformation_type,
        batch_name=batch_name,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        transform_function=transform_function,
        group_options='\n'.join(group_options),
        group_content='\n'.join(group_content)
    )
    
    method_name = transformation_type.lower().replace('-', '').replace(' ', '')
    html_filename = f'{method_name}_all_plots_{batch_name}.html'
    with open(html_filename, 'w') as f:
        f.write(html_content)
    
    print(f"Combined HTML report generated: {html_filename}")
    return html_filename


def build_results(transformation_type, batch_name, df_original, df_transformed, target_cols, html_file, extra_data=None):
    skewness_df = calculate_skewness_metrics(df_original, df_transformed, target_cols)
    outlier_df = calculate_outlier_metrics(df_original, df_transformed, target_cols)
    
    # Build per-marker skewness data
    skewness_per_marker = [
        {
            'marker': row['marker'],
            'before': float(row['skewness_before']),
            'after': float(row['skewness_after']),
            'ratio': float(row['reduction_ratio'])
        }
        for _, row in skewness_df.iterrows()
    ]
    
    # Build per-marker outlier data
    outlier_per_marker = [
        {
            'marker': row['marker'],
            'before': int(row['outliers_before']),
            'after': int(row['outliers_after']),
            'ratio': float(row['reduction_ratio'])
        }
        for _, row in outlier_df.iterrows()
    ]
    
    results = {
        'transformation_type': transformation_type,
        'batch_name': batch_name,
        'total_markers': len(target_cols),
        'total_slides': df_original['Image'].nunique(),
        'total_cells': len(df_original),
        'skewness_metrics': {
            'mean_reduction_ratio': float(skewness_df['reduction_ratio'].mean()),
            'median_reduction_ratio': float(skewness_df['reduction_ratio'].median()),
            'mean_skewness_before': float(skewness_df['skewness_before'].mean()),
            'mean_skewness_after': float(skewness_df['skewness_after'].mean()),
            'markers_improved': int((skewness_df['reduction_ratio'] > 1).sum()),
            'markers_worsened': int((skewness_df['reduction_ratio'] < 1).sum()),
            'per_marker': skewness_per_marker
        },
        'outlier_metrics': {
            'mean_reduction_ratio': float(outlier_df['reduction_ratio'].mean()),
            'total_outliers_before': int(outlier_df['outliers_before'].sum()),
            'total_outliers_after': int(outlier_df['outliers_after'].sum()),
            'markers_improved': int((outlier_df['reduction_ratio'] > 1).sum()),
            'markers_worsened': int((outlier_df['reduction_ratio'] < 1).sum()),
            'per_marker': outlier_per_marker
        },
        'html_report': html_file,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if extra_data:
        results.update(extra_data)
    
    return results


def save_results(df_transformed, results, batch_name, method):
    output_file = f"{method}_transformed_{batch_name}.tsv"
    df_transformed.to_csv(output_file, sep="\t", index=False)
    
    if results is not None:
        json_file = f'{method}_results_{batch_name}.json'
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)


# =============================================================================
# TRANSFORMATION METHODS
# =============================================================================

def apply_log_transform(df, batch_name, target_feature):
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
    
    results = build_results('log', batch_name, df, df_transformed, target_cols, html_file)
    return df_transformed, results


def apply_quantile_transform(df, batch_name, target_feature, quantile_split):
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
    
    extra_data = {'quantile_splits': quantile_split}
    results = build_results('quantile', batch_name, df, df_transformed, target_cols, html_file, extra_data)
    return df_transformed, results


def apply_minmax_transform(df, batch_name, target_feature):
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
    
    results = build_results('minmax', batch_name, df, df_transformed, target_cols, html_file)
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
    
    extra_data = {
        'boxcox_metrics': {
            'total_features': len(boxcox_metrics),
            'successful_transforms': len(boxcox_metrics[~boxcox_metrics['Lambda'].isin(['Failed', 'SkippedEmpty'])]),
            'failed_transforms': len(boxcox_metrics[boxcox_metrics['Lambda'].isin(['Failed', 'SkippedEmpty'])]),
            'mean_lambda': float(boxcox_metrics[~boxcox_metrics['Lambda'].isin(['Failed', 'SkippedEmpty'])]['Lambda'].astype(float).mean()),
            'median_lambda': float(boxcox_metrics[~boxcox_metrics['Lambda'].isin(['Failed', 'SkippedEmpty'])]['Lambda'].astype(float).median())
        }
    }
    results = build_results('boxcox', batch_name, df, df_transformed, target_cols, html_file, extra_data)
    return df_transformed, results


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Apply normalization transformations to quantification data.")
    parser.add_argument('--pickleTable', required=True, help='Input pickle file')
    parser.add_argument('--batchID', required=True, help='Batch ID for output file naming')
    parser.add_argument('--method', required=True, choices=['log', 'quantile', 'minmax', 'boxcox', 'none'],
                       help='Transformation method to apply')
    parser.add_argument('--target-feature', dest='targetFeature', default='Cell: Mean',
                       help='Comma-separated list of column name patterns to target')
    parser.add_argument('--quantileSplit', type=int, 
                       help='Number of quantiles for QuantileTransformer (required for quantile method)')
    
    args = parser.parse_args()
    
    if args.method == 'quantile' and args.quantileSplit is None:
        parser.error("--quantileSplit is required when using quantile method")
    
    print(f"Loading data from {args.pickleTable}...")
    df = pd.read_pickle(args.pickleTable)
    df = clean_image_names(df)
    
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
        results = None
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    save_results(df_transformed, results, args.batchID, args.method)


if __name__ == "__main__":
    main()