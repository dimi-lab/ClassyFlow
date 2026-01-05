#!/usr/bin/env python3

import sys, os
import argparse
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema
from jinja2 import Template
import base64
from io import BytesIO

# Static field to be applied as a fixed heuristic for marker columns
markerColumnBase = 'Mean'


def parse_args():
    parser = argparse.ArgumentParser(description="GMM gating for normalized tables with HTML report.")
    parser.add_argument('--input', required=True, help='Input normalized TSV file')
    parser.add_argument('--output', required=True, help='Output TSV file (gated)')
    parser.add_argument('--html_report', default='gmm_gating_report.html', help='Output HTML report file')
    parser.add_argument('--target-feature', dest='targetFeature', default='Cell: Mean',
                       help='Comma-separated list of column name patterns to process')
    parser.add_argument('--batch-name', dest='batchName', default='Unknown', help='Batch name for the report')
    return parser.parse_args()


def plot_histogram_and_gmm(col, values, threshold, gmm):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(values, bins=50, kde=True, ax=ax, color='skyblue', stat='density')
    x = np.linspace(values.min(), values.max(), 1000)
    logprob = gmm.score_samples(x.reshape(-1, 1))
    responsibilities = gmm.predict_proba(x.reshape(-1, 1))
    pdf = np.exp(logprob)
    ax.plot(x, pdf, '-r', label='GMM fit')
    ax.axvline(threshold, color='orange', linestyle='--', label=f'Threshold: {threshold:.2f}')
    ax.set_title(f'{col} - GMM Gating')
    ax.legend()
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_b64


def plot_delta_scatter(pre, post, col):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pre, post, alpha=0.3, s=1)
    ax.set_xlabel('Pre-Gating')
    ax.set_ylabel('Post-Gating')
    ax.set_title(f'{col}: Pre vs Post Gating')
    
    # Add diagonal line
    max_val = max(pre.max(), post.max())
    min_val = min(pre.min(), post.min())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Identity line')
    ax.legend()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_b64


def get_target_columns(df, target_features):
    if target_features:
        target_list = [f.strip() for f in target_features.split(',')]
        
        # Find all columns that match any of the target features
        target_cols = []
        for target in target_list:
            matching_cols = [col for col in df.columns if target in col]
            target_cols.extend(matching_cols)
    
        # Remove duplicates while preserving order
        target_cols = list(dict.fromkeys(target_cols))
        return target_cols
    else:
        # Default behavior: process all Mean columns
        return [c for c in df.columns if markerColumnBase in c]


def main():
    args = parse_args()
    df = pd.read_csv(args.input, sep='\t')
    df_gated = df.copy()
    gating_results = []
    plots = []
    delta_plots = []

    target_cols = get_target_columns(df, args.targetFeature)
    print(f"Found {len(target_cols)} columns for GMM gating")
    
    for col in target_cols:
        values = df[col].values.astype(float)
        # Skip columns with all NaN or fewer than 2 valid values
        valid_values = values[~np.isnan(values)]
        if valid_values.size < 2:
            print(f"[WARNING] Skipping column '{col}' for GMM gating: not enough valid (non-NaN) values.")
            continue
            
        X = valid_values.reshape(-1, 1)
        # Fit 1- and 2-component GMMs
        gmm1 = GaussianMixture(n_components=1, random_state=0).fit(X)
        gmm2 = GaussianMixture(n_components=2, random_state=0).fit(X)
        bic1 = gmm1.bic(X)
        bic2 = gmm2.bic(X)
        best_gmm = gmm2 if bic2 < bic1 else gmm1
        means = best_gmm.means_.flatten()
        stds = np.sqrt(best_gmm.covariances_).flatten()
        sorted_idx = np.argsort(means)
        bg_mean = means[sorted_idx[0]]
        bg_std = stds[sorted_idx[0]]
        threshold = bg_mean + 2 * bg_std

        if len(np.unique(values)) <= 1:
            print(f"[WARNING] Skipping KDE for feature '{col}' due to insufficient unique values.", flush=True)
            continue
        # KDE fallback
        kde = gaussian_kde(values)
        x_grid = np.linspace(values.min(), values.max(), 1000)
        pdf = kde(x_grid)
        local_minima = argrelextrema(pdf, np.less)[0]
        if len(local_minima) > 0:
            kde_thresh = x_grid[local_minima[0]]
            threshold = min(threshold, kde_thresh)
            
        # Apply gating: set values below threshold to threshold
        pre_vals = df_gated[col].copy()
        df_gated[col] = np.where(df_gated[col] < threshold, threshold, df_gated[col])
        post_vals = df_gated[col]
        delta = df_gated[col] - pre_vals
        
        # Propagate delta to all columns with same prefix
        prefix = col.split(':')[0] if ':' in col else col
        for other_col in df.columns:
            if other_col == col:
                continue
            if other_col.startswith(prefix):
                df_gated[other_col] = df_gated[other_col] + delta

        # Propagate threshold using z-score standardization to all columns with the same prefix (best practices)
        col_split = col.split(":")
        if len(col_split) >= 3:
            prefix = ":".join(col_split[:2])
        else:
            prefix = col_split[0]
        # Get all columns with this prefix, excluding col itself
        related_cols = [c for c in df.columns if c != col and c.startswith(prefix)]
        for other_col in related_cols:
            vals_B = df_gated[other_col].values.astype(float)
            # Standardize using Field A (col) background mean/std
            z_vals_B = (vals_B - bg_mean) / bg_std if bg_std > 0 else vals_B * 0
            threshold_z = 2
            # Apply threshold in z-score space
            mask = z_vals_B < threshold_z
            # Set values below threshold to the threshold in original scale
            df_gated[other_col] = np.where(mask, bg_mean + threshold_z * bg_std, vals_B)

        # Save results for report
        gating_results.append({
            'column': col,
            'threshold': threshold,
            'gmm_means': means.tolist(),
            'gmm_stds': stds.tolist(),
            'pre_mean': float(pre_vals.mean()),
            'post_mean': float(post_vals.mean()),
            'delta_mean': float(delta.mean()),
            'n_components': len(means),
            'cells_gated': int((pre_vals < threshold).sum()),
            'percent_gated': float((pre_vals < threshold).mean() * 100)
        })
        plots.append({'column': col, 'img': plot_histogram_and_gmm(col, values, threshold, best_gmm)})
        delta_plots.append({'column': col, 'img': plot_delta_scatter(pre_vals, post_vals, col)})

    # Save gated output
    df_gated.to_csv(args.output, sep='\t', index=False)

    # Save JSON results for report integration
    json_results = {
        'batch_name': args.batchName,
        'total_features': len(gating_results),
        'avg_percent_gated': float(np.mean([r['percent_gated'] for r in gating_results])) if gating_results else 0,
        'avg_cells_gated': int(np.mean([r['cells_gated'] for r in gating_results])) if gating_results else 0,
        'per_marker': [
            {
                'marker': r['column'],
                'threshold': r['threshold'],
                'percent_gated': r['percent_gated'],
                'cells_gated': r['cells_gated']
            }
            for r in gating_results
        ]
    }
    
    json_output = f"gmm_results_{args.batchName}.json"
    with open(json_output, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"JSON results saved to {json_output}")

    # Generate HTML report
    html_template = Template('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GMM Gating Report - {{ batch_name }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        h1 { color: #333; text-align: center; margin-bottom: 10px; }
        h2 { color: #333; margin-top: 30px; border-bottom: 2px solid #667eea; padding-bottom: 10px; }
        .metadata { text-align: center; color: #666; margin-bottom: 20px; }
        .summary-stats { background-color: white; padding: 20px; border-radius: 5px; margin-bottom: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-top: 20px; }
        .stat-item { text-align: center; background-color: #f8f9fa; padding: 15px; border-radius: 5px; }
        .stat-value { font-size: 24px; font-weight: bold; color: #667eea; }
        .stat-label { font-size: 12px; color: #666; margin-top: 5px; }
        table { width: 100%; border-collapse: collapse; background-color: white; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        th { background-color: #f8f9fa; color: #333; padding: 12px; text-align: left; font-weight: 600; border-bottom: 2px solid #dee2e6; }
        td { padding: 10px 12px; border-bottom: 1px solid #dee2e6; }
        tr:hover { background-color: #f8f9fa; }
        .plot-section { background-color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .plot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; margin-top: 20px; }
        .plot-item { background-color: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .plot-item h3 { margin-top: 0; color: #333; font-size: 14px; border-bottom: 1px solid #eee; padding-bottom: 10px; }
        .plot-item img { width: 100%; height: auto; }
        .good-threshold { color: #28a745; font-weight: bold; }
        .moderate-threshold { color: #ffc107; font-weight: bold; }
        .poor-threshold { color: #dc3545; font-weight: bold; }
    </style>
</head>
<body>
    <h1>GMM Gating Report</h1>
    <div class="metadata">
        <p><strong>Batch:</strong> {{ batch_name }} | <strong>Processed:</strong> {{ timestamp }} | <strong>Total Features:</strong> {{ total_features }}</p>
    </div>
    
    <div class="summary-stats">
        <h2>Summary Statistics</h2>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-value">{{ total_features }}</div>
                <div class="stat-label">Features Gated</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{{ avg_cells_gated }}</div>
                <div class="stat-label">Avg Cells Gated</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{{ avg_percent_gated }}%</div>
                <div class="stat-label">Avg Percent Gated</div>
            </div>
        </div>
    </div>
    
    <h2>Gating Results Table</h2>
    <table>
        <thead>
            <tr>
                <th>Column</th>
                <th>Threshold</th>
                <th>Components</th>
                <th>Cells Gated</th>
                <th>% Gated</th>
                <th>Pre Mean</th>
                <th>Post Mean</th>
                <th>Delta Mean</th>
            </tr>
        </thead>
        <tbody>
            {% for row in gating_results %}
            <tr>
                <td><strong>{{ row.column }}</strong></td>
                <td class="{% if row.percent_gated < 5 %}good-threshold{% elif row.percent_gated < 15 %}moderate-threshold{% else %}poor-threshold{% endif %}">
                    {{ '%.3f' % row.threshold }}
                </td>
                <td>{{ row.n_components }}</td>
                <td>{{ row.cells_gated }}</td>
                <td>{{ '%.1f' % row.percent_gated }}%</td>
                <td>{{ '%.3f' % row.pre_mean }}</td>
                <td>{{ '%.3f' % row.post_mean }}</td>
                <td>{{ '%.3f' % row.delta_mean }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    
    <div class="plot-section">
        <h2>GMM Gating Distributions</h2>
        <div class="plot-grid">
            {% for p in plots %}
            <div class="plot-item">
                <h3>{{ p.column }}</h3>
                <img src="data:image/png;base64,{{ p.img }}"/>
            </div>
            {% endfor %}
        </div>
    </div>
    
    <div class="plot-section">
        <h2>Pre vs Post Gating Comparison</h2>
        <div class="plot-grid">
            {% for p in delta_plots %}
            <div class="plot-item">
                <h3>{{ p.column }}</h3>
                <img src="data:image/png;base64,{{ p.img }}"/>
            </div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
    ''')
    
    total_features = len(gating_results)
    avg_cells_gated = int(np.mean([r['cells_gated'] for r in gating_results])) if gating_results else 0
    avg_percent_gated = round(np.mean([r['percent_gated'] for r in gating_results]), 1) if gating_results else 0
    
    # Write HTML report
    with open(args.html_report, 'w') as f:
        f.write(html_template.render(
            batch_name=args.batchName,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_features=total_features,
            avg_cells_gated=avg_cells_gated,
            avg_percent_gated=avg_percent_gated,
            gating_results=gating_results,
            plots=plots,
            delta_plots=delta_plots
        ))
    
    print(f"GMM gating complete. Output saved to {args.output}")
    print(f"HTML report saved to {args.html_report}")


if __name__ == "__main__":
    main()