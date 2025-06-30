#!/usr/bin/env python3

import sys, os
import argparse
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
    return parser.parse_args()


def plot_histogram_and_gmm(col, values, threshold, gmm, outdir, prefix):
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
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_b64


def plot_delta_scatter(pre, post, col, outdir):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pre, post, alpha=0.3)
    ax.set_xlabel('Pre-Gating')
    ax.set_ylabel('Post-Gating')
    ax.set_title(f'{col}: Pre vs Post Gating')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_b64


def main():
    args = parse_args()
    df = pd.read_csv(args.input, sep='\t')
    df_gated = df.copy()
    gating_results = []
    plots = []
    delta_plots = []
    # Remove reference to args.html_report, use static name
    outdir = '.'

    # Find all mean/median columns
    mm_cols = [c for c in df.columns if (markerColumnBase in c)]
    print(f"Found {len(mm_cols)} columns for GMM gating: {mm_cols}")
    prefix_map = {}
    for c in mm_cols:
        prefix = c.split(':')[0] if ':' in c else c
        prefix_map.setdefault(prefix, []).append(c)

    for col in mm_cols:
        values = df[col].values.astype(float)
        X = values.reshape(-1, 1)
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
        #print(f"Column: {col}, GMM Means: {means}, BIC1: {bic1}, BIC2: {bic2}, Threshold: {threshold:.3f}")

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
        })
        plots.append({'column': col, 'img': plot_histogram_and_gmm(col, values, threshold, best_gmm, outdir, prefix)})
        delta_plots.append({'column': col, 'img': plot_delta_scatter(pre_vals, post_vals, col, outdir)})

    # Save output
    df_gated.to_csv(args.output, sep='\t', index=False)

    # HTML report
    html_template = Template('''
    <html><head><title>GMM Gating Report</title></head><body>
    <h1>GMM Gating Report</h1>
    <h2>Summary Table</h2>
    <table border="1" cellpadding="4">
    <tr><th>Column</th><th>Threshold</th><th>GMM Means</th><th>GMM Stds</th><th>Pre Mean</th><th>Post Mean</th><th>Delta Mean</th></tr>
    {% for row in gating_results %}
    <tr><td>{{row.column}}</td><td>{{'%.3f' % row.threshold}}</td><td>{{row.gmm_means}}</td><td>{{row.gmm_stds}}</td><td>{{'%.3f' % row.pre_mean}}</td><td>{{'%.3f' % row.post_mean}}</td><td>{{'%.3f' % row.delta_mean}}</td></tr>
    {% endfor %}
    </table>
    <h2>GMM Gating Plots</h2>
    {% for p in plots %}
    <h3>{{p.column}}</h3>
    <img src="data:image/png;base64,{{p.img}}"/><br/>
    {% endfor %}
    <h2>Pre vs Post Gating Scatter Plots</h2>
    {% for p in delta_plots %}
    <h3>{{p.column}}</h3>
    <img src="data:image/png;base64,{{p.img}}"/><br/>
    {% endfor %}
    </body></html>
    ''')
    with open(args.html_report, 'w') as f:
        f.write(html_template.render(gating_results=gating_results, plots=plots, delta_plots=delta_plots))

if __name__ == "__main__":
    main()