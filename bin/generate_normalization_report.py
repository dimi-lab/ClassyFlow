#!/usr/bin/env python3

import json
import glob
import argparse
from pathlib import Path
import logging
import numpy as np
import plotly.graph_objects as go

from jinja2 import Environment, FileSystemLoader, select_autoescape

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def number_format(value):
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:,}"
    return value


def setup_jinja_environment(template_dir):
    jinja_env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(['html', 'xml'])
    )
    jinja_env.filters['number_format'] = number_format
    jinja_env.filters['default'] = lambda v, d="N/A": v if v is not None else d
    jinja_env.filters['format'] = lambda fmt, v: fmt % v if v is not None else "N/A"
    jinja_env.filters['title'] = lambda s: s.title() if s else ""
    return jinja_env


def create_no_normalization_data():
    return {
        'no_normalization_applied': True,
        'no_normalization_reason': "No normalization was performed on the data",
        'aggregate': None,
        'forest_plots': None,
        'batches': []
    }


def read_batch_results(norm_dir, transformation_type):
    results = []
    pattern = f"{transformation_type}_results_*.json"
    result_files = glob.glob(str(norm_dir / pattern))
    
    for file_path in sorted(result_files):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            results.append(data)
            logger.info(f"Loaded {transformation_type} results for batch: {data.get('batch_name', 'Unknown')}")
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
    
    return results


def read_gmm_results(norm_dir):
    results = {}
    pattern = "gmm_results_*.json"
    result_files = glob.glob(str(norm_dir / pattern))
    
    for file_path in sorted(result_files):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            batch_name = data.get('batch_name', 'Unknown')
            results[batch_name] = data
            logger.info(f"Loaded GMM results for batch: {batch_name}")
        except Exception as e:
            logger.error(f"Error reading GMM file {file_path}: {e}")
    
    return results


def compute_per_marker_aggregates(batch_results, metric_type):
    marker_data = {}
    
    for batch in batch_results:
        metrics = batch.get(f'{metric_type}_metrics', {})
        per_marker = metrics.get('per_marker', [])
        
        for marker_info in per_marker:
            marker = marker_info['marker']
            ratio = marker_info['ratio']
            
            if marker not in marker_data:
                marker_data[marker] = []
            
            if ratio is not None and not np.isnan(ratio) and not np.isinf(ratio):
                marker_data[marker].append(ratio)
    
    aggregates = []
    for marker, ratios in marker_data.items():
        if ratios:
            aggregates.append({
                'marker': marker,
                'mean': np.mean(ratios),
                'std': np.std(ratios),
                'n': len(ratios)
            })
    
    aggregates.sort(key=lambda x: x['mean'])
    return aggregates


def compute_gmm_per_marker_aggregates(gmm_results):
    marker_data = {}
    
    for batch_name, gmm_data in gmm_results.items():
        per_marker = gmm_data.get('per_marker', [])
        
        for marker_info in per_marker:
            marker = marker_info['marker']
            percent_gated = marker_info['percent_gated']
            
            if marker not in marker_data:
                marker_data[marker] = []
            
            if percent_gated is not None and not np.isnan(percent_gated):
                marker_data[marker].append(percent_gated)
    
    aggregates = []
    for marker, values in marker_data.items():
        if values:
            aggregates.append({
                'marker': marker,
                'mean': np.mean(values),
                'std': np.std(values),
                'n': len(values)
            })
    
    # Sort by mean % gated descending (highest first)
    aggregates.sort(key=lambda x: x['mean'], reverse=True)
    return aggregates


def create_forest_plot(marker_aggregates, title, threshold=1.0, higher_is_better=True):
    if not marker_aggregates:
        return None
    
    markers = [m['marker'] for m in marker_aggregates]
    means = [m['mean'] for m in marker_aggregates]
    stds = [m['std'] for m in marker_aggregates]
    
    display_names = []
    for m in markers:
        short = m.replace('Cell: ', '').replace(': Mean', '')
        if len(short) > 25:
            short = short[:22] + '...'
        display_names.append(short)
    
    if higher_is_better:
        colors = ['#28a745' if m >= threshold else '#dc3545' for m in means]
    else:
        colors = ['#28a745' if m <= threshold else '#dc3545' for m in means]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=means,
        y=display_names,
        mode='markers',
        marker=dict(size=10, color=colors),
        error_x=dict(type='data', array=stds, color='#6c757d', thickness=1.5),
        hovertemplate='<b>%{y}</b><br>Value: %{x:.2f}<extra></extra>'
    ))
    
    fig.add_vline(x=threshold, line_dash="dash", line_color="#6c757d", line_width=2)
    
    fig.add_annotation(
        x=threshold, y=1.02, yref='paper',
        text="Threshold", showarrow=False,
        font=dict(size=10, color='#6c757d')
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Value",
        yaxis_title=None,
        height=max(400, len(markers) * 20),
        margin=dict(l=150, r=50, t=50, b=50),
        showlegend=False,
        plot_bgcolor='white',
        xaxis=dict(gridcolor='#e9ecef', zeroline=False),
        yaxis=dict(gridcolor='#e9ecef')
    )
    
    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_gmm_forest_plot(marker_aggregates, title, threshold=25.0):
    if not marker_aggregates:
        return None
    
    markers = [m['marker'] for m in marker_aggregates]
    means = [m['mean'] for m in marker_aggregates]
    stds = [m['std'] for m in marker_aggregates]
    
    display_names = []
    for m in markers:
        short = m.replace('Cell: ', '').replace(': Mean', '')
        if len(short) > 25:
            short = short[:22] + '...'
        display_names.append(short)
    
    # Lower is better for % gated
    colors = ['#28a745' if m <= threshold else '#dc3545' for m in means]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=means,
        y=display_names,
        mode='markers',
        marker=dict(size=10, color=colors),
        error_x=dict(type='data', array=stds, color='#6c757d', thickness=1.5),
        hovertemplate='<b>%{y}</b><br>% Gated: %{x:.1f}%<extra></extra>'
    ))
    
    fig.add_vline(x=threshold, line_dash="dash", line_color="#dc3545", line_width=2)
    
    fig.add_annotation(
        x=threshold, y=1.02, yref='paper',
        text=f"Flag Threshold ({threshold}%)", showarrow=False,
        font=dict(size=10, color='#dc3545')
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="% Gated (lower is better)",
        yaxis_title=None,
        height=max(400, len(markers) * 20),
        margin=dict(l=150, r=50, t=50, b=50),
        showlegend=False,
        plot_bgcolor='white',
        xaxis=dict(gridcolor='#e9ecef', zeroline=False),
        yaxis=dict(gridcolor='#e9ecef')
    )
    
    return fig.to_html(full_html=False, include_plotlyjs=False)


def collect_normalization_data(norm_dir, no_normalization, gating_threshold):
    if no_normalization:
        logger.info("No normalization flag set")
        return create_no_normalization_data()
    
    if not norm_dir.exists():
        logger.error(f"Normalization directory not found: {norm_dir}")
        return create_no_normalization_data()
    
    # Find which transformation type was used
    batch_results = []
    transformation_type = None
    
    for trans_type in ['boxcox', 'quantile', 'log', 'minmax']:
        results = read_batch_results(norm_dir, trans_type)
        if results:
            batch_results = results
            transformation_type = trans_type
            break
    
    if not batch_results:
        logger.warning("No normalization results found")
        data = create_no_normalization_data()
        data['no_normalization_reason'] = "No normalization results found in directory"
        return data
    
    # Read GMM results
    gmm_results = read_gmm_results(norm_dir)
    
    # Compute aggregates
    skew_ratios = []
    outlier_ratios = []
    gmm_percent_gated = []
    
    for batch in batch_results:
        skew_metrics = batch.get('skewness_metrics', {})
        outlier_metrics = batch.get('outlier_metrics', {})
        
        skew_ratio = skew_metrics.get('mean_reduction_ratio')
        outlier_ratio = outlier_metrics.get('mean_reduction_ratio')
        
        if skew_ratio is not None:
            skew_ratios.append(skew_ratio)
        if outlier_ratio is not None:
            outlier_ratios.append(outlier_ratio)
    
    for gmm_data in gmm_results.values():
        avg_pct = gmm_data.get('avg_percent_gated')
        if avg_pct is not None:
            gmm_percent_gated.append(avg_pct)
    
    avg_skew_ratio = np.mean(skew_ratios) if skew_ratios else None
    avg_outlier_ratio = np.mean(outlier_ratios) if outlier_ratios else None
    avg_gmm_percent_gated = np.mean(gmm_percent_gated) if gmm_percent_gated else None
    
    # Build batch table data - flagging based on GMM gating only
    batches = []
    flagged_count = 0
    
    for batch in batch_results:
        batch_name = batch.get('batch_name', 'Unknown')
        skew_metrics = batch.get('skewness_metrics', {})
        outlier_metrics = batch.get('outlier_metrics', {})
        
        skew_ratio = skew_metrics.get('mean_reduction_ratio')
        outlier_ratio = outlier_metrics.get('mean_reduction_ratio')
        
        # Get GMM data for this batch
        gmm_data = gmm_results.get(batch_name, {})
        gmm_avg_percent_gated = gmm_data.get('avg_percent_gated')
        
        # Flag based on GMM gating threshold only
        flagged = False
        if gmm_avg_percent_gated is not None and gmm_avg_percent_gated > gating_threshold:
            flagged = True
        
        if flagged:
            flagged_count += 1
        
        batch_entry = {
            'batch_name': batch_name,
            'total_cells': batch.get('total_cells', 0),
            'total_slides': batch.get('total_slides', 0),
            'skew_ratio': skew_ratio,
            'outlier_ratio': outlier_ratio,
            'gmm_avg_percent_gated': gmm_avg_percent_gated,
            'flagged': flagged,
            'transformation_plots_html': f"pages/{transformation_type}_all_plots_{batch_name}.html",
            'gmm_gating_report': f"pages/gmm_gated_{batch_name}.html"
        }
        
        if transformation_type == 'boxcox':
            boxcox_metrics = batch.get('boxcox_metrics', {})
            total = boxcox_metrics.get('total_features', 0)
            successful = boxcox_metrics.get('successful_transforms', 0)
            batch_entry['success_rate'] = (successful / total * 100) if total > 0 else None
        
        batches.append(batch_entry)
    
    # Sort batches: flagged first, then by name
    batches.sort(key=lambda x: (not x['flagged'], x['batch_name']))
    
    # Compute per-marker aggregates for forest plots
    skew_marker_agg = compute_per_marker_aggregates(batch_results, 'skewness')
    outlier_marker_agg = compute_per_marker_aggregates(batch_results, 'outlier')
    gmm_marker_agg = compute_gmm_per_marker_aggregates(gmm_results)
    
    skew_forest = create_forest_plot(skew_marker_agg, "Skewness Reduction by Marker", threshold=1.0, higher_is_better=True)
    outlier_forest = create_forest_plot(outlier_marker_agg, "Outlier Reduction by Marker", threshold=1.0, higher_is_better=True)
    gmm_forest = create_gmm_forest_plot(gmm_marker_agg, "% Gated by Marker (GMM)", threshold=gating_threshold)
    
    return {
        'no_normalization_applied': False,
        'aggregate': {
            'method': transformation_type.title() if transformation_type else 'Unknown',
            'total_batches': len(batch_results),
            'avg_skew_ratio': avg_skew_ratio,
            'avg_outlier_ratio': avg_outlier_ratio,
            'avg_gmm_percent_gated': avg_gmm_percent_gated,
            'flagged_count': flagged_count,
            'gating_threshold': gating_threshold
        },
        'forest_plots': {
            'skewness': skew_forest,
            'outlier': outlier_forest,
            'gmm': gmm_forest
        },
        'batches': batches,
        'is_boxcox': transformation_type == 'boxcox'
    }


def generate_report(norm_data, output_file, jinja_env):
    try:
        template = jinja_env.get_template('normalization.html')
        html_content = template.render(normalization_data=norm_data)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Report generated: {output_file}")
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Generate normalization report")
    parser.add_argument('--norm-dir', type=Path, default=Path('./'),
                        help='Directory containing normalization outputs')
    parser.add_argument('--output-file', type=Path, default=Path('./normalization_report.html'),
                        help='Path to output HTML file')
    parser.add_argument('--template-dir', type=Path, default=Path('templates'),
                        help='Directory containing Jinja2 templates')
    parser.add_argument('--no-normalization', action='store_true',
                        help='Indicate that no normalization was applied')
    parser.add_argument('--gating-threshold', type=float, default=25.0,
                        help='Threshold for flagging batches based on %% gated (default: 25)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Collecting normalization data from: {args.norm_dir}")
    logger.info(f"Gating threshold: {args.gating_threshold}%")
    norm_data = collect_normalization_data(args.norm_dir, args.no_normalization, args.gating_threshold)
    
    jinja_env = setup_jinja_environment(str(args.template_dir))
    generate_report(norm_data, args.output_file, jinja_env)
    
    print("\n" + "=" * 60)
    print("NORMALIZATION REPORT COMPLETE")
    print("=" * 60)
    print(f"Output: {args.output_file}")
    
    if norm_data.get('no_normalization_applied'):
        print("No normalization was applied")
    else:
        agg = norm_data['aggregate']
        print(f"Method: {agg['method']}")
        print(f"Batches: {agg['total_batches']}")
        print(f"Avg % Gated: {agg['avg_gmm_percent_gated']:.1f}%" if agg['avg_gmm_percent_gated'] else "Avg % Gated: N/A")
        print(f"Flagged Batches (>{agg['gating_threshold']}% gated): {agg['flagged_count']}")
        print(f"Avg Skewness Ratio: {agg['avg_skew_ratio']:.2f}" if agg['avg_skew_ratio'] else "Avg Skewness Ratio: N/A")
        print(f"Avg Outlier Ratio: {agg['avg_outlier_ratio']:.2f}" if agg['avg_outlier_ratio'] else "Avg Outlier Ratio: N/A")
    
    print("=" * 60)


if __name__ == "__main__":
    main()