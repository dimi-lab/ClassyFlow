#!/usr/bin/env python3
"""
HTML Report Generator for Cell Type Classification Pipeline

This script reads JSON outputs from various pipeline stages and generates
a comprehensive HTML report using Jinja2 templates.

Usage:
    python generate_report.py --output-dir /path/to/pipeline/output --report-dir /path/to/report/output
"""

import os
import json
import glob
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import base64
import mimetypes
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
from typing import Any, Dict, Union

from jinja2 import Environment, FileSystemLoader, select_autoescape

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """
    Convert an image file to a base64 data URI.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 data URI string or None if file doesn't exist/error
    """
    try:
        image_path = Path(image_path)
        if not image_path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return None
        
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if not mime_type or not mime_type.startswith('image/'):
            # Default to PNG if we can't determine the type
            mime_type = 'image/png'
        
        # Read and encode the image
        with open(image_path, 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Create data URI
        data_uri = f"data:{mime_type};base64,{encoded_image}"
        logger.info(f"Successfully embedded image: {image_path}")
        return data_uri
        
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return None



def number_format(value):
    """Custom Jinja2 filter for number formatting."""
    if isinstance(value, (int, float)):
        return f"{value:,}"
    return value


def setup_jinja_environment(template_dir: str):
    """Set up and return Jinja2 environment with custom filters."""
    jinja_env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(['html', 'xml'])
    )
    
    # Add custom filters
    jinja_env.filters['number_format'] = number_format
    
    return jinja_env


def load_metrics_json(json_file):
    """
    Load a single metrics JSON file
    
    Args:
        json_file (str): Path to the metrics JSON file
        
    Returns:
        dict: Dictionary containing the metrics data
    """
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
            print(f"Loaded metrics: {json_file}")
            return data
    except json.JSONDecodeError as e:
        print(f"Error reading {json_file}: {e}")
        return {}
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        return {}
    

def read_df(file_path):
    """Read cell count table from CSV file."""
    try:
        df = pd.read_csv(file_path)
        return {
            'headers': df.columns.tolist(),
            'rows': df.to_dict('records')
        }
    except Exception as e:
        logger.warning(f"Could not df: {e}")
        return {'headers': [], 'rows': []}
    
def read_html_chunk(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    return html_content


def create_synthetic_marker_heatmap(input_metrics):
    """
    Create plotly heatmap from input_metrics marker_status_matrix.
    """
    marker_status_matrix = input_metrics.get('marker_status_matrix', {})
    all_markers = input_metrics.get('all_markers', [])
    
    if not marker_status_matrix or not all_markers:
        return None
    
    first_marker = all_markers[0] if all_markers else None
    if not first_marker:
        return None
    batch_ids = sorted(marker_status_matrix.get(first_marker, {}).keys())
    
    if not batch_ids:
        return None
    
    matrix = []
    hover_text = []
    for marker in all_markers:
        row = []
        hover_row = []
        for batch in batch_ids:
            status = marker_status_matrix.get(marker, {}).get(batch, 'real')
            is_synthetic = status == 'synthetic'
            row.append(1 if is_synthetic else 0)
            hover_row.append(f"Marker: {marker}<br>Batch: {batch}<br>Status: {status}")
        matrix.append(row)
        hover_text.append(hover_row)
    
    n_markers = len(all_markers)
    n_batches = len(batch_ids)
    fig_height = max(400, min(800, 100 + n_markers * 20))
    
    min_col_width = 120
    max_width = min(1200, 150 + n_batches * min_col_width)
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=matrix,
        x=batch_ids,
        y=all_markers,
        hovertext=hover_text,
        hoverinfo='text',
        colorscale=[
            [0, '#e8f5e9'],
            [1, '#fff3e0']
        ],
        showscale=False,
        xgap=2,
        ygap=2
    ))
    
    # Add invisible traces for legend
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=15, color='#e8f5e9', symbol='square', line=dict(color='#ccc', width=1)),
        name='Real',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=15, color='#fff3e0', symbol='square', line=dict(color='#ccc', width=1)),
        name='Synthetic',
        showlegend=True
    ))
    
    fig.update_layout(
        title={
            'text': 'Marker Data Status by Batch',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'family': 'Segoe UI, sans-serif'}
        },
        xaxis={
            'title': 'Batch',
            'tickangle': 45,
            'side': 'bottom'
        },
        yaxis={
            'title': 'Marker',
            'autorange': 'reversed'
        },
        autosize=True,
        height=fig_height,
        margin=dict(l=120, r=40, t=80, b=100),
        plot_bgcolor='white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        )
    )
    
    html = fig.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})
    return f'<div style="max-width: {max_width}px; margin: 0 auto;">{html}</div>'


def collect_html_content(input_dir):
    input_dir = Path(input_dir)
    all_jsons = {
        'normalization_html_content': read_html_chunk(input_dir / "normalization_report.html"),
        'feature_selection_html_content': read_html_chunk(input_dir / "feature_selection_report.html"),
        'model_html_content': read_html_chunk(input_dir / "model_report.html")
        }

    return all_jsons

def collect_metric_data(input_dir):
    input_dir = Path(input_dir)
    input_metrics = load_metrics_json(input_dir / "input_batch_metrics.json")
    
    # Generate heatmap HTML from input_metrics
    heatmap_html = None
    if input_metrics.get('marker_status_matrix'):
        heatmap_html = create_synthetic_marker_heatmap(input_metrics)
    
    # Check for warnings (any batch >25% synthetic)
    any_warnings = any(b.get('percent_synthetic', 0) > 25 for b in input_metrics.get('batches', []))
    input_metrics['synthetic_heatmap_html'] = heatmap_html
    input_metrics['any_synthetic_warnings'] = any_warnings
    
    all_jsons = {
        'input_metrics': input_metrics,
    }

    return all_jsons


def read_aggregated_counts(tsv_path):
    """Read pre-aggregated counts from collectFile output"""
    df = pd.read_csv(tsv_path, sep='\t')
    return df


def compute_prediction_metrics_from_counts(counts_df):
    """Derive all prediction metrics from aggregated counts TSV"""
    
    # Global stats
    total_cells = counts_df.groupby('sample_name')['total_cells'].first().sum()
    total_low_density = counts_df.groupby('sample_name')['low_density_cells'].first().sum()
    
    # Overall cell type abundance
    global_counts = counts_df.groupby('cell_type')['count'].sum().sort_values(ascending=False)
    
    # Per-sample stats for the ROI table
    samples = []
    for sample_name, group in counts_df.groupby('sample_name'):
        sample_total = group['total_cells'].iloc[0]
        cell_counts = group.set_index('cell_type')['count'].sort_values(ascending=False)
        percentages = (cell_counts / sample_total * 100).round(1)
        
        samples.append({
            'sample_name': sample_name,
            'total_cells': int(sample_total),
            'unique_classes': len(cell_counts),
            'low_density_cells': int(group['low_density_cells'].iloc[0]),
            'most_common_class': cell_counts.index[0] if len(cell_counts) > 0 else None,
            'most_common_percentage': float(percentages.iloc[0]) if len(percentages) > 0 else None,
            'second_common_class': cell_counts.index[1] if len(cell_counts) > 1 else None,
            'second_common_percentage': float(percentages.iloc[1]) if len(percentages) > 1 else None,
            'least_common_class': cell_counts.index[-1] if len(cell_counts) > 0 else None,
            'least_common_percentage': float(percentages.iloc[-1]) if len(percentages) > 0 else None,
            'roi_report': group['roi_report'].iloc[0]
        })
    
    return {
        'total_predicted_cells': int(total_cells),
        'total_low_density_cells': int(total_low_density),
        'most_common_prediction': global_counts.index[0] if len(global_counts) > 0 else None,
        'most_rare_prediction': global_counts.index[-1] if len(global_counts) > 0 else None,
        'samples': sorted(samples, key=lambda x: x['sample_name'])
    }


def create_abundance_plot_from_counts(counts_df, output_file):
    """Create stacked bar plot from pre-aggregated counts"""
    
    # Pivot to sample × cell_type matrix
    pivot_df = counts_df.pivot(
        index='sample_name', 
        columns='cell_type', 
        values='count'
    ).fillna(0)
    
    # Convert to percentages
    pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
    
    # Order columns by total abundance
    col_order = counts_df.groupby('cell_type')['count'].sum().sort_values(ascending=False).index
    pivot_df = pivot_df.reindex(columns=col_order, fill_value=0)
    
    # Sizing
    n_samples = len(pivot_df)
    n_cell_types = len(pivot_df.columns)
    plot_width = max(12, min(30, 10 + n_samples * 0.4))
    plot_height = max(8, min(12, 7 + n_cell_types * 0.15))
    
    # Plot
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(plot_width, plot_height), dpi=300)
    
    colors = sns.color_palette("Set2", n_cell_types)
    bottom = np.zeros(len(pivot_df))
    
    for i, cell_type in enumerate(pivot_df.columns):
        ax.bar(
            range(len(pivot_df)), 
            pivot_df[cell_type], 
            bottom=bottom,
            label=cell_type,
            color=colors[i % len(colors)],
            alpha=0.85,
            edgecolor='white',
            linewidth=0.8
        )
        bottom += pivot_df[cell_type].values
    
    # Labels
    fig.suptitle('Predicted Cell Type Composition by Sample', fontsize=15, fontweight='bold', y=1.02)
    ax.set_xlabel('Sample', fontsize=11, fontweight='bold')
    ax.set_ylabel('Percentage of Cells (%)', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 100)
    
    # X-axis
    ax.set_xticks(range(len(pivot_df)))
    rotation = 30 if n_samples <= 10 else 45 if n_samples <= 25 else 70
    fontsize = 10 if n_samples <= 10 else 9 if n_samples <= 25 else 8
    ax.set_xticklabels(pivot_df.index, rotation=rotation, ha='right', fontsize=fontsize)
    
    # Sample counts
    sample_totals = counts_df.groupby('sample_name')['total_cells'].first()
    for i, sample in enumerate(pivot_df.index):
        ax.text(i, 104, f'n={sample_totals[sample]:,}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                rotation=0 if n_samples <= 15 else 30)
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels),
              bbox_to_anchor=(1.05, 1), loc='upper left',
              title='Cell Type', title_fontsize=11, fontsize=10)
    
    ax.grid(True, alpha=0.3)
    sns.despine(top=True, right=True)
    ax.set_facecolor('#fafafa')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_abundance_plot_heatmap_from_counts(counts_df, output_file):
    """Create clustered heatmap of cell type composition from pre-aggregated counts"""
    
    n_samples = counts_df['sample_name'].nunique()
    
    # Pivot to sample × cell_type matrix
    proportions = counts_df.pivot(
        index='sample_name',
        columns='cell_type',
        values='count'
    ).fillna(0)
    proportions = proportions.div(proportions.sum(axis=1), axis=0) * 100
    
    # Sort columns by overall abundance
    col_order = counts_df.groupby('cell_type')['count'].sum().sort_values(ascending=False).index
    proportions = proportions.reindex(columns=col_order, fill_value=0)
    
    # Build batch color annotation if batch column exists and has values
    row_colors = None
    if 'batch' in counts_df.columns and counts_df['batch'].notna().any() and (counts_df['batch'] != '').any():
        batch_map = counts_df.drop_duplicates('sample_name').set_index('sample_name')['batch']
        batch_map = batch_map.reindex(proportions.index)
        if batch_map.nunique() > 1:
            batch_palette = dict(zip(batch_map.unique(), sns.color_palette('tab20', batch_map.nunique())))
            row_colors = batch_map.map(batch_palette)
            row_colors.name = 'Batch'
    
    # Dynamic sizing
    fig_height = np.clip(6 + n_samples * 0.02, 8, 24)
    fig_width = max(10, 6 + len(col_order) * 0.5)
    
    # Clustering and dendrogram settings based on dataset size
    show_row_dendrogram = n_samples <= 500
    
    g = sns.clustermap(
        proportions,
        row_colors=row_colors,
        col_cluster=True,
        row_cluster=True,
        dendrogram_ratio=(0.15 if show_row_dendrogram else 0.001, 0.15),
        cmap='Blues',
        figsize=(fig_width, fig_height),
        xticklabels=True,
        yticklabels=n_samples <= 100,
        cbar_kws={'label': 'Percentage (%)'},
        linewidths=0 if n_samples > 200 else 0.1,
    )
    
    # Hide row dendrogram for large datasets (still clusters, just doesn't show tree)
    if not show_row_dendrogram:
        g.ax_row_dendrogram.set_visible(False)
    
    g.ax_heatmap.set_xlabel('Cell Type', fontsize=11, fontweight='bold')
    g.ax_heatmap.set_ylabel('Sample' if n_samples <= 100 else '', fontsize=11, fontweight='bold')
    g.figure.suptitle('Predicted Cell Type Composition by Sample', fontsize=14, fontweight='bold', y=1.02)
    
    # Rotate column labels for readability
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    
    # Add batch legend if applicable
    if row_colors is not None:
        batch_map = counts_df.drop_duplicates('sample_name').set_index('sample_name')['batch']
        batch_palette = dict(zip(batch_map.unique(), sns.color_palette('tab20', batch_map.nunique())))
        for batch, color in batch_palette.items():
            g.ax_col_dendrogram.bar(0, 0, color=color, label=batch, linewidth=0)
        g.ax_col_dendrogram.legend(
            title='Batch', 
            loc='upper left', 
            bbox_to_anchor=(1.05, 1),
            fontsize=9
        )
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_prediction_content(counts_tsv_path, plots_dir='.'):
    """Generate prediction content from aggregated counts TSV"""
    
    counts_df = read_aggregated_counts(counts_tsv_path)
    
    # Compute metrics
    metrics = compute_prediction_metrics_from_counts(counts_df)
    
    # Generate plots
    abundance_plot_name = "prediction_abundance_plot.png"
    abundance_plot_path = os.path.join(plots_dir, abundance_plot_name)
    create_abundance_plot_from_counts(counts_df, abundance_plot_path)
    metrics['abundance_plot'] = abundance_plot_name
    
    # Generate heatmap
    heatmap_plot_path = os.path.join(plots_dir, "prediction_abundance_plot_heatmap.png")
    create_abundance_plot_heatmap_from_counts(counts_df, heatmap_plot_path)
    
    return {'prediction_metrics': metrics}


def generate_report(all_data, output_file, jinja_env, letterhead, pipeline_version, template_name = "base.html"):
    try:
        template = jinja_env.get_template(template_name)
        
        template_data = {
            'pipeline_version': pipeline_version,
            'generation_date': datetime.now().strftime("%B %d, %Y"),
            'letterhead': encode_image_to_base64(letterhead) if letterhead else None,
        }

        template_data.update(all_data)

        # Render the template
        html_content = template.render(**template_data)
        
        # Write to output file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Report generated successfully: {output_path}")
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise


def main():
    """Main function to parse arguments and generate report."""
    parser = argparse.ArgumentParser(description="Generate HTML report from pipeline outputs")
    parser.add_argument('--input-dir', default="./", help='Directory containing pipeline outputs (default: ./)')
    parser.add_argument('--counts-tsv', required=True, help='Aggregated cell counts TSV from collectFile')
    parser.add_argument('--template-dir', default='templates', help='Directory containing Jinja2 templates (default: templates)')
    parser.add_argument('--report-name', default='cell_classification_report.html', help='Name of the output report file (default: cell_classification_report.html)')
    parser.add_argument('--letterhead', help='Path to header logo image (will be embedded in report)')
    parser.add_argument('--version', help='Pipeline version to be displayed in final report', default="N/A")

    args = parser.parse_args()
    
    # Set up Jinja2 environment
    jinja_env = setup_jinja_environment(args.template_dir)
    
    # Collect all data
    all_data = collect_html_content(args.input_dir)

    all_data.update(collect_metric_data(args.input_dir))

    all_data.update(generate_prediction_content(args.counts_tsv))

    generate_report(
        all_data=all_data, 
        output_file=args.report_name, 
        jinja_env=jinja_env, 
        letterhead=args.letterhead,
        pipeline_version=args.version
    )

    with open('output.json', 'w') as f:
        json.dump(all_data, f, indent=2, default=str)

if __name__ == "__main__":
    main()