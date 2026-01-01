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


# Threshold for switching between per-sample and aggregated views
STACKED_BAR_SAMPLE_THRESHOLD = 50
# Threshold for hiding x-axis labels (when labels become unreadable)
XAXIS_LABEL_THRESHOLD = 25


def create_abundance_plot_from_counts(counts_df, output_html):
    """
    Create adaptive abundance visualization based on sample count.
    
    ≤ STACKED_BAR_SAMPLE_THRESHOLD: Interactive stacked bar per sample
    > STACKED_BAR_SAMPLE_THRESHOLD: Batch-aggregated stacked bar + violin plots
    
    Returns HTML string of the Plotly figure(s).
    """
    n_samples = counts_df['sample_name'].nunique()
    
    if n_samples <= STACKED_BAR_SAMPLE_THRESHOLD:
        html = _create_stacked_bar_per_sample(counts_df)
    else:
        html = _create_large_dataset_view(counts_df)
    
    # Write HTML to file
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return html


def _get_cell_type_colors(cell_types):
    """Generate consistent color mapping for cell types"""
    colors = sns.color_palette("Set2", len(cell_types)).as_hex()
    return dict(zip(cell_types, colors))


def _create_stacked_bar_per_sample(counts_df):
    """Create interactive stacked bar chart for small datasets"""
    
    # Pivot to sample × cell_type matrix
    pivot_df = counts_df.pivot(
        index='sample_name',
        columns='cell_type',
        values='count'
    ).fillna(0)
    
    # Convert to percentages
    pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
    
    # Order columns by total abundance (most abundant first)
    col_order = counts_df.groupby('cell_type')['count'].sum().sort_values(ascending=False).index
    pivot_df = pivot_df.reindex(columns=col_order, fill_value=0)
    
    # Get sample totals for hover
    sample_totals = counts_df.groupby('sample_name')['total_cells'].first()
    
    # Color mapping
    color_map = _get_cell_type_colors(pivot_df.columns)
    
    n_samples = len(pivot_df)
    
    # Create stacked bar chart
    fig = go.Figure()
    
    for cell_type in pivot_df.columns:
        fig.add_trace(go.Bar(
            name=cell_type,
            x=pivot_df.index,
            y=pivot_df[cell_type],
            marker_color=color_map[cell_type],
            hovertemplate=(
                '<b>%{x}</b><br>'
                f'{cell_type}: %{{y:.1f}}%<br>'
                '<extra></extra>'
            )
        ))
    
    # Dynamic sizing
    width = max(600, min(1400, 400 + n_samples * 20))
    height = 500
    
    # Determine if x-axis labels should be shown
    show_xaxis_labels = n_samples <= XAXIS_LABEL_THRESHOLD
    
    fig.update_layout(
        barmode='stack',
        title={
            'text': f'Cell Type Composition by Sample (n={n_samples})',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis={
            'title': 'Sample' if show_xaxis_labels else f'Samples (n={n_samples})',
            'tickangle': 45 if show_xaxis_labels else 0,
            'showticklabels': show_xaxis_labels,
        },
        yaxis={
            'title': 'Percentage (%)',
            'range': [0, 100]
        },
        legend={
            'title': 'Cell Type',
            'traceorder': 'normal'
        },
        width=width,
        height=height,
        hovermode='x unified',
        plot_bgcolor='white'
    )
    
    fig.update_xaxes(gridcolor='#eee')
    fig.update_yaxes(gridcolor='#eee')
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def _create_large_dataset_view(counts_df):
    """Create batch-aggregated bar + violin plots for large datasets"""
    from plotly.subplots import make_subplots
    
    has_batch = 'batch' in counts_df.columns and counts_df['batch'].notna().any() and (counts_df['batch'] != '').any()
    
    # Get cell type order by overall abundance
    col_order = counts_df.groupby('cell_type')['count'].sum().sort_values(ascending=False).index.tolist()
    color_map = _get_cell_type_colors(col_order)
    
    # Calculate per-sample percentages for violin plots
    sample_percentages = []
    for sample_name, group in counts_df.groupby('sample_name'):
        total = group['total_cells'].iloc[0]
        batch = group['batch'].iloc[0] if has_batch else 'All'
        for _, row in group.iterrows():
            sample_percentages.append({
                'sample_name': sample_name,
                'batch': batch,
                'cell_type': row['cell_type'],
                'percentage': (row['count'] / total * 100) if total > 0 else 0
            })
    pct_df = pd.DataFrame(sample_percentages)
    
    n_samples = counts_df['sample_name'].nunique()
    n_batches = pct_df['batch'].nunique() if has_batch else 1
    
    # Create subplots: top for stacked bar, bottom for violin
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.4, 0.6],
        vertical_spacing=0.12,
        subplot_titles=(
            f'Cell Type Composition by {"Batch" if has_batch else "Dataset"} (n={n_samples} samples)',
            'Cell Type Distribution Across Samples'
        )
    )
    
    # === Top plot: Batch-aggregated stacked bar ===
    if has_batch:
        # Aggregate by batch
        batch_counts = counts_df.groupby(['batch', 'cell_type'])['count'].sum().reset_index()
        batch_totals = batch_counts.groupby('batch')['count'].transform('sum')
        batch_counts['percentage'] = (batch_counts['count'] / batch_totals * 100)
        
        pivot_batch = batch_counts.pivot(index='batch', columns='cell_type', values='percentage').fillna(0)
        pivot_batch = pivot_batch.reindex(columns=col_order, fill_value=0)
        
        x_labels = pivot_batch.index.tolist()
    else:
        # Single bar for entire dataset
        total_counts = counts_df.groupby('cell_type')['count'].sum()
        total_pct = (total_counts / total_counts.sum() * 100).reindex(col_order, fill_value=0)
        
        pivot_batch = pd.DataFrame({'All Samples': total_pct}).T
        x_labels = ['All Samples']
    
    # Add stacked bars
    for cell_type in col_order:
        fig.add_trace(
            go.Bar(
                name=cell_type,
                x=x_labels,
                y=pivot_batch[cell_type] if cell_type in pivot_batch.columns else [0] * len(x_labels),
                marker_color=color_map[cell_type],
                legendgroup=cell_type,
                hovertemplate=f'{cell_type}: %{{y:.1f}}%<extra></extra>'
            ),
            row=1, col=1
        )
    
    # === Bottom plot: Violin plots per cell type ===
    for i, cell_type in enumerate(col_order):
        ct_data = pct_df[pct_df['cell_type'] == cell_type]
        
        fig.add_trace(
            go.Violin(
                x=[cell_type] * len(ct_data),
                y=ct_data['percentage'],
                name=cell_type,
                legendgroup=cell_type,
                showlegend=False,
                fillcolor=color_map[cell_type],
                line_color=color_map[cell_type],
                opacity=0.7,
                box_visible=True,
                meanline_visible=True,
                points='outliers',
                hovertemplate=(
                    f'<b>{cell_type}</b><br>'
                    'Percentage: %{y:.1f}%<br>'
                    '<extra></extra>'
                )
            ),
            row=2, col=1
        )
    
    # Layout
    fig.update_layout(
        barmode='stack',
        height=800,
        width=max(800, min(1400, 400 + len(col_order) * 80)),
        legend={
            'title': 'Cell Type',
            'traceorder': 'normal',
            'orientation': 'v',
            'yanchor': 'top',
            'y': 1,
            'xanchor': 'left',
            'x': 1.02
        },
        plot_bgcolor='white',
        hovermode='closest'
    )
    
    # Update axes
    fig.update_xaxes(title_text='Batch' if has_batch else '', row=1, col=1, gridcolor='#eee')
    fig.update_yaxes(title_text='Percentage (%)', range=[0, 100], row=1, col=1, gridcolor='#eee')
    fig.update_xaxes(title_text='Cell Type', tickangle=45, row=2, col=1, gridcolor='#eee')
    fig.update_yaxes(title_text='Percentage (%)', row=2, col=1, gridcolor='#eee')
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')



def generate_prediction_content(counts_tsv_path, plots_dir='.'):
    """Generate prediction content from aggregated counts TSV"""
    
    counts_df = read_aggregated_counts(counts_tsv_path)
    
    # Compute metrics
    metrics = compute_prediction_metrics_from_counts(counts_df)
    
    # Generate abundance plot (now outputs HTML)
    abundance_plot_name = "prediction_abundance_plot.html"
    abundance_plot_path = os.path.join(plots_dir, abundance_plot_name)
    abundance_html = create_abundance_plot_from_counts(counts_df, abundance_plot_path)
    metrics['abundance_plot'] = abundance_plot_name
    metrics['abundance_plot_html'] = abundance_html
    
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