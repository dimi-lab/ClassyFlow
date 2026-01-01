#!/usr/bin/env python3
"""
HTML Report Generator for Cell Type Classification Pipeline

Generates a comprehensive HTML report from pipeline stage outputs using Jinja2 templates.

Usage:
    python generate_report.py \
        --input-metrics input_batch_metrics.json \
        --normalization-html normalization_report.html \
        --feature-selection-html feature_selection_report.html \
        --model-html model_report.html \
        --counts-tsv aggregated_counts.tsv \
        --report-name cell_classification_report.html
"""

import argparse
import base64
import json
import logging
import mimetypes
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================

def encode_image_to_base64(image_path: str) -> Optional[str]:
    """Convert an image file to a base64 data URI."""
    try:
        image_path = Path(image_path)
        if not image_path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return None
        
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if not mime_type or not mime_type.startswith('image/'):
            mime_type = 'image/png'
        
        with open(image_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        
        return f"data:{mime_type};base64,{encoded}"
        
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return None


def number_format(value):
    """Jinja2 filter for number formatting with commas."""
    if isinstance(value, (int, float)):
        return f"{value:,}"
    return value


def setup_jinja_environment(template_dir: str) -> Environment:
    """Set up Jinja2 environment with custom filters."""
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(['html', 'xml'])
    )
    env.filters['number_format'] = number_format
    return env


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_json(json_file: str) -> Dict:
    """Load a JSON file."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            logger.info(f"Loaded: {json_file}")
            return data
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {json_file}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading {json_file}: {e}")
        return {}


def load_html(html_file: str) -> str:
    """Load an HTML file as string."""
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
            logger.info(f"Loaded: {html_file}")
            return content
    except Exception as e:
        logger.error(f"Error loading {html_file}: {e}")
        return ""


def load_tsv(tsv_path: str) -> pd.DataFrame:
    """Load a TSV file as DataFrame."""
    try:
        df = pd.read_csv(tsv_path, sep='\t')
        logger.info(f"Loaded: {tsv_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading {tsv_path}: {e}")
        return pd.DataFrame()


# =============================================================================
# Visualization Functions
# =============================================================================

def create_synthetic_marker_heatmap(input_metrics: Dict) -> Optional[str]:
    """Create Plotly heatmap showing marker data status (real vs synthetic) by batch."""
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
    
    # Build matrix
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
    
    # Dynamic sizing
    n_markers = len(all_markers)
    fig_height = max(400, min(800, 100 + n_markers * 20))
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=matrix,
        x=batch_ids,
        y=all_markers,
        hovertext=hover_text,
        hoverinfo='text',
        colorscale=[[0, '#e8f5e9'], [1, '#fff3e0']],
        showscale=False,
        xgap=2,
        ygap=2
    ))
    
    # Legend traces
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=15, color='#e8f5e9', symbol='square', line=dict(color='#ccc', width=1)),
        name='Real', showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=15, color='#fff3e0', symbol='square', line=dict(color='#ccc', width=1)),
        name='Synthetic', showlegend=True
    ))
    
    fig.update_layout(
        title={'text': 'Marker Data Status by Batch', 'x': 0.5, 'xanchor': 'center',
               'font': {'size': 16, 'family': 'Segoe UI, sans-serif'}},
        xaxis={'title': 'Batch', 'tickangle': 45, 'side': 'bottom'},
        yaxis={'title': 'Marker', 'autorange': 'reversed'},
        autosize=True,
        height=fig_height,
        margin=dict(l=120, r=40, t=80, b=100),
        plot_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    
    return fig.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})


# =============================================================================
# Prediction Metrics & Visualization
# =============================================================================

# Thresholds for visualization switching
STACKED_BAR_SAMPLE_THRESHOLD = 50
XAXIS_LABEL_THRESHOLD = 25


def _get_cell_type_colors(cell_types) -> Dict[str, str]:
    """Generate consistent color mapping for cell types."""
    colors = sns.color_palette("Set2", len(cell_types)).as_hex()
    return dict(zip(cell_types, colors))


def compute_prediction_metrics(counts_df: pd.DataFrame) -> Dict:
    """Derive prediction metrics from aggregated counts DataFrame."""
    
    # Global stats
    total_cells = counts_df.groupby('sample_name')['total_cells'].first().sum()
    total_low_density = counts_df.groupby('sample_name')['low_density_cells'].first().sum()
    
    # Overall cell type abundance
    global_counts = counts_df.groupby('cell_type')['count'].sum().sort_values(ascending=False)
    
    # Per-sample stats
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


def create_abundance_plot(counts_df: pd.DataFrame, output_path: str) -> str:
    """
    Create adaptive abundance visualization based on sample count.
    
    Returns HTML string of the Plotly figure.
    """
    n_samples = counts_df['sample_name'].nunique()
    
    if n_samples <= STACKED_BAR_SAMPLE_THRESHOLD:
        html = _create_stacked_bar_per_sample(counts_df)
    else:
        html = _create_large_dataset_view(counts_df)
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return html


def _create_stacked_bar_per_sample(counts_df: pd.DataFrame) -> str:
    """Create interactive stacked bar chart for small datasets."""
    
    # Pivot to percentages
    pivot_df = counts_df.pivot(
        index='sample_name', columns='cell_type', values='count'
    ).fillna(0)
    pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
    
    # Order by abundance
    col_order = counts_df.groupby('cell_type')['count'].sum().sort_values(ascending=False).index
    pivot_df = pivot_df.reindex(columns=col_order, fill_value=0)
    
    color_map = _get_cell_type_colors(pivot_df.columns)
    n_samples = len(pivot_df)
    
    fig = go.Figure()
    
    for cell_type in pivot_df.columns:
        fig.add_trace(go.Bar(
            name=cell_type,
            x=pivot_df.index,
            y=pivot_df[cell_type],
            marker_color=color_map[cell_type],
            hovertemplate=f'<b>%{{x}}</b><br>{cell_type}: %{{y:.1f}}%<extra></extra>'
        ))
    
    show_labels = n_samples <= XAXIS_LABEL_THRESHOLD
    
    fig.update_layout(
        barmode='stack',
        title={'text': f'Cell Type Composition by Sample (n={n_samples})', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 16}},
        xaxis={'title': 'Sample' if show_labels else f'Samples (n={n_samples})',
               'tickangle': 45 if show_labels else 0, 'showticklabels': show_labels},
        yaxis={'title': 'Percentage (%)', 'range': [0, 100]},
        legend={'title': 'Cell Type', 'traceorder': 'normal'},
        autosize=True,
        height=500,
        hovermode='x unified',
        plot_bgcolor='white'
    )
    fig.update_xaxes(gridcolor='#eee')
    fig.update_yaxes(gridcolor='#eee')
    
    return fig.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})


def _create_large_dataset_view(counts_df: pd.DataFrame) -> str:
    """Create batch-aggregated bar + violin plots for large datasets."""
    from plotly.subplots import make_subplots
    
    has_batch = 'batch' in counts_df.columns and counts_df['batch'].notna().any() and (counts_df['batch'] != '').any()
    
    col_order = counts_df.groupby('cell_type')['count'].sum().sort_values(ascending=False).index.tolist()
    color_map = _get_cell_type_colors(col_order)
    
    # Per-sample percentages for violin
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
    
    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.4, 0.6], vertical_spacing=0.12,
        subplot_titles=(
            f'Cell Type Composition by {"Batch" if has_batch else "Dataset"} (n={n_samples} samples)',
            'Cell Type Distribution Across Samples'
        )
    )
    
    # Batch-aggregated stacked bar
    if has_batch:
        batch_counts = counts_df.groupby(['batch', 'cell_type'])['count'].sum().reset_index()
        batch_totals = batch_counts.groupby('batch')['count'].transform('sum')
        batch_counts['percentage'] = batch_counts['count'] / batch_totals * 100
        pivot_batch = batch_counts.pivot(index='batch', columns='cell_type', values='percentage').fillna(0)
        pivot_batch = pivot_batch.reindex(columns=col_order, fill_value=0)
        x_labels = pivot_batch.index.tolist()
    else:
        total_counts = counts_df.groupby('cell_type')['count'].sum()
        total_pct = (total_counts / total_counts.sum() * 100).reindex(col_order, fill_value=0)
        pivot_batch = pd.DataFrame({'All Samples': total_pct}).T
        x_labels = ['All Samples']
    
    for cell_type in col_order:
        fig.add_trace(go.Bar(
            name=cell_type, x=x_labels,
            y=pivot_batch[cell_type] if cell_type in pivot_batch.columns else [0] * len(x_labels),
            marker_color=color_map[cell_type], legendgroup=cell_type,
            hovertemplate=f'{cell_type}: %{{y:.1f}}%<extra></extra>'
        ), row=1, col=1)
    
    # Violin plots
    for cell_type in col_order:
        ct_data = pct_df[pct_df['cell_type'] == cell_type]
        fig.add_trace(go.Violin(
            x=[cell_type] * len(ct_data), y=ct_data['percentage'],
            name=cell_type, legendgroup=cell_type, showlegend=False,
            fillcolor=color_map[cell_type], line_color=color_map[cell_type],
            opacity=0.7, box_visible=True, meanline_visible=True, points='outliers',
            hovertemplate=f'<b>{cell_type}</b><br>Percentage: %{{y:.1f}}%<extra></extra>'
        ), row=2, col=1)
    
    fig.update_layout(
        barmode='stack',
        autosize=True,
        height=800,
        legend={'title': 'Cell Type', 'traceorder': 'normal', 'orientation': 'v',
                'yanchor': 'top', 'y': 1, 'xanchor': 'left', 'x': 1.02},
        plot_bgcolor='white', hovermode='closest'
    )
    
    fig.update_xaxes(title_text='Batch' if has_batch else '', row=1, col=1, gridcolor='#eee')
    fig.update_yaxes(title_text='Percentage (%)', range=[0, 100], row=1, col=1, gridcolor='#eee')
    fig.update_xaxes(title_text='Cell Type', tickangle=45, row=2, col=1, gridcolor='#eee')
    fig.update_yaxes(title_text='Percentage (%)', row=2, col=1, gridcolor='#eee')
    
    return fig.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})


# =============================================================================
# Data Collection Functions
# =============================================================================

def collect_input_metrics(input_metrics_path: str) -> Dict:
    """Load and process input metrics JSON."""
    input_metrics = load_json(input_metrics_path)
    
    # Generate heatmap if marker status matrix exists
    if input_metrics.get('marker_status_matrix'):
        input_metrics['synthetic_heatmap_html'] = create_synthetic_marker_heatmap(input_metrics)
    
    # Check for synthetic warnings
    input_metrics['any_synthetic_warnings'] = any(
        b.get('percent_synthetic', 0) > 25 for b in input_metrics.get('batches', [])
    )
    
    return {'input_metrics': input_metrics}


def collect_html_content(normalization_html: str, feature_selection_html: str, model_html: str) -> Dict:
    """Load pre-rendered HTML content for each section."""
    return {
        'normalization_html_content': load_html(normalization_html),
        'feature_selection_html_content': load_html(feature_selection_html),
        'model_html_content': load_html(model_html)
    }


def collect_prediction_content(counts_tsv: str, plots_dir: str) -> Dict:
    """Generate prediction metrics and visualizations from counts TSV."""
    counts_df = load_tsv(counts_tsv)
    
    if counts_df.empty:
        return {'prediction_metrics': {}}
    
    metrics = compute_prediction_metrics(counts_df)
    
    # Generate abundance plot
    abundance_plot_path = os.path.join(plots_dir, "prediction_abundance_plot.html")
    metrics['abundance_plot_html'] = create_abundance_plot(counts_df, abundance_plot_path)
    
    return {'prediction_metrics': metrics}


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(all_data: Dict, output_file: str, jinja_env: Environment,
                    letterhead: Optional[str], pipeline_version: str,
                    template_name: str = "base.html"):
    """Render and write the final HTML report."""
    try:
        template = jinja_env.get_template(template_name)
        
        template_data = {
            'pipeline_version': pipeline_version,
            'generation_date': datetime.now().strftime("%B %d, %Y"),
            'letterhead': encode_image_to_base64(letterhead) if letterhead else None,
        }
        template_data.update(all_data)
        
        html_content = template.render(**template_data)
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Report generated: {output_path}")
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML report from pipeline outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input files (all explicit)
    parser.add_argument('--input-metrics', required=True,
                        help='Path to input_batch_metrics.json')
    parser.add_argument('--normalization-html', required=True,
                        help='Path to normalization_report.html')
    parser.add_argument('--feature-selection-html', required=True,
                        help='Path to feature_selection_report.html')
    parser.add_argument('--model-html', required=True,
                        help='Path to model_report.html')
    parser.add_argument('--counts-tsv', required=True,
                        help='Path to aggregated cell counts TSV')
    
    # Output configuration
    parser.add_argument('--plots-dir', default='.',
                        help='Directory for generated plot files')
    parser.add_argument('--template-dir', default='templates',
                        help='Directory containing Jinja2 templates')
    parser.add_argument('--report-name', default='cell_classification_report.html',
                        help='Output report filename')
    
    # Optional
    parser.add_argument('--letterhead',
                        help='Path to header logo image (embedded in report)')
    parser.add_argument('--version', default='N/A',
                        help='Pipeline version for report footer')
    
    args = parser.parse_args()
    
    # Set up Jinja2
    jinja_env = setup_jinja_environment(args.template_dir)
    
    # Collect all data
    all_data = {}
    all_data.update(collect_input_metrics(args.input_metrics))
    all_data.update(collect_html_content(
        args.normalization_html,
        args.feature_selection_html,
        args.model_html
    ))
    all_data.update(collect_prediction_content(args.counts_tsv, args.plots_dir))
    
    # Generate report
    generate_report(
        all_data=all_data,
        output_file=args.report_name,
        jinja_env=jinja_env,
        letterhead=args.letterhead,
        pipeline_version=args.version
    )
    
    # Debug output
    with open('output.json', 'w') as f:
        json.dump(all_data, f, indent=2, default=str)


if __name__ == "__main__":
    main()