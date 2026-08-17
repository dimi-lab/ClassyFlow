#!/usr/bin/env python3
"""
Light HTML Report Generator for the ClassyFlow Cell Type Classification Pipeline.

This is a *self-contained* assembler: Nextflow stages each ``bin/`` script
individually, so this file deliberately copies the small loader/plot helpers it
needs from ``generate_final_report.py`` rather than importing them.

The light report is a single scrolling page aimed at a PI audience. It reuses
~80% of artifacts already produced by the pipeline (JSON summaries + pre-rendered
Plotly div HTML) and drops the heavy visualizations of the full report.

Four sections:
    1. Input data summary
    2. Feature selection + concordance scoring vs marker definitions
    3. Model performance (metric cards, per-class AUC/AP, confusion/ROC/PR)
    4. Prediction results/summary

Usage:
    generate_light_report.py \
        --input-metrics input_batch_metrics.json \
        --holdout-eval-dir . \
        --fs-dir . \
        --concordance-csv feature_concordance.csv \
        --counts-tsv all_cell_counts.tsv \
        --template-dir assets/html_templates \
        --report-name classyflow_report_light.html
"""

import argparse
import base64
import csv
import glob
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions (copied from generate_final_report.py — keep in sync)
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


def load_json(json_file: str) -> Dict:
    """Load a JSON file, returning {} on error."""
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
    """Load an HTML file as string, returning '' on error."""
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


def _resolve_div_path(referenced_path: str, search_dir: str) -> Optional[str]:
    """Resolve a Plotly div HTML path referenced inside a JSON file.

    The holdout JSON stores plain filenames; Nextflow stages those div files
    alongside the JSON, so we look in ``search_dir`` (falling back to the raw
    path).
    """
    if not referenced_path:
        return None
    candidate = os.path.join(search_dir, os.path.basename(referenced_path))
    if os.path.exists(candidate):
        return candidate
    if os.path.exists(referenced_path):
        return referenced_path
    logger.warning(f"Plotly div not found: {referenced_path}")
    return None


# =============================================================================
# Prediction Metrics & Visualization (copied from generate_final_report.py)
# =============================================================================

STACKED_BAR_SAMPLE_THRESHOLD = 50
XAXIS_LABEL_THRESHOLD = 25


def _get_cell_type_colors(cell_types) -> Dict[str, str]:
    """Generate consistent color mapping for cell types."""
    colors = sns.color_palette("Set2", len(cell_types)).as_hex()
    return dict(zip(cell_types, colors))


def compute_prediction_metrics(counts_df: pd.DataFrame) -> Dict:
    """Derive prediction metrics from aggregated counts DataFrame."""
    total_cells = counts_df.groupby('sample_name')['total_cells'].first().sum()
    total_low_density = counts_df.groupby('sample_name')['low_density_cells'].first().sum()

    global_counts = counts_df.groupby('cell_type')['count'].sum().sort_values(ascending=False)

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
        })

    return {
        'total_predicted_cells': int(total_cells),
        'total_low_density_cells': int(total_low_density),
        'most_common_prediction': global_counts.index[0] if len(global_counts) > 0 else None,
        'most_rare_prediction': global_counts.index[-1] if len(global_counts) > 0 else None,
        'samples': sorted(samples, key=lambda x: x['sample_name'])
    }


def create_abundance_plot(counts_df: pd.DataFrame, output_path: str) -> str:
    """Create adaptive abundance visualization based on sample count."""
    n_samples = counts_df['sample_name'].nunique()

    if n_samples <= STACKED_BAR_SAMPLE_THRESHOLD:
        html = _create_stacked_bar_per_sample(counts_df)
    else:
        html = _create_large_dataset_view(counts_df)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return html


def _create_stacked_bar_per_sample(counts_df: pd.DataFrame) -> str:
    """Create interactive stacked bar chart for small datasets."""
    pivot_df = counts_df.pivot(
        index='sample_name', columns='cell_type', values='count'
    ).fillna(0)
    pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100

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
# Section collectors
# =============================================================================

def collect_input_metrics(input_metrics_path: str) -> Dict:
    """Section 1: load the input batch metrics summary."""
    input_metrics = load_json(input_metrics_path)
    input_metrics['any_synthetic_warnings'] = any(
        b.get('percent_synthetic', 0) > 25 for b in input_metrics.get('batches', [])
    )
    return {'input_metrics': input_metrics}


def _score_class(auc_value: float) -> str:
    if auc_value >= 0.9:
        return 'score-excellent'
    if auc_value >= 0.8:
        return 'score-good'
    if auc_value >= 0.7:
        return 'score-moderate'
    return 'score-poor'


def collect_feature_selection(fs_dir: str, fs_files: Optional[List[str]]) -> Dict:
    """Section 2: read per-celltype feature_selection_*_results.json files."""
    if fs_files:
        json_files = sorted(f for f in fs_files if f.endswith('_results.json'))
    else:
        json_files = sorted(glob.glob(os.path.join(fs_dir, "feature_selection_*_results.json")))

    results = []
    all_selected = set()
    original_features = 0
    reduction_rates = []
    cv_folds_values = []

    for jf in json_files:
        data = load_json(jf)
        if not data:
            continue
        summary = data.get('feature_selection_summary', {})
        selected = data.get('selected_features', []) or []
        all_selected.update(selected)
        original_features = max(original_features, summary.get('original_features', 0))
        red = summary.get('reduction_percentage')
        if red is not None:
            reduction_rates.append(red)
        cv = data.get('cv_folds', data.get('n_folds'))
        if isinstance(cv, (int, float)):
            cv_folds_values.append(int(cv))

        results.append({
            'celltype': data.get('celltype', 'Unknown'),
            'optimal_n_features': data.get('optimal_n_features', 0),
            'non_variant_removed': summary.get('non_variant_removed'),
            'reduction_percentage': red,
            'cv_folds': cv if cv is not None else 'N/A',
            'selected_features': selected,
            'selected_features_count': len(selected),
            'rfe_warning': data.get('rfe_warning', False),
            'min_features_threshold': data.get('min_features_threshold'),
        })

    results.sort(key=lambda r: str(r['celltype']))

    if cv_folds_values:
        cv_summary = (f"{min(cv_folds_values)}-{max(cv_folds_values)}"
                      if min(cv_folds_values) != max(cv_folds_values)
                      else str(cv_folds_values[0]))
    else:
        cv_summary = "N/A"

    summary_stats = {
        'total_celltypes': len(results),
        'unique_original_features': int(original_features),
        'unique_selected_features': len(all_selected),
        'avg_reduction_rate': round(sum(reduction_rates) / len(reduction_rates), 1) if reduction_rates else 0,
        'cv_folds_summary': cv_summary,
    }

    return {
        'feature_selection': {
            'results': results,
            'summary': summary_stats,
        }
    }


def collect_concordance(concordance_csv: Optional[str]) -> Dict:
    """Section 2: read the concordance headline table (feature_concordance.csv)."""
    if not concordance_csv or not os.path.exists(concordance_csv):
        return {'concordance': {'available': False, 'rows': []}}

    rows = []
    try:
        with open(concordance_csv, newline='') as f:
            for row in csv.DictReader(f):
                conflicting = row.get('conflicting_markers', '') or ''
                rows.append({
                    'cf_class': row.get('cf_class', ''),
                    'self_cell_type': row.get('self_cell_type', ''),
                    'self_score': row.get('self_score', ''),
                    'best_match': row.get('best_match', ''),
                    'best_score': row.get('best_score', ''),
                    'relation': (row.get('relation', '') or '').strip(),
                    'shared_markers': row.get('shared_markers', ''),
                    'conflicting_markers': [m for m in conflicting.split('|') if m],
                    'note': row.get('note', ''),
                })
    except Exception as e:
        logger.error(f"Error reading concordance CSV {concordance_csv}: {e}")
        return {'concordance': {'available': False, 'rows': []}}

    rows.sort(key=lambda r: r['cf_class'])
    return {'concordance': {'available': True, 'rows': rows}}


def collect_model_performance(holdout_eval_dir: str, holdout_files: Optional[List[str]]) -> Dict:
    """Section 3: read holdout evaluation JSON for the best (First) model."""
    if holdout_files:
        json_files = sorted(f for f in holdout_files if f.endswith('_results.json'))
        search_dirs = {os.path.dirname(os.path.abspath(f)) or '.' for f in holdout_files}
    else:
        json_files = sorted(glob.glob(os.path.join(holdout_eval_dir, "holdoutEval_*_results.json")))
        search_dirs = {holdout_eval_dir}

    if not json_files:
        logger.warning("No holdout evaluation JSON files found.")
        return {'model_performance': None}

    def priority(path: str) -> int:
        name = os.path.basename(path)
        if 'First' in name:
            return 0
        if 'Second' in name:
            return 1
        return 2

    json_files.sort(key=priority)
    best_json = json_files[0]
    data = load_json(best_json)
    if not data:
        return {'model_performance': None}

    search_dir = os.path.dirname(os.path.abspath(best_json)) or '.'
    all_search_dirs = list(search_dirs) + [search_dir]

    def _load_div(referenced: str) -> str:
        for d in all_search_dirs:
            resolved = _resolve_div_path(referenced, d)
            if resolved:
                return load_html(resolved)
        return ""

    class_names = data.get('class_names', [])
    class_counts = data.get('class_counts', [])
    auc_lookup = {a['class_name']: a['auc'] for a in data.get('auc_scores', [])}
    ap_lookup = {a['class_name']: a['ap'] for a in data.get('ap_scores', [])}
    total = sum(class_counts) if class_counts else 0

    class_rows = []
    for i, name in enumerate(class_names):
        count = class_counts[i] if i < len(class_counts) else 0
        pct = (count / total * 100) if total > 0 else 0
        auc_val = auc_lookup.get(name)
        ap_val = ap_lookup.get(name)
        class_rows.append({
            'name': name,
            'count': count,
            'percentage': round(pct, 1),
            'auc': auc_val,
            'auc_class': _score_class(auc_val) if auc_val is not None else None,
            'ap': ap_val,
        })
    class_rows.sort(key=lambda r: (r['auc'] is not None, r['auc'] or 0), reverse=True)

    model_name = os.path.basename(best_json)
    if '_Model_' in model_name:
        model_name = model_name.split('_Model_')[1].split('_results')[0]

    performance = {
        'model_name': model_name,
        'accuracy': data.get('accuracy'),
        'f1_score': data.get('f1_score'),
        'n_classes': data.get('n_classes', len(class_names)),
        'total_samples': data.get('total_samples', total),
        'class_imbalance_detected': data.get('class_imbalance_detected', False),
        'best_performing_class': data.get('max_auc', {}).get('class_name'),
        'worst_performing_class': data.get('min_auc', {}).get('class_name'),
        'class_rows': class_rows,
        'confusion_html': _load_div(data.get('confusion_matrix_html_path', '')),
        'roc_curves_html': _load_div(data.get('roc_curves_plot_path', '')),
        'pr_curves_html': _load_div(data.get('pr_curves_plot_path', '')),
    }
    return {'model_performance': performance}


def collect_predictions(counts_tsv: str, plots_dir: str) -> Dict:
    """Section 4: prediction metrics + abundance plot from aggregated counts."""
    counts_df = load_tsv(counts_tsv)
    if counts_df.empty:
        return {'prediction_metrics': {}}

    metrics = compute_prediction_metrics(counts_df)
    abundance_plot_path = os.path.join(plots_dir, "light_prediction_abundance_plot.html")
    metrics['abundance_plot_html'] = create_abundance_plot(counts_df, abundance_plot_path)
    return {'prediction_metrics': metrics}


def build_provenance(args) -> Dict:
    """Assemble the provenance block context."""
    input_dirs = []
    if args.input_dirs:
        input_dirs = [d.strip() for d in args.input_dirs.split(',') if d.strip()]

    exclude_markers = []
    if args.exclude_markers:
        exclude_markers = [m.strip() for m in args.exclude_markers.split('|') if m.strip()]

    config_link = None
    if args.config_file:
        config_link = os.path.basename(args.config_file)

    return {
        'provenance': {
            'pipeline_version': args.version,
            'run_date': datetime.now().strftime("%B %d, %Y %H:%M"),
            'input_dirs': input_dirs,
            'normalization': args.normalization or 'auto',
            'holdout_fraction': args.holdout_fraction,
            'minimum_label_count': args.min_label_count,
            'exclude_markers': exclude_markers,
            'config_link': config_link,
        }
    }


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(all_data: Dict, output_file: str, jinja_env: Environment,
                    letterhead: Optional[str], pipeline_version: str,
                    template_name: str = "light_base.html"):
    """Render and write the light HTML report."""
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

    logger.info(f"Light report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a lightweight HTML report from pipeline outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--input-metrics', required=True,
                        help='Path to input_batch_metrics.json')
    parser.add_argument('--holdout-eval-dir', default='.',
                        help='Directory containing holdoutEval_*_results.json + div HTMLs')
    parser.add_argument('--holdout-eval-files', nargs='*',
                        help='Explicit holdout eval files (JSON + div HTMLs); overrides --holdout-eval-dir')
    parser.add_argument('--fs-dir', default='.',
                        help='Directory containing feature_selection_*_results.json')
    parser.add_argument('--fs-files', nargs='*',
                        help='Explicit feature-selection files; overrides --fs-dir')
    parser.add_argument('--concordance-csv', default=None,
                        help='Optional feature_concordance.csv (headline table)')
    parser.add_argument('--concordance-json', default=None,
                        help='Optional feature_concordance.json (detail; currently unused)')
    parser.add_argument('--counts-tsv', required=True,
                        help='Path to aggregated cell counts TSV')

    parser.add_argument('--plots-dir', default='.',
                        help='Directory for generated plot files')
    parser.add_argument('--template-dir', default='templates',
                        help='Directory containing Jinja2 templates')
    parser.add_argument('--report-name', default='classyflow_report_light.html',
                        help='Output report filename')

    parser.add_argument('--letterhead', help='Path to header logo image (embedded)')
    parser.add_argument('--version', default='N/A', help='Pipeline version')

    # Provenance parameters
    parser.add_argument('--input-dirs', default='', help='Comma-separated input directories')
    parser.add_argument('--normalization', default='', help='Normalization method used')
    parser.add_argument('--holdout-fraction', default='N/A', help='Holdout fraction')
    parser.add_argument('--min-label-count', default='N/A', help='Minimum label count')
    parser.add_argument('--exclude-markers', default='', help='Pipe-delimited excluded markers')
    parser.add_argument('--config-file', default=None, help='Path to nextflow.config')

    args = parser.parse_args()

    jinja_env = setup_jinja_environment(args.template_dir)

    all_data: Dict[str, Any] = {}
    all_data.update(collect_input_metrics(args.input_metrics))
    all_data.update(collect_feature_selection(args.fs_dir, args.fs_files))
    all_data.update(collect_concordance(args.concordance_csv))
    all_data.update(collect_model_performance(args.holdout_eval_dir, args.holdout_eval_files))
    all_data.update(collect_predictions(args.counts_tsv, args.plots_dir))
    all_data.update(build_provenance(args))

    generate_report(
        all_data=all_data,
        output_file=args.report_name,
        jinja_env=jinja_env,
        letterhead=args.letterhead,
        pipeline_version=args.version
    )


if __name__ == "__main__":
    main()
