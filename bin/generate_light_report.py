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
    2. Feature selection: per-celltype selected-feature bar chart, plus
       concordance against the marker definitions when a profile is supplied
    3. Model performance (metric cards, per-class AUC/AP, confusion/ROC/PR)
    4. Prediction results/summary

Feature importance and direction are NOT computed here. They are read from each
feature_selection_<celltype>_results.json's ``feature_importance`` list, written
by rank_selected_features() in bin/generate_cell_type_selection.py. That makes
the bar chart independent of whether a cell-type profile was supplied.

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
import yaml
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
    """Derive the prediction headline metrics from the aggregated counts."""
    total_cells = counts_df.groupby('sample_name')['total_cells'].first().sum()
    global_counts = counts_df.groupby('cell_type')['count'].sum().sort_values(ascending=False)

    return {
        'total_predicted_cells': int(total_cells),
        'most_common_prediction': global_counts.index[0] if len(global_counts) > 0 else None,
        'most_rare_prediction': global_counts.index[-1] if len(global_counts) > 0 else None,
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


# --- Feature-selection helpers -----------------------------------------------

MAX_FEATURES_SHOWN = 15   # bars per cell type in the feature-selection chart


def _bar_color(t: float, direction: Optional[str]) -> str:
    """Importance ramp for a marker bar, hue chosen by association direction.

    Positive markers use the blue ramp, negative-association markers a warm
    (orange) ramp, so the two are distinguishable at a glance while still
    encoding importance through shade. A marker with no direction (its
    coefficient was exactly zero, so it carries no sign) gets a neutral grey
    ramp rather than borrowing the positive hue.
    """
    t = max(0.0, min(1.0, t))
    d = str(direction or '').lower()
    if d == 'negative':
        lo, hi = (255, 247, 237), (154, 52, 18)   # #fff7ed -> #9a3412 (orange)
    elif d == 'positive':
        lo, hi = (239, 246, 255), (30, 58, 138)   # #eff6ff -> #1e3a8a (blue)
    else:
        lo, hi = (249, 250, 251), (107, 114, 128) # #f9fafb -> #6b7280 (grey)
    r = round(lo[0] + (hi[0] - lo[0]) * t)
    g = round(lo[1] + (hi[1] - lo[1]) * t)
    b = round(lo[2] + (hi[2] - lo[2]) * t)
    return f'#{r:02x}{g:02x}{b:02x}'


def _feature_bars(feature_importance: List[Dict],
                  limit: int = MAX_FEATURES_SHOWN) -> List[Dict]:
    """Turn the upstream ``feature_importance`` list into bar records.

    Presentation only. The importance, the signed direction and the ranking all
    come from rank_selected_features() in
    bin/generate_cell_type_selection.py; nothing is re-derived here.
    """
    shown = feature_importance[:limit]
    max_imp = shown[0].get('importance', 0.0) if shown else 0.0
    bars = []
    for f in shown:
        imp = f.get('importance', 0.0) or 0.0
        t = (imp / max_imp) if max_imp > 0 else 0.0
        bars.append({
            'feature': f.get('feature', ''),
            'marker': f.get('marker', ''),
            'direction': f.get('direction'),
            'importance': imp,
            'intensity': round(t, 3),
            'height_pct': round(max(t * 100, 8), 1),  # floor so weak bars stay visible
            'bg': _bar_color(t, f.get('direction')),
        })
    return bars


def _relation_class(relation: str) -> str:
    """Map a concordance relation to a colour class.

    * match     (green)  -> best match IS the cell type's own definition.
    * related   (yellow) -> best match is a nearby lineage (ancestor/descendant/sibling).
    * undefined (grey)   -> the cell type has no definition in the profile, so
                            nothing could be checked. Not a finding.
    * mismatch  (red)    -> best match is a genuinely unrelated definition.
    """
    r = str(relation or '').lower()
    if r == 'self':
        return 'match'
    if r in ('ancestor', 'descendant', 'sibling'):
        return 'related'
    if r in ('no_self_def', 'no_match', ''):
        return 'undefined'
    return 'mismatch'


def _verdict(relation_class: str, celltype: str, best_match: str,
             n_matched: int, n_expected: int) -> str:
    """One plain-language sentence per cell type, for a non-technical reader."""
    found = f"{n_matched} of {n_expected} expected markers found"
    if relation_class == 'match':
        return f"Markers match the expected {celltype} definition ({found})."
    if relation_class == 'related':
        return (f"Markers point to {best_match}, a closely related cell type "
                f"({found}).")
    if relation_class == 'undefined':
        return ("No definition was provided for this cell type, so its markers "
                "could not be checked.")
    return (f"Markers point to {best_match}, which is not related to {celltype} "
            f"— worth checking the training labels.")


def _hierarchy_breadcrumb(name: str, nodes: Dict[str, Dict]) -> List[str]:
    """Root-to-node lineage path for a profile node (e.g. Epithelial > Basal)."""
    if not name or name not in nodes:
        return []
    chain, cur, guard = [], name, 0
    while cur and guard < 100:
        chain.append(cur)
        cur = nodes.get(cur, {}).get('parent')
        guard += 1
    return list(reversed(chain))


def load_celltype_profile(profile_path: Optional[str]) -> Dict:
    """Parse the cell-type profile YAML into nodes + a rooted tree.

    Returns {'nodes': {name: {name, parent, markers:[{marker,state}], sig:set}},
             'roots': [tree_node]} or {} when no usable profile is given.
    """
    if not profile_path or not os.path.exists(profile_path):
        return {}
    try:
        with open(profile_path) as f:
            profile = yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Error loading celltype profile {profile_path}: {e}")
        return {}

    nodes: Dict[str, Dict] = {}
    for t in profile.get("cell_types", []) or []:
        name = t.get("name")
        if not name:
            continue
        markers = []
        sig = set()
        for marker, state in (t.get("markers") or {}).items():
            st = str(state).strip().lower()
            if st in ("positive", "negative"):
                markers.append({'marker': marker, 'state': st})
                sig.add(marker)
        nodes[name] = {'name': name, 'parent': t.get('parent') or None,
                       'markers': markers, 'sig': sig}
    if not nodes:
        return {}

    # Build the rooted tree (children lists) for the graphical view.
    children: Dict[Optional[str], List[str]] = {}
    for n in nodes.values():
        children.setdefault(n['parent'], []).append(n['name'])

    def _build(name: str) -> Dict:
        node = nodes[name]
        return {'name': name, 'markers': node['markers'],
                'children': [_build(c) for c in sorted(children.get(name, []))]}

    roots = [_build(r) for r in sorted(children.get(None, []))]
    return {'nodes': nodes, 'roots': roots}


def _read_concordance_rows(concordance_csv: Optional[str]) -> Dict[str, Dict]:
    """Index concordance CSV rows by cf_class.

    ``cf_class`` is written straight from the FS results JSON's ``celltype``
    field, which is also what this report keys on, so the join is exact.
    """
    if not concordance_csv or not os.path.exists(concordance_csv):
        return {}
    indexed = {}
    try:
        with open(concordance_csv, newline='') as f:
            for row in csv.DictReader(f):
                conflicting = row.get('conflicting_markers', '') or ''
                indexed[row.get('cf_class', '')] = {
                    'cf_class': row.get('cf_class', ''),
                    'self_cell_type': row.get('self_cell_type', ''),
                    'self_score': row.get('self_score', ''),
                    'best_match': (row.get('best_match', '') or '').strip(),
                    'best_score': row.get('best_score', ''),
                    'relation': (row.get('relation', '') or '').strip(),
                    'shared_markers': row.get('shared_markers', ''),
                    'conflicting_markers': [m for m in conflicting.split('|') if m],
                    'note': row.get('note', ''),
                }
    except Exception as e:
        logger.error(f"Error reading concordance CSV {concordance_csv}: {e}")
    return indexed


def build_feature_section(fs_dir: str, fs_files: Optional[List[str]],
                          concordance_csv: Optional[str],
                          profile_path: Optional[str]) -> Dict:
    """Assemble the Feature Selection section from the per-celltype FS results.

    The selected-feature bar chart is always built: per-feature importance and
    direction come from the FS results JSON (``feature_importance``), which does
    not depend on a cell-type profile. When a profile *is* supplied, each cell
    type additionally gains its best-matching definition, a plain-language
    verdict, and the profile tree is rendered.
    """
    if fs_files:
        json_files = sorted(f for f in fs_files if f.endswith('_results.json'))
    else:
        json_files = sorted(glob.glob(os.path.join(fs_dir, "feature_selection_*_results.json")))

    concordance = _read_concordance_rows(concordance_csv)
    profile = load_celltype_profile(profile_path)
    profile_nodes = profile.get('nodes', {})
    has_definitions = bool(concordance)
    if profile_nodes and not has_definitions:
        logger.warning("A cell-type profile was supplied but no concordance rows "
                       "were found; rendering marker charts without definitions.")

    celltypes = []
    for jf in json_files:
        data = load_json(jf)
        if not data:
            continue
        celltype = data.get('celltype', 'Unknown')
        feature_importance = data.get('feature_importance', []) or []

        record = {
            'celltype': celltype,
            'rfe_warning': data.get('rfe_warning', False),
            'feature_bars': _feature_bars(feature_importance),
            'n_features_total': len(feature_importance),
            'n_features_shown': min(len(feature_importance), MAX_FEATURES_SHOWN),
        }

        # Both sides read `celltype` from the same JSON field, so match exactly.
        conc = concordance.get(celltype)
        if conc:
            # Marker-level match against this cell type's definition: prefer its
            # own profile node, else fall back to the best-matching definition.
            def_name = conc['self_cell_type'] if conc['self_cell_type'] in profile_nodes \
                else conc['best_match']
            def_node = profile_nodes.get(def_name, {})
            selected_markers = {f['marker'] for f in feature_importance}
            conflicting = set(conc['conflicting_markers'])
            marker_matches = []
            for m in def_node.get('markers', []):
                marker = m['marker']
                if marker in conflicting:
                    status = 'conflict'
                elif marker in selected_markers:
                    status = 'matched'
                else:
                    status = 'missing'
                marker_matches.append({'marker': marker, 'state': m['state'],
                                       'status': status})
            marker_matches.sort(key=lambda x: ({'matched': 0, 'conflict': 1,
                                                'missing': 2}[x['status']], x['marker']))
            n_matched = sum(1 for m in marker_matches if m['status'] == 'matched')
            relation_class = _relation_class(conc['relation'])

            record.update({
                'best_match': conc['best_match'],
                'best_score': conc['best_score'],
                'relation': conc['relation'],
                'relation_class': relation_class,
                'verdict': _verdict(relation_class, celltype, conc['best_match'],
                                    n_matched, len(marker_matches)),
                'conflicting_markers': conc['conflicting_markers'],
                'shared_marker_count': conc['shared_markers'],
                'hierarchy_breadcrumb': _hierarchy_breadcrumb(conc['best_match'], profile_nodes),
                'marker_matches': marker_matches,
                'n_matched': n_matched,
                'n_conflict': sum(1 for m in marker_matches if m['status'] == 'conflict'),
                'n_missing': sum(1 for m in marker_matches if m['status'] == 'missing'),
                'n_expected': len(marker_matches),
            })
        elif has_definitions:
            # Definitions were supplied, but none cover this cell type. That is
            # not a finding — say so explicitly rather than leaving it to a
            # default that would read as a mismatch.
            record.update({
                'relation_class': 'undefined',
                'verdict': _verdict('undefined', celltype, '', 0, 0),
            })
        celltypes.append(record)

    celltypes.sort(key=lambda c: str(c['celltype']))

    overview = None
    if has_definitions:
        overview = {'match': 0, 'related': 0, 'mismatch': 0, 'undefined': 0}
        for c in celltypes:
            # No concordance row at all means undefined, not a mismatch.
            overview[c.get('relation_class', 'undefined')] += 1
        overview['total'] = len(celltypes)

    return {
        'feature_selection': {
            'has_definitions': has_definitions,
            'celltypes': celltypes,
            'tree': profile.get('roots'),
            'overview': overview,
            'max_features_shown': MAX_FEATURES_SHOWN,
        }
    }


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

    return {
        'provenance': {
            'pipeline_version': args.version,
            'run_date': datetime.now().strftime("%B %d, %Y %H:%M"),
            'input_dirs': input_dirs,
            'normalization': args.normalization or 'auto',
            'holdout_fraction': args.holdout_fraction,
            'minimum_label_count': args.min_label_count,
            'exclude_markers': exclude_markers,
            # Plain filename, not a link: the light report is published as a
            # single standalone file with no sibling pages/ directory.
            'config_name': os.path.basename(args.config_file) if args.config_file else None,
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
    parser.add_argument('--celltype-profile', default=None,
                        help='Optional cell-type profile YAML (enables profile mode + tree)')
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
    all_data.update(build_feature_section(args.fs_dir, args.fs_files,
                                          args.concordance_csv, args.celltype_profile))
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
