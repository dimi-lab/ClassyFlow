#!/usr/bin/env python3

import argparse
import os
import sys
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from jinja2 import Template
import base64
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Generate HTML report for cell type predictions.")
    parser.add_argument('--input_tsv', required=True, help='Input prediction TSV file')
    parser.add_argument('--output_html', required=True, help='Output HTML report file')
    parser.add_argument('--batch', default='', help='Batch ID for this sample')
    return parser.parse_args()


def encode_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 encoded string for embedding."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            # Determine MIME type based on file extension
            ext = Path(image_path).suffix.lower()
            if ext in ['.png']:
                mime_type = 'image/png'
            elif ext in ['.jpg', '.jpeg']:
                mime_type = 'image/jpeg'
            elif ext in ['.svg']:
                mime_type = 'image/svg+xml'
            else:
                mime_type = 'image/png'  # Default
            
            return f"data:{mime_type};base64,{encoded_string}"
        
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return ""
    
def get_color_map(cell_types):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import to_hex
    from matplotlib import rcParams
    import seaborn as sns

    # Use seaborn color palette for distinct colors
    unique_types = sorted(cell_types)
    n_types = len(unique_types)
    palette = sns.color_palette("Dark2", n_types).as_hex()
    color_map = dict(zip(unique_types, palette))

    # Override specific cell types with custom colors
    overrides = {
        "B Cell": "#0000ff",
        "CytoT": "#40cb80",
        "HelperT": "#de8f0d",
        "Tumor": "#7e1104"
    }
    for k, v in overrides.items():
        if k in color_map:
            color_map[k] = v
    return color_map

def calculate_roi_summary_stats(df):
    """Calculate summary statistics for the ROI table"""
    cell_type_counts = df['CellTypePrediction'].value_counts()
    cell_type_percentages = df['CellTypePrediction'].value_counts(normalize=True) * 100
    cell_type_percentages = cell_type_percentages.round(1)
    
    sample_low_density = 0
    if 'low_bin_density' in df.columns:
        sample_low_density = int(df['low_bin_density'].sum())

    summary_stats = {
        'total_cells': len(df),
        'unique_classes': df['CellTypePrediction'].nunique(),
        'most_common_class': cell_type_counts.index[0] if len(cell_type_counts) > 0 else None,
        'most_common_percentage': cell_type_percentages.iloc[0] if len(cell_type_percentages) > 0 else None,
        'second_common_class': cell_type_counts.index[1] if len(cell_type_counts) > 1 else None,
        'second_common_percentage': cell_type_percentages.iloc[1] if len(cell_type_percentages) > 1 else None,
        'least_common_class': cell_type_counts.index[-1] if len(cell_type_counts) > 0 else None,
        'least_common_percentage': cell_type_percentages.iloc[-1] if len(cell_type_percentages) > 2 else None,
        'cell_type_distribution': {k: int(v) for k, v in cell_type_counts.items()},
        'percentage_distribution': {k: float(v) for k, v in cell_type_percentages.items()},
        'low_density_cells': sample_low_density
    }
    
    return summary_stats

def write_counts_tsv(df, sample_name, summary_stats, output_file, batch_id=None):
    """Write cell type counts in long format for collectFile aggregation"""
    cell_counts = df['CellTypePrediction'].value_counts()
    
    rows = []
    for cell_type, count in cell_counts.items():
        rows.append({
            'sample_name': sample_name,
            'batch': batch_id or '',
            'cell_type': cell_type,
            'count': count,
            'total_cells': summary_stats['total_cells'],
            'low_density_cells': summary_stats['low_density_cells'],
            'roi_report': f"{sample_name}.html"
        })
    
    counts_df = pd.DataFrame(rows)
    counts_df.to_csv(output_file, sep='\t', index=False)

def plot_spatial(df, color_map, slide_name, output_file):
    mx = df["Centroid Y µm"].max() + 1
    df["invertY"] = mx - df["Centroid Y µm"]
    fig = px.scatter(
        df, x="Centroid X µm", y="invertY",
        color="CellTypePrediction",
        color_discrete_map=color_map,
        opacity=0.8,
        title=f"{slide_name} [{len(df)} cells]",
        width=700, height=600
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(
        xaxis_title=None, yaxis_title=None,
        showlegend=False,
        template="simple_white"
    )

    fig.write_html(
        output_file,
        include_plotlyjs='cdn'
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def plot_umap(df, color_map):
    # Select numeric features only, drop metadata columns
    feature_cols = [col for col in df.columns if col not in ["Slide", "CellTypePrediction", "Centroid X µm", "Centroid Y µm", "invertY"] and pd.api.types.is_numeric_dtype(df[col])]
    if len(feature_cols) < 2:
        return ""
    features = df[feature_cols].fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    # PCA to 90% variance
    pca = PCA(n_components=min(30, scaled.shape[1]))
    pca_matrix = pca.fit_transform(scaled)
    explained = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(explained >= 0.9) + 1
    pca_matrix = pca_matrix[:, :n_components]
    # UMAP
    reducer = umap.UMAP(random_state=42)
    umap_matrix = reducer.fit_transform(pca_matrix)
    umap_df = pd.DataFrame(umap_matrix, columns=["UMAP1", "UMAP2"])
    umap_df["CellTypePrediction"] = df["CellTypePrediction"].values
    fig = px.scatter(
        umap_df, x="UMAP1", y="UMAP2", color="CellTypePrediction",
        color_discrete_map=color_map,
        opacity=0.3,
        title="UMAP of PCA-Reduced Quantification Data"
    )
    fig.update_layout(template="simple_white")
    return fig.to_html(full_html=False, include_plotlyjs=False)

def plot_dendrogram(df):
    # Sample up to 1000 rows or 10%
    feature_cols = [col for col in df.columns if col not in ["Slide", "CellTypePrediction", "Centroid X µm", "Centroid Y µm", "invertY"] and pd.api.types.is_numeric_dtype(df[col])]
    if len(feature_cols) < 2:
        return ""
    sample = df.sample(n=min(1000, int(0.1*len(df))), random_state=42) if len(df) > 1000 else df
    features = sample[feature_cols].fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    # PCA to 90% variance
    pca = PCA(n_components=min(30, scaled.shape[1]))
    pca_matrix = pca.fit_transform(scaled)
    explained = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(explained >= 0.9) + 1
    pca_matrix = pca_matrix[:, :n_components]
    # Dendrogram
    dist_matrix = pdist(pca_matrix)
    linkage_matrix = linkage(dist_matrix, method='ward')
    fig = ff.create_dendrogram(pca_matrix, orientation='top', labels=None, color_threshold=None)
    fig.update_layout(title="Hierarchical Clustering Dendrogram", width=800, height=400)
    return fig.to_html(full_html=False, include_plotlyjs=False)

def create_cell_type_bar_plot(df, sample_name, output_file):
    # Get cell type counts
    cell_counts = df['CellTypePrediction'].value_counts()
    cell_percentages = df['CellTypePrediction'].value_counts(normalize=True) * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create horizontal bar plot
    y_pos = np.arange(len(cell_counts))
    colors = plt.cm.get_cmap('Set2')(np.linspace(0, 1, len(cell_counts)))
    
    bars = ax.barh(y_pos, cell_counts.values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize the plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cell_counts.index, fontsize=11)
    ax.set_xlabel('Cell Count', fontsize=12, fontweight='bold')
    ax.set_title(f'Cell Type Distribution - {sample_name}\nTotal Cells: {len(df):,}', 
                fontsize=14, fontweight='bold', pad=20)
    

    for i, (bar, count, pct) in enumerate(zip(bars, cell_counts.values, cell_percentages.values)):
        # Position label at end of bar
        label_x = bar.get_width() + max(cell_counts) * 0.01
        
        # Create label text
        label_parts = []
        label_parts.append(f'{count:,}')
        label_parts.append(f'({pct:.1f}%)')
        
        label_text = ' '.join(label_parts)
        
        ax.text(label_x, bar.get_y() + bar.get_height()/2, label_text,
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    # Add grid for easier reading
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor='white')

def main():
    args = parse_args()
    df = pd.read_csv(args.input_tsv, sep='\t')
    slide_name = os.path.basename(args.input_tsv).split(".")[0]
    color_map = get_color_map(df["CellTypePrediction"].unique())

    # Calculate summary statistics for the ROI table
    summary_stats = calculate_roi_summary_stats(df)

    # Write counts TSV for aggregation
    counts_output = f"{slide_name}_counts.tsv"
    write_counts_tsv(df, slide_name, summary_stats, counts_output, batch_id=args.batch)

    results = {
        'sample_name': slide_name, 
        'celltype_barplot': f"{slide_name}_celltype_barplot.png",
        'spatial_plot': f"{slide_name}_spatial_plot.html",
        'roi_report': args.output_html
    }
    
    # Add summary statistics to results
    results.update(summary_stats)

    # Generate plots
    spatial_html = plot_spatial(df, color_map, slide_name, results["spatial_plot"])
    umap_html = plot_umap(df, color_map)
    dendro_html = plot_dendrogram(df)
    create_cell_type_bar_plot(df, slide_name, results["celltype_barplot"])

    #Save json with summary statistics
    with open(f"{slide_name}_classified.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Compose HTML report
    html_template = generate_roi_html_template()
    template = Template(html_template)
    template.globals['number_format'] = lambda x: f"{x:,}" if isinstance(x, (int, float)) else str(x)
    html = template.render(
        slide_name=slide_name,
        total_cells=summary_stats['total_cells'],
        unique_classes=summary_stats['unique_classes'],
        most_common_class=summary_stats['most_common_class'],
        most_common_percentage=f"{summary_stats['most_common_percentage']:.1f}" if summary_stats['most_common_percentage'] else "N/A",
        least_common_class=summary_stats['least_common_class'],
        least_common_percentage=f"{summary_stats['least_common_percentage']:.1f}" if summary_stats['least_common_percentage'] else "N/A",
        second_common_class=summary_stats['second_common_class'],
        second_common_percentage=f"{summary_stats['second_common_percentage']:.1f}" if summary_stats['second_common_percentage'] else "N/A",
        barplot_image=encode_image_to_base64(results["celltype_barplot"]),
        spatial_html=spatial_html,
        umap_html=umap_html,
        dendro_html=dendro_html
    )
    with open(args.output_html, 'w') as f:
        f.write(html)




def generate_roi_html_template() -> str:
    """Generate the HTML template for individual ROI pages."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ roi_name }} - ROI Detail Report</title>
    <style>
        /* Reset and base styles */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }

        /* Header */
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px 12px 0 0;
            text-align: center;
            margin-bottom: 0;
        }

        .header h1 {
            font-size: 2.5rem;
            font-weight: 300;
            margin-bottom: 0.5rem;
        }

        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        /* Navigation */
        .nav-back {
            background: white;
            padding: 1rem 2rem;
            border-radius: 0 0 12px 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }

        .nav-back a {
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
            font-size: 1rem;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.3s ease;
        }

        .nav-back a:hover {
            color: #5a6fd8;
            transform: translateX(-2px);
        }

        /* Main content */
        .content-container {
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
            overflow: hidden;
            margin-bottom: 2rem;
        }

        .section {
            padding: 2rem;
            border-bottom: 1px solid #e9ecef;
        }

        .section:last-child {
            border-bottom: none;
        }

        .section-title {
            font-size: 1.8rem;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #667eea;
        }

        /* Metrics grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin: 1.5rem 0;
        }

        .metric-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 0.5rem;
        }

        .metric-label {
            font-size: 0.9rem;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }

        /* Cell type distribution */
        .celltype-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }

        .celltype-card {
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            transition: all 0.3s ease;
        }

        .celltype-card:hover {
            border-color: #667eea;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
        }

        .celltype-name {
            font-weight: 600;
            color: #495057;
            margin-bottom: 0.5rem;
        }

        .celltype-count {
            font-size: 1.2rem;
            font-weight: 700;
            color: #28a745;
            margin-bottom: 0.25rem;
        }

        .celltype-percentage {
            font-size: 0.9rem;
            color: #6c757d;
        }

        /* Plot containers */
        .plot-container {
            text-align: center;
            margin: 2rem 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }

        .plot-container img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }

        .plot-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #495057;
            padding: 1rem;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
        }

        .spatial-container {
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }

        .spatial-container iframe {
            width: 100%;
            height: 600px;
            border: none;
        }

        /* Alert styles */
        .alert {
            padding: 1rem 1.5rem;
            margin: 1rem 0;
            border-radius: 8px;
            border-left: 4px solid;
        }

        .alert-info {
            background-color: #d1ecf1;
            border-color: #17a2b8;
            color: #0c5460;
        }

        .alert-warning {
            background-color: #fff3cd;
            border-color: #ffc107;
            color: #856404;
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem;
            color: #6c757d;
            font-size: 0.9rem;
        }

        /* Responsive design */
        @media (max-width: 768px) {
            body {
                padding: 1rem;
            }

            .header h1 {
                font-size: 2rem;
            }

            .metrics-grid {
                grid-template-columns: 1fr;
            }

            .section {
                padding: 1.5rem;
            }
        }
    </style>
</head>
<body>
    <!-- Header -->
    <div class="header">
        <h1>{{ roi_name }}</h1>
        <p>Detailed Cell Type Analysis Report</p>
    </div>
    <!-- Main Content -->
    <div class="content-container">
        <!-- ROI Overview -->
        <div class="section">
            <h2 class="section-title">ROI Overview</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{{ number_format(total_cells) }}</div>
                    <div class="metric-label">Total Cells</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ unique_classes }}</div>
                    <div class="metric-label">Unique Cell Types</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ most_common_class }}</div>
                    <div class="metric-label">Most Common Type</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ least_common_class }}</div>
                    <div class="metric-label">Least Common Type</div>
                </div>
            </div>

            <div class="alert alert-info">
                <strong>Key Statistics:</strong>
                Most abundant cell type is <strong>{{ most_common_class }}</strong> at {{ most_common_percentage }}%. 
                {% if second_common_class != 'N/A' %}
                Second most common is <strong>{{ second_common_class }}</strong> at {{ second_common_percentage }}%.
                {% endif %}
                {% if least_common_class != 'N/A' %}
                Least common type is <strong>{{ least_common_class }}</strong> at {{ least_common_percentage }}%.
                {% endif %}
            </div>
        </div>

        <!-- Plots Section -->
        <div class="section">
            <h2 class="section-title">Plots</h2>
            {% if barplot_image %}
            <div class="plot-container">
                <div class="plot-title">Cell Type Distribution Bar Plot</div>
                <img src="{{ barplot_image }}" alt="Cell Type Distribution" />
            </div>
            {% endif %}

            {% if spatial_html %}
            <div class="spatial-container">
                <div class="plot-title">Interactive Spatial Visualization</div>
                {{ spatial_html | safe }}
            </div>
            {% endif %}

            {% if umap_html %}
            <div class="spatial-container">
                <div class="plot-title">Interactive Spatial Visualization</div>
                {{ umap_html | safe }}
            </div>
            {% endif %}

            {% if dendro_html %}
            <div class="spatial-container">
                <div class="plot-title">Interactive Spatial Visualization</div>
                {{ dendro_html | safe }}
            </div>
            {% endif %}
        </div>

        {% if not barplot_image and not spatial_html and not umap_html and not dendro_html %}
        <div class="section">
            <div class="alert alert-warning">
                <strong>Note:</strong> No visualization plots are available for this ROI.
            </div>
        </div>
        {% endif %}
    </div>

    <!-- Footer -->
    <div class="footer">
        <p>Generated by ClassyFlow Pipeline | ROI Detail Report</p>
    </div>
</body>
</html>
"""

if __name__ == "__main__":
    main()