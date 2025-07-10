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

def parse_args():
    parser = argparse.ArgumentParser(description="Generate HTML report for cell type predictions.")
    parser.add_argument('--input_tsv', required=True, help='Input prediction TSV file')
    parser.add_argument('--output_html', required=True, help='Output HTML report file')
    return parser.parse_args()

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
    fig, ax = plt.subplots(figsize=(12, 8))
    
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

    results = {
        'sample_name': slide_name, 
        'celltype_barplot': f"{slide_name}_celltype_barplot.png",
        'spatial_plot': f"{slide_name}_spatial_plot.html"
    }

    # Generate plots
    spatial_html = plot_spatial(df, color_map, slide_name, results["spatial_plot"])
    umap_html = plot_umap(df, color_map)
    dendro_html = plot_dendrogram(df)
    create_cell_type_bar_plot(df, slide_name, results["celltype_barplot"])

    #Save json
    with open(f"{slide_name}_classified.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)


    # Compose HTML report
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cell Type Classification Report: {{ slide_name }}</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 30px; }
            h1 { color: #333; }
            .plotly-graph-div { margin-bottom: 40px; }
        </style>
    </head>
    <body>
        <h1>Cell Type Classification Report: {{ slide_name }}</h1>
        <h2>Spatial Cell Type Prediction</h2>
        {{ spatial_html|safe }}
        <h2>UMAP of Quantification Data</h2>
        {{ umap_html|safe }}
        <h2>Hierarchical Clustering Dendrogram</h2>
        {{ dendro_html|safe }}
    </body>
    </html>
    """
    template = Template(html_template)
    html = template.render(
        slide_name=slide_name,
        spatial_html=spatial_html,
        umap_html=umap_html,
        dendro_html=dendro_html
    )
    with open(args.output_html, 'w') as f:
        f.write(html)

if __name__ == "__main__":
    main()

