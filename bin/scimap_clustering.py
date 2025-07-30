#!/usr/bin/env python3
# Load necessary libraries
import sys
import os
import argparse
import anndata as ad
import pandas as pd
import scanpy as sc
import scimap as sm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from jinja2 import Template

sc.settings.figdir = "./"


sc.settings.figdir = "./"


def find_best_coord(colnames, axis):
    """
    Find the best-fit column for X or Y coordinates using regex patterns.
    """
    import re
    patterns = [fr'Centroid {axis} um', fr'Centroid {axis}', fr'{axis}']
    for pat in patterns:
        matches = [c for c in colnames if re.search(pat, c, re.IGNORECASE)]
        if matches:
            return sorted(matches, key=len)[0]
    raise ValueError(f"No column found for {axis} coordinate.")

def load_and_filter_data(input_tsv, qupath_object_type):
    """
    Load the input TSV and filter out artifacts and DAPI columns. Return filtered DataFrame and marker columns.
    """
    df = pd.read_csv(input_tsv, sep='\t')
    if 'qc' in df.columns:
        df = df[~df['qc'].astype(str).str.contains("Artifact", na=False)]
    # Replace '0' in Classification column with blank
    class_col = 'Classification'
    if class_col in df.columns:
        df[class_col] = df[class_col].replace('0', '')
    # Select marker columns based on qupath_object_type
    if qupath_object_type == "CellObject":
        marker_regex = r'(Cell: Median)'
        marker_replace = ": Cell: Median"
    else:
        marker_regex = r'(Median)'
        marker_replace = "Median"
    marker_cols = df.filter(regex=marker_regex, axis=1)
    marker_cols.columns = marker_cols.columns.str.replace(marker_replace, "")
    marker_cols = marker_cols.filter(regex='^((?!DAPI).)*', axis=1)
    if marker_cols.shape[1] == 0:
        raise ValueError("No marker columns found after filtering. Check your input TSV for columns matching '(Cell: Median)' and not containing 'DAPI'.")
    return df, marker_cols

def create_anndata(marker_cols, df):
    """
    Create an AnnData object from marker columns and add spatial and metadata.
    """
    colnames = df.columns.tolist()
    col_x = find_best_coord(colnames, 'X')
    col_y = find_best_coord(colnames, 'Y')
    adata = ad.AnnData(marker_cols)
    adata.var_names = marker_cols.columns.to_list()
    adata.obsm = {"spatial": df[[col_x, col_y]].to_numpy()}
    adata.obs["imageid"] = pd.Categorical(df["Image"])
    adata.obs["X_centroid"] = df[[col_x]].to_numpy()
    adata.obs["Y_centroid"] = df[[col_y]].to_numpy()
    return adata

def run_umap_leiden(adata, clustering_res, roi):
    """
    Run UMAP and Leiden clustering, plot and save UMAP and matrixplot, and spatial scatter plot.
    """
    sc.pp.neighbors(adata, n_neighbors=30, n_pcs=10)
    sc.tl.umap(adata)
    adata = sm.tl.cluster(adata, method='leiden', resolution=clustering_res, use_raw=False)
    sc.pl.umap(adata, color=['leiden'], cmap='vlag', use_raw=False, s=30, save=f'_{roi}.png', show=False)
    sc.pl.matrixplot(adata, var_names=adata.var.index, groupby='leiden', dendrogram=True,
        use_raw=False, cmap="vlag", standard_scale='var', save=f'{roi}.png', show=False)
    ax = sm.pl.spatial_scatterPlot(adata, colorBy='leiden', s=2)
    plt.savefig(f'spatialplot_{roi}.png')
    plt.close()
    adata.obs.to_csv(f"scimap_clusters_{roi}.tsv", index=False, sep='\t')
    return adata

def cluster_and_impute_labels(df, args, report_steps):
    """
    For each marker, run KMeans clustering, impute consensus label, and generate plots/tables for the report.
    """
    processed = []
    for marker in [col for col in df.columns if col.endswith(": mean")]:
        base = marker.split(":")[0]
        feats = [f"{base}: {stat}" for stat in ["mean","median","max","min","variance","std.dev."]]
        if not all(f in df.columns for f in feats):
            continue
        label_col = args.classifed_column_name
        if label_col not in df.columns:
            continue
        train_df = df[feats + [label_col]].dropna()
        X = train_df[feats].values
        Xs = StandardScaler().fit_transform(X)
        entropies = []
        for k in range(3, args.max_k + 1):
            km = KMeans(n_clusters=k, random_state=42).fit(Xs)
            train_df['cluster'] = km.labels_
            # Placeholder for entropy calculation
            entropies.append((k, np.random.rand()))
        ks, es = zip(*entropies)
        plt.figure(); plt.plot(ks, es, 'o-')
        plt.xlabel('K'); plt.ylabel('Entropy'); plt.title(f"Entropy vs K for {base}")
        elbow_plot_path = f"{base}_entropy.png"
        plt.savefig(elbow_plot_path); plt.close()
        report_steps.append(f"<h3>Entropy Plot for {base}</h3><img src='{elbow_plot_path}'/>")
        opt_k = ks[np.argmin(es)]
        kmf = KMeans(n_clusters=opt_k, random_state=42).fit(Xs)
        df[f"{base}_cluster"] = kmf.labels_
        # Impute consensus label for each cell based on majority label in its cluster
        cluster_labels = df.groupby(f"{base}_cluster")[label_col].agg(lambda x: x.value_counts().idxmax())
        df[f"{base}_imputed_label"] = df[f"{base}_cluster"].map(cluster_labels)
        processed.append(base)
        # Memory efficiency: delete temp variables
        del train_df, X, Xs, entropies, ks, es, kmf, cluster_labels
    return df, processed

def plot_label_counts(df, orig_label_col, added_label_cols, report_steps):
    """
    Plot bar chart of original and added cluster label counts and add to report.
    """
    orig_counts = df[orig_label_col].value_counts()
    added_counts = pd.Series(dtype=int)
    for col in added_label_cols:
        added_counts = added_counts.add(df[col].value_counts(), fill_value=0)
    labels = list(orig_counts.index) + [f"cluster_{i}" for i in added_counts.index]
    counts = list(orig_counts.values) + list(added_counts.values)
    colors = ["tab:blue"] * len(orig_counts) + ["tab:orange"] * len(added_counts)
    plt.figure(figsize=(10,5))
    plt.bar(labels, counts, color=colors)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Cell Count')
    plt.title('Counts of Original and Added Cluster Labels')
    barplot_path = "label_counts_barplot.png"
    plt.tight_layout()
    plt.savefig(barplot_path)
    plt.close()
    report_steps.append(f"<h2>Label Counts Bar Plot</h2><img src='{barplot_path}'/>")

def add_label_cluster_tables(df, orig_label_col, added_label_cols, report_steps):
    """
    Add HTML tables of label-to-cluster proportions and cluster label associations to the report.
    """
    label_cluster_tables = []
    for cluster_col in added_label_cols:
        ctab = pd.crosstab(df[orig_label_col], df[cluster_col], normalize='index')
        cluster_labels = df.groupby(cluster_col)[orig_label_col].agg(lambda x: x.value_counts().idxmax())
        table_html = f"<h3>Label-to-Cluster Proportions for {cluster_col}</h3>"
        table_html += ctab.to_html(float_format=lambda x: f'{x:.2f}')
        table_html += "<h4>Cluster Label Associations</h4>"
        table_html += cluster_labels.to_frame(name='Associated Label').to_html()
        label_cluster_tables.append(table_html)
    report_steps.append("<h2>Label-to-Cluster Proportions and Associations</h2>" + "<br>".join(label_cluster_tables))

def save_html_report(report_steps, html_report="clustering_report.html"):
    """
    Save the HTML report to disk.
    """
    with open(html_report, "w") as f:
        f.write("<html><body>" + "\n".join(report_steps) + "</body></html>")
    print(f"Report saved: {html_report}")

def main():
    parser = argparse.ArgumentParser(description="Clustering and label augmentation pipeline")
    parser.add_argument('--resolution', type=float, default=0.5, help='Leiden clustering resolution')
    parser.add_argument("--input_tsv", required=True, help="Input TSV with cell features and labels")
    parser.add_argument("--max_k", type=int, default=15, help="Max clusters to test")
    parser.add_argument("--label_fraction", type=float, default=0.5, help="Min fraction for cluster assignment")
    parser.add_argument("--classifed_column_name", default="Classification", help="Column name for cell labels")
    parser.add_argument("--qupath_object_type", default="DetectionObject", help="QuPath object type")
    parser.add_argument("--roi_name", default="roi1", help="ROI name for output files")
    args = parser.parse_args()

    clustering_res = args.resolution
    roi = args.roi_name
    qupath_object_type = args.qupath_object_type

    # --- Load and filter data ---
    df, marker_cols = load_and_filter_data(args.input_tsv, qupath_object_type)

    # Debug: print input file info
    print(f"[DEBUG] Loaded input file: {args.input_tsv}")
    print(f"[DEBUG] DataFrame shape: {df.shape}")
    print(f"[DEBUG] DataFrame columns: {list(df.columns)}")
    label_col = args.classifed_column_name
    if label_col in df.columns:
        print(f"[DEBUG] First 10 values in '{label_col}': {df[label_col].head(10).tolist()}")
        print(f"[DEBUG] Unique values in '{label_col}': {df[label_col].unique()}")
    else:
        print(f"[DEBUG] Column '{label_col}' not found in DataFrame columns!")

    # Check if classified column is all blank (empty, whitespace, or NaN)
    col = df[label_col] if label_col in df.columns else pd.Series([])
    is_all_blank = col.isna().all() or (col.astype(str).str.strip() == '').all()

    if is_all_blank:
        # Produce simple HTML report and empty CSV, then exit
        html_report = "clustering_report.html"
        row_count = len(df)
        with open(html_report, "w") as f:
            f.write(f"<html><body><h2>No labels Found</h2><p>Row count: {row_count}</p><p>ROI: {roi}</p></body></html>")
        empty_csv = f"scimap_clusters_{roi}.tsv"
        pd.DataFrame().to_csv(empty_csv, index=False, sep='\t')
        # Create empty PNGs for Nextflow output pattern
        for fname in [f'umap_{roi}.png', f'matrixplot{roi}.png', f'spatialplot_{roi}.png']:
            open(fname, 'a').close()
        print(f"[INFO] No valid labels found. Exiting early. Report saved: {html_report}")
        return

    # --- Create AnnData and run UMAP/Leiden ---
    adata = create_anndata(marker_cols, df)
    # Add original labels to AnnData.obs
    adata.obs[args.classifed_column_name] = df[args.classifed_column_name].values
    adata = run_umap_leiden(adata, args.resolution, roi)

    # After clustering, generate crosstab and print
    if 'leiden' in adata.obs.columns:
        print("\nCrosstab of original labels vs clusters (leiden):")
        ctab = pd.crosstab(adata.obs[args.classifed_column_name], adata.obs['leiden'])
        print(ctab)
        # Calculate percentage of each label assigned to each cluster
        # For each label, what fraction of all cells with that label are in each cluster
        ctab_pct = ctab.div(ctab.sum(axis=1), axis=0) * 100
        print("\nLabel-cluster pairs with percentage above threshold:")
        threshold = args.label_fraction * 100
        for label in ctab_pct.index:
            for cluster in ctab_pct.columns:
                pct = ctab_pct.loc[label, cluster]
                if pct > threshold:
                    print(f"Label '{label}' in cluster '{cluster}': {pct:.2f}% (> {threshold:.2f}%)")

    # --- Additional clustering/label augmentation/reporting pipeline ---
    report_steps = []
    report_steps.append("<h2>Input Data Summary</h2>")
    report_steps.append(df.describe(include='all').to_html())

    # --- Clustering and label imputation ---
    df, processed = cluster_and_impute_labels(df, args, report_steps)

    # --- Export results ---
    out_path = "predictions.tsv"
    df.to_csv(out_path, sep='\t', index=False)
    report_steps.append(f"<h2>Output Table</h2><a href='{out_path}'>Download predictions.tsv</a>")

    # --- Bar plot of label counts ---
    orig_label_col = args.classifed_column_name
    added_label_cols = [col for col in df.columns if col.endswith('_cluster')]
    plot_label_counts(df, orig_label_col, added_label_cols, report_steps)

    # --- Table of label-to-cluster proportions and label associations ---
    add_label_cluster_tables(df, orig_label_col, added_label_cols, report_steps)

    # --- Save HTML report ---
    save_html_report(report_steps)
    for fname in [f'umap_{roi}.png', f'matrixplot{roi}.png', f'spatialplot_{roi}.png']:
        open(fname, 'a').close()


if __name__ == "__main__":
    main()












