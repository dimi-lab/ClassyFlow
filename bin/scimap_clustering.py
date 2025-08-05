#!/usr/bin/env python3

import argparse
import sys
import pandas as pd
import numpy as np
import scanpy as sc
import scimap as sm
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
from sklearn.cluster import KMeans
from sklearn.feature_selection import f_classif

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

def load_and_filter_data(input_tsv, qupath_object_type, label_col):
    df = pd.read_csv(input_tsv, sep='\t')
    if 'qc' in df.columns:
        df = df[~df['qc'].astype(str).str.contains("Artifact", na=False)]
    if label_col in df.columns:
        df[label_col] = df[label_col].replace('0', '')
    marker_regex = r'(Cell: Median)' if qupath_object_type == "CellObject" else r'(Median)'
    marker_replace = ": Cell: Median" if qupath_object_type == "CellObject" else "Median"
    marker_cols = df.filter(regex=marker_regex, axis=1)
    marker_cols.columns = marker_cols.columns.str.replace(marker_replace, "")
    marker_cols = marker_cols.filter(regex='^((?!DAPI).)*', axis=1)
    if marker_cols.shape[1] == 0:
        raise ValueError("No marker columns found after filtering.")
    return df, marker_cols

def create_anndata(marker_cols, df):
    """
    Create an AnnData object from marker columns and add spatial and metadata.
    """
    col_x = find_best_coord(df.columns, 'X')
    col_y = find_best_coord(df.columns, 'Y')
    adata = sc.AnnData(marker_cols)
    adata.obsm["spatial"] = df[[col_x, col_y]].to_numpy()
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
    adata.obs.to_json(f"scimap_clusters_{roi}.json", orient="records", lines=True)
    return adata

def impute_labels(df, adata, label_col, cluster_col, label_fraction):
    ctab = pd.crosstab(adata.obs[label_col], adata.obs[cluster_col])
    ctab_pct = ctab.div(ctab.sum(axis=1), axis=0)
    cluster_to_label = {}
    for cluster in ctab.columns:
        for label in ctab.index:
            if ctab_pct.loc[label, cluster] > label_fraction:
                cluster_to_label[cluster] = label
    cluster_assignments = adata.obs[cluster_col].astype(str)
    # Add LeidenClusters column with 'leiden_' prefix
    df["LeidenClusters"] = [f"leiden_{cid}" for cid in cluster_assignments]

    # Only override a random subset of blank labels, N=3x count of each label
    imputed_labels = cluster_assignments.map(cluster_to_label)
    orig_label = df[label_col].astype(str).str.strip()
    override_status = ["No"] * len(df)

    # For each label, override up to N blank rows in the cluster, N=3x count of that label
    for label_name in ctab.index:
        # Find indices where original label is blank and imputed label matches label_name
        idx_blank = [i for i, (ol, il) in enumerate(zip(orig_label, imputed_labels)) if ol == "" and il == label_name]
        n_label = (df[label_col] == label_name).sum()
        n_override = min(len(idx_blank), 3 * n_label)
        if n_override > 0:
            chosen_idx = np.random.choice(idx_blank, n_override, replace=False)
            for i in chosen_idx:
                df.at[i, label_col] = label_name
                override_status[i] = "Yes"

    df[label_col + "_imputed"] = override_status
    return df, ctab, ctab_pct, cluster_to_label

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

def save_html_report(report_steps, html_report="clustering_report.html"):
    with open(html_report, "w") as f:
        f.write("<html><body>\n")
        for step in report_steps:
            f.write(step + "\n")
        f.write("</body></html>\n")

def plot_coefficients_heatmap(df, feature_cols, label_col, fractions, image_col, report_steps):
    """
    For each image, fit a LASSO model at different label fractions and plot a heatmap of coefficients.
    """
    warnings.filterwarnings("ignore")

    images = df[image_col].unique()
    heatmap_paths = []
    for img in images:
        img_df = df[df[image_col] == img]
        coef_matrix = []
        valid_fracs = []
        for frac in fractions:
            frac_df = img_df.sample(frac=frac, random_state=42) if frac < 1.0 else img_df.copy()
            if frac_df[label_col].nunique() < 2 or frac_df.shape[0] < 10:
                coef_matrix.append([np.nan]*len(feature_cols))
                valid_fracs.append(f"{int(frac*100)}%")
                continue
            X = frac_df[feature_cols].values
            y = frac_df[label_col].values
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
            try:
                model = LogisticRegressionCV(
                    Cs=10, cv=3, penalty='l1', solver='saga', scoring='roc_auc',
                    max_iter=500, n_jobs=-1, refit=True, multi_class='ovr')
                model.fit(Xs, y_enc)
                # For multiclass, take mean absolute value across classes
                coefs = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
                coef_matrix.append(coefs)
            except Exception as e:
                coef_matrix.append([np.nan]*len(feature_cols))
            valid_fracs.append(f"{int(frac*100)}%")
        coef_matrix = np.array(coef_matrix)
        plt.figure(figsize=(max(8, len(feature_cols)//2), 6))
        sns.heatmap(coef_matrix, xticklabels=feature_cols, yticklabels=valid_fracs, cmap="vlag", cbar_kws={"label": "|Coefficient|"})
        plt.title(f"Feature Coefficients Heatmap - Image: {img}")
        plt.xlabel("Feature")
        plt.ylabel("Label Fraction")
        plt.tight_layout()
        heatmap_path = f"coeff_heatmap_{img}.png"
        plt.savefig(heatmap_path)
        plt.close()
        report_steps.append(f"<h3>Coefficient Heatmap for Image: {img}</h3><img src='{heatmap_path}'/>")
        heatmap_paths.append(heatmap_path)
    return heatmap_paths


def main():

    parser = argparse.ArgumentParser(description="Cluster and impute labels using Leiden clustering.")
    parser.add_argument("--input_tsv", required=True)
    parser.add_argument("--resolution", type=float, default=0.5)
    parser.add_argument("--label_fraction", type=float, default=0.5)
    parser.add_argument("--classifed_column_name", default="Classification")
    parser.add_argument("--qupath_object_type", default="DetectionObject")
    parser.add_argument("--roi_name", default="roi1")
    parser.add_argument("--perc_top_features", type=float, default=1.0, help="Percent of top differentiating features to keep for clustering (0-1, e.g. 0.3 for top 30%)")
    args = parser.parse_args()

    label_col = args.classifed_column_name
    df, marker_cols = load_and_filter_data(args.input_tsv, args.qupath_object_type, label_col)

    # --- Feature selection: keep top N% differentiating features ---
    perc = args.perc_top_features
    if perc < 1.0:
        # Calculate feature importance by ANOVA F-value between labels
        from sklearn.feature_selection import f_classif
        valid_idx = df[label_col].notna() & (df[label_col].astype(str).str.strip() != '')
        marker_data = marker_cols[valid_idx]
        label_data = df.loc[valid_idx, label_col]
        try:
            fvals, _ = f_classif(marker_data, label_data)
            n_keep = max(1, int(len(marker_cols.columns) * perc))
            top_idx = np.argsort(fvals)[::-1][:n_keep]
            top_features = marker_cols.columns[top_idx]
            marker_cols = marker_cols[top_features]
            print(f"[INFO] Keeping top {perc*100:.1f}% ({n_keep}) features for clustering.")
        except Exception as e:
            print(f"[WARN] Feature selection failed: {e}. Using all features.")

    adata = create_anndata(marker_cols, df)
    adata.obs[label_col] = df[label_col].values

    # Check if classified column is all blank (empty, whitespace, or NaN)
    col = df[label_col] if label_col in df.columns else pd.Series([])
    is_all_blank = col.isna().all() or (col.astype(str).str.strip() == '').all()
    roi = args.roi_name
    if is_all_blank:
        html_report = "clustering_report.html"
        row_count = len(df)
        with open(html_report, "w") as f:
            f.write(f"<html><body><h2>No labels Found</h2><p>Row count: {row_count}</p><p>ROI: {roi}</p></body></html>")
        empty_csv = f"scimap_clusters_{roi}.tsv"
        pd.DataFrame().to_csv(empty_csv, index=False, sep='\t')
        for fname in [f'umap_{roi}.png', f'matrixplot{roi}.png', f'spatialplot_{roi}.png']:
            open(fname, 'a').close()
        print(f"[INFO] No valid labels found. Exiting early. Report saved: {html_report}")
        return

    adata = run_umap_leiden(adata, args.resolution, roi)
    cluster_col = "leiden"

    # Impute labels
    df, ctab, ctab_pct, cluster_to_label = impute_labels(df, adata, label_col, cluster_col, args.label_fraction)

    # Debug print statements
    print("[DEBUG] DataFrame with imputed labels (first 5 rows):")
    print(df.head())
    print("[DEBUG] Crosstab of labels vs clusters:")
    print(ctab)
    print("[DEBUG] Crosstab percentages:")
    print(ctab_pct)
    print("[DEBUG] Cluster to label mapping:")
    print(cluster_to_label)

    # Save output with all original columns + imputed label
    out_path = f"scimap_extended_{roi}.tsv"
    df.to_csv(out_path, sep='\t', index=False)

    # Generate report
    report_steps = []
    report_steps.append("<h2>Input Data Summary</h2>")
    report_steps.append(df.describe(include='all').to_html())
    report_steps.append("<h2>Label-Cluster Crosstab</h2>")
    report_steps.append(ctab.to_html())
    report_steps.append("<h2>Label-Cluster Proportions</h2>")
    report_steps.append(ctab_pct.to_html(float_format=lambda x: f'{x:.2f}'))
    report_steps.append(f"<h2>Imputed Label Mapping</h2><pre>{cluster_to_label}</pre>")
    report_steps.append(f"<h2>Output Table</h2><a href='{out_path}'>Download {out_path}</a>")
    save_html_report(report_steps, "clustering_report.html")

    # Save plots (optional, can add UMAP, barplots, etc. as needed)
    # sc.pl.umap(adata, color=[cluster_col], save=f'_{roi}.png', show=False)

if __name__ == "__main__":
    main()












