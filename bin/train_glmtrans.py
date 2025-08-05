#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_data(data_path, features_csv):
    # Support both CSV/TSV and pickle DataFrame
    if data_path.endswith('.pkl') or data_path.endswith('.pickle'):
        df = pd.read_pickle(data_path)
        print(f"[DEBUG] Loaded pickle file: {data_path}")
    else:
        df = pd.read_csv(data_path, sep=None, engine='python')
        print(f"[DEBUG] Loaded CSV/TSV file: {data_path}")
    print(f"[DEBUG] DataFrame shape: {df.shape}")
    print(f"[DEBUG] DataFrame columns: {list(df.columns)}")
    features = pd.read_csv(features_csv)
    if features.shape[1] == 1:
        feature_cols = features[features.columns[0]].tolist()
    else:
        feature_cols = features.columns.tolist()
    print(f"[DEBUG] Feature columns used: {feature_cols}")
    X = df[feature_cols].values
    y = df['Classification'] if 'Classification' in df.columns else df.iloc[:, -1]
    print(f"[DEBUG] y value counts: {pd.Series(y).value_counts().to_dict()}")
    return X, y, df, feature_cols

def train_lasso(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"[DEBUG] X_scaled shape: {X_scaled.shape}")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"[DEBUG] Encoded y classes: {le.classes_}")
    model = LogisticRegressionCV(
        Cs=10,
        cv=5,
        penalty='l1',
        solver='saga',
        scoring='roc_auc',
        max_iter=1000,
        n_jobs=-1,
        refit=True,
        multi_class='ovr'
    )
    print("[DEBUG] Fitting LogisticRegressionCV...")
    model.fit(X_scaled, y_enc)
    print(f"[DEBUG] Model fitted. Coef shape: {model.coef_.shape}")
    return model, scaler, le

def evaluate(model, scaler, le, X, y, out_prefix):
    X_scaled = scaler.transform(X)
    print(f"[DEBUG] Evaluation X_scaled shape: {X_scaled.shape}")
    y_enc = le.transform(y)
    print(f"[DEBUG] Evaluation y value counts: {pd.Series(y).value_counts().to_dict()}")
    y_pred = model.predict(X_scaled)
    print(f"[DEBUG] Prediction counts: {pd.Series(y_pred).value_counts().to_dict()}")
    y_prob = model.predict_proba(X_scaled)
    acc = accuracy_score(y_enc, y_pred)
    n_classes = y_prob.shape[1] if y_prob.ndim > 1 else 1
    # ROC/AUC handling
    auc_macro = auc_micro = None
    roc_plot_path = f'{out_prefix}_roc.png'
    if n_classes == 2:
        # Binary
        y_prob_1 = y_prob[:, 1]
        try:
            auc_score = roc_auc_score(y_enc, y_prob_1)
        except Exception:
            auc_score = float('nan')
        print(f"[DEBUG] Accuracy: {acc}, AUC: {auc_score}")
        report = classification_report(y_enc, y_pred, output_dict=True)
        print(f"[DEBUG] Classification report: {json.dumps(report, indent=2)}")
        fpr, tpr, _ = roc_curve(y_enc, y_prob_1)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {evaluate.celltype}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(roc_plot_path)
        plt.close()
        metrics = {
            'accuracy': acc,
            'auc': auc_score,
            'classification_report': report
        }
        with open(f'{out_prefix}_eval.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        return metrics, y_pred, y_prob_1
    else:
        # Multiclass: one-vs-rest ROC for each class, macro/micro AUC
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_enc, classes=np.arange(n_classes))
        auc_macro = roc_auc_score(y_bin, y_prob, average='macro', multi_class='ovr')
        auc_micro = roc_auc_score(y_bin, y_prob, average='micro', multi_class='ovr')
        print(f"[DEBUG] Accuracy: {acc}, Macro AUC: {auc_macro}, Micro AUC: {auc_micro}")
        report = classification_report(y_enc, y_pred, output_dict=True)
        print(f"[DEBUG] Classification report: {json.dumps(report, indent=2)}")
        # Plot ROC for each class
        plt.figure(figsize=(8, 6))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            auc_i = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (AUC={auc_i:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Multiclass ROC Curve for {evaluate.celltype} (macro AUC={auc_macro:.2f}, micro AUC={auc_micro:.2f})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(roc_plot_path)
        plt.close()
        metrics = {
            'accuracy': acc,
            'auc_macro': auc_macro,
            'auc_micro': auc_micro,
            'classification_report': report
        }
        with open(f'{out_prefix}_eval.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        # For compatibility, return the predicted probability for the true class of each sample
        y_prob_true = y_prob[np.arange(len(y_enc)), y_enc]
        return metrics, y_pred, y_prob_true

def plot_coefficients(model, feature_names, out_path):
    coefs = model.coef_
    n_classes = coefs.shape[0] if coefs.ndim > 1 else 1
    print(f"[DEBUG] Plotting coefficients. n_classes: {n_classes}, feature count: {len(feature_names)}")
    plt.figure(figsize=(10, 6))
    # celltype should be passed as argument
    if n_classes == 1:
        abs_coefs = np.abs(coefs.flatten())
        sorted_idx = np.argsort(-abs_coefs)
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_coefs = coefs.flatten()[sorted_idx]
        plt.barh(sorted_features, sorted_coefs)
        plt.xlabel('Coefficient')
        plt.title(f'GLM/LASSO Coefficients for {plot_coefficients.celltype}')
    else:
        for i in range(n_classes):
            abs_coefs = np.abs(coefs[i])
            sorted_idx = np.argsort(-abs_coefs)
            sorted_features = [feature_names[j] for j in sorted_idx]
            sorted_coefs = coefs[i][sorted_idx]
            plt.barh(sorted_features, sorted_coefs, alpha=0.7, label=f'Class {i}')
        plt.xlabel('Coefficient')
        plt.title(f'GLM/LASSO Coefficients (One-vs-Rest) for {plot_coefficients.celltype}')
        plt.legend()
    plt.tight_layout()
    celltype_str = str(plot_coefficients.celltype).replace(' ', '')
    out_path_mod = out_path.replace('.png', f'_{celltype_str}.png')
    plt.savefig(out_path_mod)
    plt.close()
    print(f"[DEBUG] Coefficient plot saved: {out_path}")

def compare_xgb(glm_pred, glm_prob, xgb_json, y_true, out_path):
    # xgb_json: path to XGBoost results JSON (should have predictions and/or probabilities)
    try:
        with open(xgb_json) as f:
            xgb = json.load(f)
        xgb_prob = np.array(xgb.get('probabilities', []))
        xgb_pred = np.array(xgb.get('predictions', []))
        print(f"[DEBUG] XGBoost probabilities shape: {xgb_prob.shape}, predictions shape: {xgb_pred.shape}")
        print(f"[DEBUG] y_true shape: {np.array(y_true).shape}, glm_prob shape: {np.array(glm_prob).shape}")
        # Compare ROC curves
        fpr_glm, tpr_glm, _ = roc_curve(y_true, glm_prob)
        fpr_xgb, tpr_xgb, _ = roc_curve(y_true, xgb_prob)
        auc_glm = auc(fpr_glm, tpr_glm)
        auc_xgb = auc(fpr_xgb, tpr_xgb)
        print(f"[DEBUG] GLM AUC: {auc_glm}, XGB AUC: {auc_xgb}")
        plt.figure()
        plt.plot(fpr_glm, tpr_glm, label=f'GLM AUC={auc_glm:.2f}')
        plt.plot(fpr_xgb, tpr_xgb, label=f'XGB AUC={auc_xgb:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('GLM vs XGBoost ROC')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"[DEBUG] Comparison plot saved: {out_path}")
    except Exception as e:
        print(f"[WARN] Could not compare to XGBoost: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train GLM/LASSO model with optional transfer learning and compare to XGBoost.")
    parser.add_argument('--train_data', required=True, help='Training data CSV/TSV')
    parser.add_argument('--features_csv', required=True, help='CSV with feature column names')
    parser.add_argument('--holdout_data', required=True, help='Holdout data CSV/TSV')
    parser.add_argument('--output_model', default='GLMtrans_model.pkl', help='Output model pickle')
    parser.add_argument('--output_eval', default='GLMtrans_eval.json', help='Output evaluation JSON')
    parser.add_argument('--output_coeffs', default='GLMtrans_coefficients.png', help='Output coefficients plot')
    parser.add_argument('--compare_xgb', default=None, help='XGBoost results JSON for comparison')
    parser.add_argument('--output_comparison', default='GLMtrans_vs_xgb_comparison.png', help='Output comparison plot')
    parser.add_argument('--celltype', required=True, help='Cell type label to focus on (must match classes.npy)')
    return parser.parse_args()

def load_label_names(classes_path="classes.npy"):
    import os
    if os.path.exists(classes_path):
        le_classes = np.load(classes_path, allow_pickle=True)
        label_names = le_classes.tolist()
        print(f"[DEBUG] Loaded label names from {classes_path}: {label_names}")
        return label_names
    return None

def generate_coefficient_heatmaps(df_train, feature_names, celltype):
    """
    For each image, train GLM/LASSO at various training proportions and plot coefficient heatmaps.
    """
    import seaborn as sns
    from scipy.cluster.hierarchy import linkage, leaves_list
    thresholds = [0.01, 0.05, 0.10, 0.20, 0.50, 0.75, 1.0]
    if 'Image' not in df_train.columns:
        print("[WARN] No 'Image' column found in training data. Skipping heatmap generation.")
        return
    images = df_train['Image'].unique()
    # Prepare to collect all coefficients
    all_labels = []  # (image, threshold) labels for columns
    all_coefs = []   # list of coefficient arrays (len(feature_names))
    for img in images:
        img_df = df_train[df_train['Image'] == img]
        print(f"[DEBUG] Image: {img} img_df Classification value counts: {img_df['Classification'].value_counts().to_dict()}")
        if img_df.shape[0] == 0:
            print(f"[DEBUG] Skipping image {img}: no samples for image")
            continue
        for frac in thresholds:
            n = int(img_df.shape[0] * frac)
            label = f"{img}|{int(frac*100)}%"
            if n < 2:
                print(f"[DEBUG] Skipping {label}: not enough samples for threshold (n={n})")
                all_coefs.append([np.nan]*len(feature_names))
                all_labels.append(label)
                continue
            # Subset n rows from img_df
            subset_df = img_df.sample(n=n, random_state=42) if n < img_df.shape[0] else img_df.copy()
            # Split into celltype and other
            celltype_df = subset_df[subset_df['Classification'] == celltype]
            other_df = subset_df[subset_df['Classification'] != celltype]
            print(f"[DEBUG] {label} subset_df Classification value counts: {subset_df['Classification'].value_counts().to_dict()}")
            print(f"[DEBUG] {label} celltype_df shape: {celltype_df.shape}, value counts: {celltype_df['Classification'].value_counts().to_dict()}")
            print(f"[DEBUG] {label} other_df shape: {other_df.shape}, value counts: {other_df['Classification'].value_counts().to_dict()}")
            n_celltype = min(celltype_df.shape[0], n // 2)
            if n_celltype < 1:
                print(f"[DEBUG] Skipping {label}: not enough celltype samples (celltype={n_celltype})")
                all_coefs.append([np.nan]*len(feature_names))
                all_labels.append(label)
                continue
            max_other = min(2 * n_celltype, other_df.shape[0])
            if max_other < 1:
                print(f"[DEBUG] Skipping {label}: not enough 'other' samples (other={max_other})")
                all_coefs.append([np.nan]*len(feature_names))
                all_labels.append(label)
                continue
            n_other = max_other if max_other <= n_celltype else np.random.randint(n_celltype, max_other + 1)
            print(f"[DEBUG] {label} final sample sizes: celltype={n_celltype}, other={n_other}")
            sampled_celltype = celltype_df.sample(n=n_celltype, random_state=42) if n_celltype < celltype_df.shape[0] else celltype_df.copy()
            sampled_other = other_df.sample(n=n_other, random_state=42) if n_other < other_df.shape[0] else other_df.copy()
            sample_df = pd.concat([sampled_celltype, sampled_other]).sample(frac=1, random_state=42)
            X_sub = sample_df[feature_names].values
            y_sub = np.where(pd.Series(sample_df['Classification']) == celltype, celltype, "other")
            print(f"[DEBUG] {label} y_sub value counts: {pd.Series(y_sub).value_counts().to_dict()}")
            try:
                model_sub, scaler_sub, le_sub = train_lasso(X_sub, y_sub)
                coefs = model_sub.coef_.flatten()
                print(f"[DEBUG] {label} coefs: {coefs}")
                all_coefs.append(coefs)
            except Exception as e:
                print(f"[ERROR] Exception for {label}: {e}")
                all_coefs.append([np.nan]*len(feature_names))
            all_labels.append(label)
    # Convert to matrix: shape (features, image*threshold)
    coef_mat = np.array(all_coefs).T
    print("[DEBUG] coef_mat shape:", coef_mat.shape)
    print("[DEBUG] coef_mat values:\n", coef_mat)
    # Plot single heatmap with improved font size and color scaling
    import matplotlib as mpl
    font_scale = 1.2 if len(feature_names) < 40 else 0.8
    mpl.rc('font', size=font_scale * 14)
    mpl.rc('axes', titlesize=font_scale * 16)
    mpl.rc('xtick', labelsize=font_scale * 12)
    mpl.rc('ytick', labelsize=font_scale * 12)
    plt.figure(figsize=(max(16, len(all_labels)*0.5), max(10, len(feature_names)*0.3)))
    absmax = np.nanmax(np.abs(coef_mat))
    vmin, vmax = -absmax, absmax
    celltype_str = str(celltype).replace(' ', '')
    heatmap_path = f'coeff_heatmap_all_{celltype_str}.png'
    try:
        # Sort rows by median value: highest positive at top, lowest negative at bottom
        medians = np.nanmedian(coef_mat, axis=1)
        row_order = np.argsort(-medians)  # descending order
        coef_mat_sorted = coef_mat[row_order, :]
        sorted_feature_names = [feature_names[i] for i in row_order]
        g = sns.heatmap(
            coef_mat_sorted,
            cmap='vlag', center=0, vmin=vmin, vmax=vmax,
            yticklabels=sorted_feature_names, xticklabels=all_labels
        )
        g.set_title(f'Coefficient Heatmap for {celltype}', fontsize=font_scale*16)
        g.set_xlabel('Image | Threshold', fontsize=font_scale*14)
        g.set_ylabel('Feature', fontsize=font_scale*14)
        plt.tight_layout()
        plt.savefig(heatmap_path, bbox_inches='tight')
        plt.close()
    except Exception:
        ax = sns.heatmap(
            coef_mat,
            cmap='vlag', center=0, vmin=vmin, vmax=vmax,
            yticklabels=feature_names, xticklabels=all_labels
        )
        ax.set_title(f'Coefficient Heatmap for {celltype}', fontsize=font_scale*16)
        ax.set_xlabel('Image | Threshold', fontsize=font_scale*14)
        ax.set_ylabel('Feature', fontsize=font_scale*14)
        plt.tight_layout()
        plt.savefig(heatmap_path, bbox_inches='tight')
        plt.close()

def main():
    args = parse_args()
    # Load training data
    X_train, y_train, df_train, feature_names = load_data(args.train_data, args.features_csv)
    # Load label names from classes.npy if available
    label_names = load_label_names("classes.npy")
    # Convert y_train to binary: celltype vs other
    celltype = args.celltype
    y_train_bin = np.where(pd.Series(y_train) == celltype, celltype, "other")
    # Train model
    model, scaler, le = train_lasso(X_train, y_train_bin)
    joblib.dump({'model': model, 'scaler': scaler, 'label_encoder': le, 'features': feature_names}, args.output_model)
    # Plot coefficients
    plot_coefficients.celltype = celltype
    plot_coefficients(model, feature_names, args.output_coeffs)
    # Generate coefficient heatmaps per image
    generate_coefficient_heatmaps(df_train, feature_names, celltype)
    # Load holdout data
    X_hold, y_hold, df_hold, _ = load_data(args.holdout_data, args.features_csv)
    y_hold_bin = np.where(pd.Series(y_hold) == celltype, celltype, "other")
    # Evaluate
    evaluate.celltype = celltype
    metrics, y_pred, y_prob = evaluate(model, scaler, le, X_hold, y_hold_bin, args.output_model.replace('.pkl', ''))
    with open(args.output_eval, 'w') as f:
        json.dump(metrics, f, indent=2)
    # Compare to XGBoost if provided
    if args.compare_xgb:
        y_true = le.transform(y_hold_bin)
        compare_xgb(y_pred, y_prob, args.compare_xgb, y_true, args.output_comparison)

if __name__ == "__main__":
    main()
