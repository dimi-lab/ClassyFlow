#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import xgboost as xgb
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

# Constants
METHODS = ['log', 'quantile', 'minmax', 'boxcox']
TEST_SIZE = 0.3

def get_target_columns(df, target_features):
    """Get columns matching target features"""
    target_cols = []
    for target in target_features:
        matching_cols = [col for col in df.columns if target in col]
        target_cols.extend(matching_cols)
    return list(set(target_cols))

def calculate_pearson_p_statistic(data):
    """Calculate Pearson P/df normality statistic"""
    try:
        clean_data = data.dropna()
        if len(clean_data) < 50:
            return np.nan
            
        # Use fewer bins for stability
        n_bins = min(15, max(5, len(clean_data) // 100))
        hist, bin_edges = np.histogram(clean_data, bins=n_bins)
        
        # Expected frequencies under normal distribution
        data_mean = clean_data.mean()
        data_std = clean_data.std()
        
        if data_std == 0:
            return np.nan
        
        expected = []
        for i in range(len(bin_edges) - 1):
            left = (bin_edges[i] - data_mean) / data_std
            right = (bin_edges[i + 1] - data_mean) / data_std
            prob = stats.norm.cdf(right) - stats.norm.cdf(left)
            expected.append(prob * len(clean_data))
        
        expected = np.array(expected)
        
        # Remove bins with very low expected frequencies
        valid_bins = expected >= 5
        if valid_bins.sum() < 3:
            return np.nan
            
        observed = hist[valid_bins]
        expected = expected[valid_bins]
        
        # Calculate chi-square statistic
        chi2_stat = np.sum((observed - expected) ** 2 / expected)
        df = len(observed) - 1 - 2  # -2 for estimated mean and std
        
        if df <= 0:
            return np.nan
            
        return chi2_stat / df
    except:
        return np.nan

def calculate_pearson_statistics(df, method_name, target_features):
    """Calculate Pearson P/df statistics for a single method/batch"""
    results = []
    target_cols = get_target_columns(df, target_features)
    
    for col in target_cols:
        if col in df.columns:
            p_stat = calculate_pearson_p_statistic(df[col])
            results.append({
                'method': method_name,
                'marker': col.replace('Cell: ', '').replace(': Mean', ''),
                'pearson_p_df': p_stat
            })
    
    return pd.DataFrame(results)

def create_pca_plot(df, method_name, batch_id, target_features):
    """Create PCA plot for a single method/batch"""
    target_cols = get_target_columns(df, target_features)
    
    if 'Image' not in df.columns or len(target_cols) < 2:
        print(f"Skipping PCA plot for {method_name} - insufficient data")
        return
    
    # Prepare data for PCA
    pca_data = df[target_cols].dropna()
    images = df.loc[pca_data.index, 'Image']
    
    if len(pca_data) < 10:
        print(f"Skipping PCA plot for {method_name} - insufficient samples")
        return
    
    # Standardize and apply PCA
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pca_data)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    
    # Use Image ID for coloring but limit to reasonable number of colors
    unique_images = images.unique()
    if len(unique_images) > 20:
        # Sample images if too many
        sampled_images = np.random.choice(unique_images, 20, replace=False)
        mask = images.isin(sampled_images)
        pca_result_plot = pca_result[mask]
        images_plot = images[mask]
    else:
        pca_result_plot = pca_result
        images_plot = images
    
    # Create scatter plot
    unique_images_plot = images_plot.unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_images_plot)))
    
    for j, img_id in enumerate(unique_images_plot):
        img_mask = images_plot == img_id
        plt.scatter(pca_result_plot[img_mask, 0], pca_result_plot[img_mask, 1], 
                   c=[colors[j]], label=str(img_id)[:8], alpha=0.6, s=30)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.title(f'PCA - {method_name.capitalize()} - Batch {batch_id}')
    plt.grid(True, alpha=0.3)
    
    if len(unique_images_plot) <= 10:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    output_file = f"pca_{method_name}_{batch_id}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PCA plot saved: {output_file}")

def create_density_plot(df, method_name, batch_id, target_features):
    """Create density plot for a single method/batch"""
    target_cols = get_target_columns(df, target_features)
    
    if not target_cols:
        print(f"Skipping density plot for {method_name} - no target columns")
        return
        
    # Use first few target columns
    cols_to_plot = target_cols[:3]
    
    fig, axes = plt.subplots(1, len(cols_to_plot), figsize=(5 * len(cols_to_plot), 4))
    if len(cols_to_plot) == 1:
        axes = [axes]
    
    for i, col in enumerate(cols_to_plot):
        if col in df.columns:
            data = df[col].dropna()
            if len(data) > 0:
                axes[i].hist(data, bins=50, alpha=0.7, density=True, color='skyblue')
                axes[i].set_title(f'{col.replace("Cell: ", "").replace(": Mean", "")}')
                axes[i].set_xlabel('Value')
                if i == 0:
                    axes[i].set_ylabel('Density')
            else:
                axes[i].text(0.5, 0.5, 'No data', ha='center', va='center', 
                           transform=axes[i].transAxes)
    
    plt.suptitle(f'Distribution - {method_name.capitalize()} - Batch {batch_id}')
    plt.tight_layout()
    output_file = f"density_{method_name}_{batch_id}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Density plot saved: {output_file}")

def train_xgboost_single(df, method_name, sample_size):
    """Train XGBoost on single method data"""
    print(f"Training XGBoost for {method_name}...")
    
    # Get all numeric columns (excluding metadata columns)
    exclude_cols = ['Classification', 'Image', 'batch_id', 'Slide']
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if 'Classification' not in df.columns:
        print(f"Skipping {method_name} - no 'Classification' column found")
        return None
        
    if len(feature_cols) < 5:
        print(f"Skipping {method_name} - insufficient feature columns (found {len(feature_cols)})")
        return None
    
    # Remove rows with missing Classification labels
    df_clean = df.dropna(subset=['Classification'])
    if len(df_clean) == 0:
        print(f"Skipping {method_name} - no valid Classification labels")
        return None
    
    # Subsample if dataset is too large
    if len(df_clean) > sample_size:
        df_clean = df_clean.sample(n=sample_size, random_state=42)
        print(f"  Subsampled to {len(df_clean)} rows")
    
    # Prepare features and target
    X = df_clean[feature_cols].fillna(0)
    y = df_clean['Classification']
    
    # Convert string labels to numeric using LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Ensure we have enough samples per class
    class_counts = pd.Series(y_encoded).value_counts()
    original_class_counts = y.value_counts()
    print(f"  Class distribution: {dict(original_class_counts)}")
    
    if class_counts.min() < 1:
        print(f"  Skipping {method_name} - insufficient samples per class")
        return None
    
    if len(class_counts) < 2:
        print(f"  Skipping {method_name} - need at least 2 classes")
        return None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=42, stratify=y_encoded
    )
    
    # Train XGBoost model (shallow)
    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # For multiclass AUC, use ovr (one-vs-rest)
    try:
        auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    except:
        auc_score = np.nan
    
    print(f"  Accuracy: {accuracy:.3f}, AUC: {auc_score:.3f}")
    
    # Calculate per-class AUC
    n_classes = len(label_encoder.classes_)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    
    class_aucs = {}
    for class_idx, class_name in enumerate(label_encoder.classes_):
        try:
            fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], y_pred_proba[:, class_idx])
            class_auc = auc(fpr, tpr)
            class_aucs[class_name] = class_auc
        except:
            class_aucs[class_name] = np.nan
    
    results = {
        'method': method_name,
        'accuracy': accuracy,
        'auc': auc_score,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': X.shape[1],
        'n_classes': len(label_encoder.classes_),
        'class_names': label_encoder.classes_.tolist(),
        'class_distribution': dict(original_class_counts),
        'class_aucs': class_aucs
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Process single batch/method for normalization comparison")
    parser.add_argument('--batch-id', required=True, help='Batch ID to process')
    parser.add_argument('--method', required=True, choices=['original'] + METHODS,
                       help='Normalization method to process')
    parser.add_argument('--target-features', dest='target_features', 
                       default='Cell: Mean', help='Comma-separated target features')
    parser.add_argument('--sample-size', type=int, default=20000,
                       help='Sample size for XGBoost testing')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip PCA and density plot generation')
    parser.add_argument('--skip-ml', action='store_true',
                       help='Skip XGBoost model training')
    
    args = parser.parse_args()
    
    batch_id = args.batch_id
    method = args.method
    target_features = [f.strip() for f in args.target_features.split(',')]
    
    print(f"Processing batch {batch_id} with method {method}")
    
    # Load data
    if method == 'original':
        data_file = f"merged_dataframe_{batch_id}_mod.pkl"
        if not Path(data_file).exists():
            print(f"Error: Original data file {data_file} not found")
            return
        df = pd.read_pickle(data_file)
    else:
        data_file = f"{method}_transformed_{batch_id}.tsv"
        if not Path(data_file).exists():
            print(f"Error: Transformed data file {data_file} not found")
            return
        df = pd.read_csv(data_file, sep='\t')
    
    df['batch_id'] = batch_id
    print(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
    
    # Calculate Pearson statistics
    print("Calculating Pearson P/df statistics...")
    pearson_results = calculate_pearson_statistics(df, method, target_features)
    pearson_file = f"pearson_stats_{method}_{batch_id}.csv"
    pearson_results.to_csv(pearson_file, index=False)
    print(f"Pearson statistics saved to {pearson_file}")
    
    # Create plots
    if not args.skip_plots:
        print("Creating PCA plot...")
        create_pca_plot(df, method, batch_id, target_features)
        
        print("Creating density plot...")
        create_density_plot(df, method, batch_id, target_features)
    
    # Train XGBoost model
    ml_results = None
    if not args.skip_ml:
        print("Training XGBoost model...")
        ml_results = train_xgboost_single(df, method, args.sample_size)
        
        if ml_results:
            ml_file = f"ml_results_{method}_{batch_id}.json"
            with open(ml_file, 'w') as f:
                json.dump(ml_results, f, indent=2, default=str)
            print(f"ML results saved to {ml_file}")
    
    # Create summary file
    summary = {
        'batch_id': batch_id,
        'method': method,
        'n_rows': len(df),
        'n_features': len([col for col in df.select_dtypes(include=[np.number]).columns 
                          if col not in ['Classification', 'Image', 'batch_id', 'Slide']]),
        'target_features': target_features,
        'pearson_file': pearson_file,
        'pca_plot': f"pca_{method}_{batch_id}.png" if not args.skip_plots else None,
        'density_plot': f"density_{method}_{batch_id}.png" if not args.skip_plots else None,
        'ml_file': f"ml_results_{method}_{batch_id}.json" if ml_results else None,
        'processing_timestamp': pd.Timestamp.now().isoformat()
    }
    
    summary_file = f"summary_{method}_{batch_id}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nProcessing complete for {method} - {batch_id}")
    print(f"Summary file: {summary_file}")
    print("Generated files:")
    print(f"  - {pearson_file}")
    if not args.skip_plots:
        print(f"  - pca_{method}_{batch_id}.png")
        print(f"  - density_{method}_{batch_id}.png")
    if ml_results:
        print(f"  - ml_results_{method}_{batch_id}.json")

if __name__ == "__main__":
    main()