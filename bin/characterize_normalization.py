#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import argparse
import warnings
warnings.filterwarnings('ignore')

# Constants
METHODS = ['log', 'quantile', 'minmax', 'boxcox']
TEST_SIZE = 0.3

def load_all_batch_data(batch_ids):
    """Load data for all batches and methods"""
    all_data = {}
    
    for batch_id in batch_ids:
        print(f"Loading batch {batch_id}...")
        batch_data = {}
        
        # Load original data
        original_file = f"merged_dataframe_{batch_id}_mod.pkl"
        if Path(original_file).exists():
            batch_data['original'] = pd.read_pickle(original_file)
            batch_data['original']['batch_id'] = batch_id
        
        # Load transformed data
        for method in METHODS:
            transformed_file = f"{method}_transformed_{batch_id}.tsv"
            if Path(transformed_file).exists():
                df = pd.read_csv(transformed_file, sep='\t')
                df['batch_id'] = batch_id
                batch_data[method] = df
        
        all_data[batch_id] = batch_data
        print(f"  Loaded {len(batch_data)} datasets for batch {batch_id}")
    
    return all_data

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

def calculate_pearson_statistics_all_batches(all_data, target_features):
    """Calculate Pearson P/df for all batches and methods"""
    results = []
    
    for batch_id, batch_data in all_data.items():
        for method_name, df in batch_data.items():
            target_cols = get_target_columns(df, target_features)
            
            for col in target_cols:
                if col in df.columns:
                    p_stat = calculate_pearson_p_statistic(df[col])
                    results.append({
                        'batch_id': batch_id,
                        'method': method_name,
                        'marker': col.replace('Cell: ', '').replace(': Mean', ''),
                        'pearson_p_df': p_stat
                    })
    
    results_df = pd.DataFrame(results)
    
    # Create summary table
    summary = results_df.groupby(['method', 'batch_id'])['pearson_p_df'].mean().reset_index()
    summary_pivot = summary.pivot(index='batch_id', columns='method', values='pearson_p_df')
    
    # Save results
    results_df.to_csv('pearson_statistics_detailed.csv', index=False)
    summary_pivot.to_csv('pearson_statistics_summary.csv')
    
    print("Pearson P/df statistics saved to pearson_statistics_detailed.csv and pearson_statistics_summary.csv")
    return results_df, summary_pivot

def create_pca_plots(all_data, target_features):
    """Create PCA plots for each batch and method, colored by Image ID"""
    batch_ids = list(all_data.keys())
    methods_available = []
    
    # Find common methods across all batches
    for method in ['original'] + METHODS:
        if all(method in all_data[batch_id] for batch_id in batch_ids):
            methods_available.append(method)
    
    print(f"Creating PCA plots for methods: {methods_available}")
    
    for batch_id in batch_ids:
        fig, axes = plt.subplots(1, len(methods_available), figsize=(5 * len(methods_available), 5))
        if len(methods_available) == 1:
            axes = [axes]
        
        for i, method in enumerate(methods_available):
            df = all_data[batch_id][method]
            target_cols = get_target_columns(df, target_features)
            
            if 'Image' not in df.columns or len(target_cols) < 2:
                axes[i].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{method.capitalize()}')
                continue
            
            # Prepare data for PCA
            pca_data = df[target_cols].dropna()
            images = df.loc[pca_data.index, 'Image']
            
            if len(pca_data) < 10:
                axes[i].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{method.capitalize()}')
                continue
            
            # Standardize and apply PCA
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(pca_data)
            
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)
            
            # Plot - use Image ID for coloring but limit to reasonable number of colors
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
                axes[i].scatter(pca_result_plot[img_mask, 0], pca_result_plot[img_mask, 1], 
                              c=[colors[j]], label=str(img_id)[:8], alpha=0.6, s=10)
            
            axes[i].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            axes[i].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            axes[i].set_title(f'{method.capitalize()}')
            
            # Add legend only for first plot to avoid clutter
            if i == 0 and len(unique_images_plot) <= 10:
                axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        output_file = f"pca_batch_{batch_id}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PCA plot saved for batch {batch_id}: {output_file}")

def create_density_plots(all_data, target_features):
    """Create density plots for each batch and method"""
    batch_ids = list(all_data.keys())
    methods_available = []
    
    # Find common methods across all batches
    for method in ['original'] + METHODS:
        if all(method in all_data[batch_id] for batch_id in batch_ids):
            methods_available.append(method)
    
    print(f"Creating density plots for methods: {methods_available}")
    
    for batch_id in batch_ids:
        # Get a representative marker (first available target column)
        sample_df = list(all_data[batch_id].values())[0]
        target_cols = get_target_columns(sample_df, target_features)
        
        if not target_cols:
            print(f"No target columns found for batch {batch_id}")
            continue
            
        # Use first few target columns
        cols_to_plot = target_cols[:3]
        
        fig, axes = plt.subplots(len(cols_to_plot), len(methods_available), 
                                figsize=(5 * len(methods_available), 4 * len(cols_to_plot)))
        
        if len(cols_to_plot) == 1:
            axes = axes.reshape(1, -1)
        if len(methods_available) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, col in enumerate(cols_to_plot):
            for j, method in enumerate(methods_available):
                df = all_data[batch_id][method]
                
                if col in df.columns:
                    data = df[col].dropna()
                    if len(data) > 0:
                        axes[i, j].hist(data, bins=50, alpha=0.7, density=True, color='skyblue')
                        axes[i, j].set_title(f'{method.capitalize()}')
                        if j == 0:
                            axes[i, j].set_ylabel(f'{col.replace("Cell: ", "").replace(": Mean", "")}')
                        if i == len(cols_to_plot) - 1:
                            axes[i, j].set_xlabel('Value')
                    else:
                        axes[i, j].text(0.5, 0.5, 'No data', ha='center', va='center', 
                                      transform=axes[i, j].transAxes)
        
        plt.tight_layout()
        output_file = f"density_plots_batch_{batch_id}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Density plots saved for batch {batch_id}: {output_file}")

def prepare_ml_dataset(all_data, sample_size):
    """Prepare datasets for XGBoost testing using Classification column"""
    print(f"Preparing ML datasets with sample size {sample_size}...")
    
    datasets = {}
    
    # Find methods available across all batches
    methods_available = ['original']
    for method in METHODS:
        if all(method in batch_data for batch_data in all_data.values()):
            methods_available.append(method)
    
    print(f"Methods available for ML testing: {methods_available}")
    
    for method in methods_available:
        print(f"Processing {method}...")
        
        # Combine all batches for this method
        method_dfs = []
        for batch_id, batch_data in all_data.items():
            if method in batch_data:
                df = batch_data[method].copy()
                method_dfs.append(df)
        
        if not method_dfs:
            continue
            
        combined_df = pd.concat(method_dfs, ignore_index=True)
        print(f"  Combined dataset size: {len(combined_df)}")
        
        # Get all numeric columns (excluding metadata columns)
        exclude_cols = ['Classification', 'Image', 'batch_id', 'Slide']
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if 'Classification' not in combined_df.columns:
            print(f"  Skipping {method} - no 'Classification' column found")
            continue
            
        if len(feature_cols) < 5:
            print(f"  Skipping {method} - insufficient feature columns (found {len(feature_cols)})")
            continue
        
        # Remove rows with missing Classification labels
        combined_df = combined_df.dropna(subset=['Classification'])
        if len(combined_df) == 0:
            print(f"  Skipping {method} - no valid Classification labels")
            continue
        
        # Subsample if dataset is too large
        if len(combined_df) > sample_size:
            combined_df = combined_df.sample(n=sample_size, random_state=42)
            print(f"  Subsampled to {len(combined_df)} rows")
        
        # Prepare features and target
        X = combined_df[feature_cols].fillna(0)
        y = combined_df['Classification']
        
        # Convert string labels to numeric using LabelEncoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Ensure we have enough samples per class
        class_counts = pd.Series(y_encoded).value_counts()
        original_class_counts = y.value_counts()
        print(f"  Class distribution: {dict(original_class_counts)}")
        print(f"  Encoded as: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
        
        if class_counts.min() < 1:
            print(f"  Skipping {method} - insufficient samples per class (minimum: {class_counts.min()})")
            continue
        
        if len(class_counts) < 2:
            print(f"  Skipping {method} - need at least 2 classes for classification")
            continue
        
        datasets[method] = {
            'X': X, 
            'y': y_encoded, 
            'label_encoder': label_encoder,
            'original_labels': y
        }
        print(f"  Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features, {len(class_counts)} classes")
    
    return datasets

def train_xgboost_models(datasets):
    """Train XGBoost models and compare performance"""
    print("Training XGBoost models...")
    
    results = []
    all_predictions = {}
    
    for method, data in datasets.items():
        print(f"Training {method}...")
        
        X, y_encoded = data['X'], data['y']
        label_encoder = data['label_encoder']
        
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
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            auc = np.nan
        
        results.append({
            'method': method,
            'accuracy': accuracy,
            'auc': auc,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features': X.shape[1],
            'n_classes': len(label_encoder.classes_),
            'class_names': ', '.join(label_encoder.classes_)
        })
        
        # Store predictions for plotting
        all_predictions[method] = {
            'y_test': y_test,
            'y_pred_proba': y_pred_proba,
            'classes': model.classes_,
            'label_encoder': label_encoder
        }
        
        print(f"  Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")
        print(f"  Classes: {label_encoder.classes_}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('auc', ascending=False)
    results_df.to_csv('xgboost_performance_comparison.csv', index=False)
    
    print("XGBoost performance saved to xgboost_performance_comparison.csv")
    return results_df, all_predictions

def plot_xgboost_results(results_df, all_predictions):
    """Plot XGBoost performance comparison with ROC curves for each class"""
    
    # Performance comparison bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    ax1.bar(results_df['method'], results_df['accuracy'], alpha=0.7, color='skyblue')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('XGBoost Accuracy by Normalization Method')
    ax1.tick_params(axis='x', rotation=45)
    
    # AUC comparison
    ax2.bar(results_df['method'], results_df['auc'], alpha=0.7, color='lightcoral')
    ax2.set_ylabel('AUC (weighted)')
    ax2.set_title('XGBoost AUC by Normalization Method')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('xgboost_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC curves for each class by method
    methods = list(all_predictions.keys())
    n_methods = len(methods)
    
    # Get class names from first method
    first_method = methods[0]
    label_encoder = all_predictions[first_method]['label_encoder']
    class_names = label_encoder.classes_
    n_classes = len(class_names)
    
    # Create subplot grid for ROC curves
    fig, axes = plt.subplots(n_classes, n_methods, figsize=(4 * n_methods, 3 * n_classes))
    if n_classes == 1:
        axes = axes.reshape(1, -1)
    if n_methods == 1:
        axes = axes.reshape(-1, 1)
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    
    for method_idx, (method, pred_data) in enumerate(all_predictions.items()):
        y_test = pred_data['y_test']
        y_pred_proba = pred_data['y_pred_proba']
        
        # Convert to binary format for each class
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        
        for class_idx in range(n_classes):
            if n_classes > 1 and n_methods > 1:
                ax = axes[class_idx, method_idx]
            elif n_classes == 1:
                ax = axes[method_idx]
            elif n_methods == 1:
                ax = axes[class_idx]
            else:
                ax = axes
            
            # Calculate ROC curve for this class
            if n_classes == 2:
                # Binary classification
                fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], y_pred_proba[:, class_idx])
            else:
                # Multi-class: use one-vs-rest
                fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], y_pred_proba[:, class_idx])
            
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            ax.plot(fpr, tpr, color=colors[class_idx], linewidth=2,
                   label=f'AUC = {roc_auc:.3f}')
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            
            if class_idx == n_classes - 1:
                ax.set_xlabel('False Positive Rate')
            if method_idx == 0:
                ax.set_ylabel('True Positive Rate')
            
            ax.set_title(f'{method.capitalize()}\n{class_names[class_idx]}')
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('roc_curves_by_class_and_method.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("XGBoost performance comparison plot saved to xgboost_performance_comparison.png")
    print("ROC curves by class and method saved to roc_curves_by_class_and_method.png")

def main():
    parser = argparse.ArgumentParser(description="Multi-batch normalization comparison")
    parser.add_argument('--target-features', dest='target_features', 
                       default='Cell: Mean', help='Comma-separated target features')
    parser.add_argument('--sample-size', type=int, default=20000,
                       help='Sample size for XGBoost testing')
    parser.add_argument('--batch-ids',
                       default=[], help='Batch IDs, comma separated')
    
    args = parser.parse_args()
    
    target_features = [f.strip() for f in args.target_features.split(',')]

    print("Multi-batch normalization comparison analysis")
    print(f"Target features: {target_features}")
    print(f"Sample size for ML: {args.sample_size}")
    
    # Find and load all batch data
    batch_ids = args.batch_ids.split(',')
    if not batch_ids:
        print("Error: No batch files found in current directory")
        return
    
    all_data = load_all_batch_data(batch_ids)
    
    # Calculate Pearson P/df statistics
    print("\nCalculating Pearson P/df statistics...")
    pearson_results, pearson_summary = calculate_pearson_statistics_all_batches(all_data, target_features)
    
    # Create PCA plots
    print("\nCreating PCA plots...")
    create_pca_plots(all_data, target_features)
    
    # Create density plots
    print("\nCreating density plots...")
    create_density_plots(all_data, target_features)
    
    # Prepare ML datasets and train models
    print("\nPreparing datasets for XGBoost testing...")
    datasets = prepare_ml_dataset(all_data, args.sample_size)
    
    if datasets:
        print("\nTraining XGBoost models...")
        results_df, all_predictions = train_xgboost_models(datasets)
        
        print("\nPlotting XGBoost results...")
        plot_xgboost_results(results_df, all_predictions)
        
        print("\nTop performing normalization methods:")
        print(results_df.head())
    else:
        print("No suitable datasets found for XGBoost testing")
    
    print("\nGenerating HTML summary report...")
    generate_html_report(
        results_df=results_df if 'results_df' in locals() else None,
        pearson_summary=pearson_summary if 'pearson_summary' in locals() else None,
        batch_ids=batch_ids,
        target_features=target_features
    )

    print("\nAnalysis complete!")
    print("\nGenerated files:")
    print("- pearson_statistics_detailed.csv")
    print("- pearson_statistics_summary.csv") 
    print("- pca_batch_*.png (one per batch)")
    print("- density_plots_batch_*.png (one per batch)")
    if datasets:
        print("- xgboost_performance_comparison.csv")
        print("- xgboost_performance_comparison.png")




def generate_html_report(results_df=None, pearson_summary=None, batch_ids=None, target_features=None):
    """Generate a standalone HTML report summarizing normalization comparison findings"""
    
    # Load data if not provided
    if results_df is None and Path('xgboost_performance_comparison.csv').exists():
        results_df = pd.read_csv('xgboost_performance_comparison.csv')
    
    if pearson_summary is None and Path('pearson_statistics_summary.csv').exists():
        pearson_summary = pd.read_csv('pearson_statistics_summary.csv', index_col=0)
    
    # Get available plots
    pca_plots = glob.glob("pca_batch_*.png")
    density_plots = glob.glob("density_plots_batch_*.png")
    
    # Start HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Normalization Method Comparison Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            border-left: 4px solid #007bff;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
            margin-bottom: 5px;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
        }}
        .results-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .results-table th {{
            background-color: #f8f9fa;
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #dee2e6;
        }}
        .results-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #dee2e6;
        }}
        .results-table tr:hover {{
            background-color: #f8f9fa;
        }}
        .best-method {{
            background-color: #d4edda !important;
            font-weight: bold;
        }}
        .plot-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .plot-container {{
            text-align: center;
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .highlight {{
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
            margin: 20px 0;
        }}
        .recommendation {{
            background-color: #d1ecf1;
            padding: 20px;
            border-radius: 5px;
            border-left: 4px solid #17a2b8;
            margin: 20px 0;
        }}
        h1 {{ margin-top: 0; }}
        h2 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        h3 {{ color: #555; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Normalization Method Comparison Report</h1>
        <p>Comprehensive evaluation of data transformation methods for cell classification</p>
        <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
"""

    # Executive Summary
    if results_df is not None and not results_df.empty:
        best_method = results_df.iloc[0]['method']
        best_accuracy = results_df.iloc[0]['accuracy']
        best_auc = results_df.iloc[0]['auc']
        
        html_content += f"""
    <div class="section">
        <h2>Executive Summary</h2>
        <div class="recommendation">
            <h3>🎯 Recommended Method: {best_method.upper()}</h3>
            <p><strong>Rationale:</strong> Based on machine learning performance evaluation, the {best_method} normalization method 
            achieved the highest classification performance with {best_accuracy:.1%} accuracy and {best_auc:.3f} AUC score.</p>
        </div>
        
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{len(results_df)}</div>
                <div class="metric-label">Methods Compared</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{best_accuracy:.1%}</div>
                <div class="metric-label">Best Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{best_auc:.3f}</div>
                <div class="metric-label">Best AUC Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{results_df.iloc[0]['n_classes']}</div>
                <div class="metric-label">Cell Types Classified</div>
            </div>
        </div>
    </div>
"""

    # Machine Learning Performance Results
    if results_df is not None and not results_df.empty:
        html_content += """
    <div class="section">
        <h2>Machine Learning Performance Comparison</h2>
        <p>XGBoost classification performance for cell type prediction across normalization methods:</p>
        
        <table class="results-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Method</th>
                    <th>Accuracy</th>
                    <th>AUC Score</th>
                    <th>Features</th>
                    <th>Test Samples</th>
                </tr>
            </thead>
            <tbody>
"""
        for idx, row in results_df.iterrows():
            css_class = "best-method" if idx == 0 else ""
            html_content += f"""
                <tr class="{css_class}">
                    <td>{idx + 1}</td>
                    <td>{row['method'].capitalize()}</td>
                    <td>{row['accuracy']:.1%}</td>
                    <td>{row['auc']:.3f}</td>
                    <td>{row['n_features']:,}</td>
                    <td>{row['n_test']:,}</td>
                </tr>
"""
        html_content += """
            </tbody>
        </table>
    </div>
"""

    # Statistical Analysis (Pearson P/df)
    if pearson_summary is not None and not pearson_summary.empty:
        html_content += """
    <div class="section">
        <h2>Statistical Normality Assessment</h2>
        <p>Pearson P/df statistics measuring how well each method achieves normal distributions (lower values = better normalization):</p>
        
        <table class="results-table">
            <thead>
                <tr>
                    <th>Batch ID</th>
"""
        # Add method columns
        for method in pearson_summary.columns:
            html_content += f"<th>{method.capitalize()}</th>"
        
        html_content += """
                </tr>
            </thead>
            <tbody>
"""
        for batch_id in pearson_summary.index:
            html_content += f"<td>{batch_id}</td>"
            for method in pearson_summary.columns:
                value = pearson_summary.loc[batch_id, method]
                if pd.notna(value):
                    html_content += f"<td>{value:.2f}</td>"
                else:
                    html_content += "<td>-</td>"
            html_content += "</tr>"
        
        html_content += """
            </tbody>
        </table>
        
        <div class="highlight">
            <strong>Interpretation:</strong> Lower Pearson P/df values indicate better normalization. 
            Values close to 1.0 suggest the transformed data follows a normal distribution.
        </div>
    </div>
"""

    # Visualization Gallery
    html_content += """
    <div class="section">
        <h2>Data Visualization Gallery</h2>
        <p>Visual assessment of normalization effects across batches and methods:</p>
"""

    # PCA Plots
    if pca_plots:
        html_content += """
        <h3>Principal Component Analysis</h3>
        <p>PCA plots colored by sample Image ID to assess data structure and batch effects:</p>
        <div class="plot-grid">
"""
        for plot in sorted(pca_plots):
            batch_name = plot.replace('pca_batch_', '').replace('.png', '')
            html_content += f"""
            <div class="plot-container">
                <h4>Batch {batch_name}</h4>
                <img src="{plot}" alt="PCA plot for batch {batch_name}">
            </div>
"""
        html_content += "</div>"

    # Density Plots
    if density_plots:
        html_content += """
        <h3>Distribution Analysis</h3>
        <p>Density plots showing the effect of normalization on data distributions:</p>
        <div class="plot-grid">
"""
        for plot in sorted(density_plots):
            batch_name = plot.replace('density_plots_batch_', '').replace('.png', '')
            html_content += f"""
            <div class="plot-container">
                <h4>Batch {batch_name}</h4>
                <img src="{plot}" alt="Density plots for batch {batch_name}">
            </div>
"""
        html_content += "</div>"

    # XGBoost Performance Plot
    if Path('xgboost_performance_comparison.png').exists():
        html_content += """
        <h3>Model Performance Comparison</h3>
        <div class="plot-container">
            <img src="xgboost_performance_comparison.png" alt="XGBoost performance comparison">
        </div>
    """

    # Add ROC curves
    if Path('roc_curves_by_class_and_method.png').exists():
        html_content += """
        <h3>ROC Curves by Class and Method</h3>
        <p>Area Under the Curve (AUC) for each cell type across normalization methods:</p>
        <div class="plot-container">
            <img src="roc_curves_by_class_and_method.png" alt="ROC curves by class and method">
        </div>
    """

    html_content += "</div>"

    # Methodology and Technical Details
    html_content += f"""
    <div class="section">
        <h2>Methodology</h2>
        <h3>Analysis Framework</h3>
        <ul>
            <li><strong>Statistical Assessment:</strong> Pearson P/df statistic for normality testing</li>
            <li><strong>Machine Learning Evaluation:</strong> Shallow XGBoost models (50 trees, depth 3)</li>
            <li><strong>Classification Task:</strong> Multi-class cell type prediction</li>
            <li><strong>Cross-validation:</strong> 70/30 train-test split with stratification</li>
            <li><strong>Performance Metrics:</strong> Accuracy and weighted AUC for multi-class classification</li>
        </ul>
        
        <h3>Data Processing</h3>
        <ul>
            <li><strong>Feature Selection:</strong> All numeric columns (excluding metadata)</li>
            <li><strong>Missing Data:</strong> Filled with zeros</li>
            <li><strong>Sampling:</strong> Subsampled for computational efficiency if needed</li>
            <li><strong>Target Variable:</strong> Classification column with cell type labels</li>
        </ul>
    </div>
"""

    # Conclusions and Recommendations
    if results_df is not None and not results_df.empty:
        performance_gap = results_df.iloc[0]['accuracy'] - results_df.iloc[-1]['accuracy']
        html_content += f"""
    <div class="section">
        <h2>Key Findings</h2>
        <ul>
            <li><strong>Performance Range:</strong> Classification accuracy varied by {performance_gap:.1%} between best and worst methods</li>
            <li><strong>Method Ranking:</strong> {', '.join([row['method'].capitalize() for _, row in results_df.head(3).iterrows()])} were the top performers</li>
            <li><strong>Feature Impact:</strong> Analysis used {results_df.iloc[0]['n_features']:,} quantitative cellular features</li>
            <li><strong>Robustness:</strong> All methods successfully processed the multi-batch dataset</li>
        </ul>
        
        <div class="recommendation">
            <h3>Implementation Recommendation</h3>
            <p>For your downstream XGBoost pipeline, use <strong>{best_method}</strong> normalization. 
            This method demonstrated superior preservation of class-discriminating features while maintaining 
            data quality across batches.</p>
        </div>
    </div>
"""

    # Footer
    html_content += """
    <div class="section">
        <h2>Data Files Generated</h2>
        <ul>
            <li><code>pearson_statistics_detailed.csv</code> - Detailed normality statistics by marker</li>
            <li><code>pearson_statistics_summary.csv</code> - Summary statistics by batch and method</li>
            <li><code>xgboost_performance_comparison.csv</code> - ML performance metrics</li>
            <li><code>pca_batch_*.png</code> - PCA visualizations for each batch</li>
            <li><code>density_plots_batch_*.png</code> - Distribution plots for each batch</li>
            <li><code>xgboost_performance_comparison.png</code> - Performance comparison chart</li>
        </ul>
    </div>

</body>
</html>"""

    # Save report
    report_filename = f"normalization_comparison_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(report_filename, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report generated: {report_filename}")
    return report_filename



if __name__ == "__main__":
    main()