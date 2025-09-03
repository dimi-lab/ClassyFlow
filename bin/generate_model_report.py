#!/usr/bin/env python3
"""
Model Report Generator for Cell Type Classification Pipeline

This script reads model training and evaluation outputs and generates a comprehensive HTML report 
section using Jinja2 templates.

Usage:
    python generate_model_report.py --modeling-dir /path/to/modeling --output-file modeling_report.html
"""

import os
import json
import glob
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

from jinja2 import Environment, FileSystemLoader, select_autoescape

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_train_holdout_comparison_plot(training_df, holdout_df, class_column, output_path):
    """Create side-by-side comparison of training vs holdout class distributions with statistical test"""
    
    # Get class counts for both datasets
    train_counts = training_df[class_column].value_counts()
    holdout_counts = holdout_df[class_column].value_counts()
    
    # Get all unique classes and sort by training set frequency (descending)
    all_classes = sorted(train_counts.index, key=lambda x: train_counts.get(x, 0), reverse=True)
    
    # Calculate proportions
    train_props = [(train_counts.get(cls, 0) / len(training_df)) * 100 for cls in all_classes]
    holdout_props = [(holdout_counts.get(cls, 0) / len(holdout_df)) * 100 for cls in all_classes]
    
    # Perform Chi-square test
    contingency_table = pd.DataFrame({
        'Training': [train_counts.get(cls, 0) for cls in all_classes],
        'Holdout': [holdout_counts.get(cls, 0) for cls in all_classes]
    }, index=all_classes)
    
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table.T)
    
    # Create side-by-side plot with standardized size
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.linewidth': 1,
        'grid.linewidth': 0.5
    })
    
    # Training set plot
    colors1 = plt.cm.viridis(np.linspace(0.2, 0.9, len(all_classes)))
    bars1 = ax1.barh(all_classes, train_props, color=colors1, alpha=0.8, 
                     edgecolor='white', linewidth=1.5)
    
    ax1.set_xlabel('Percentage of Total Samples', fontsize=11, fontweight='bold', color='#2c3e50')
    ax1.set_title('Training Set Distribution', fontsize=13, fontweight='bold', color='#2c3e50', pad=15)
    ax1.grid(True, alpha=0.3, axis='x', linestyle='-', color='#bdc3c7')
    ax1.set_axisbelow(True)
    ax1.set_facecolor('#f8f9fa')
    
    # Add value labels
    for bar, prop, count in zip(bars1, train_props, [train_counts.get(cls, 0) for cls in all_classes]):
        width = bar.get_width()
        ax1.text(width + max(train_props)*0.01, bar.get_y() + bar.get_height()/2, 
                f'{prop:.1f}% (n={count:,})', ha='left', va='center', fontweight='bold', 
                fontsize=10, color='#2c3e50')
    
    # Holdout set plot (same order)
    colors2 = plt.cm.viridis(np.linspace(0.2, 0.9, len(all_classes)))
    bars2 = ax2.barh(all_classes, holdout_props, color=colors2, alpha=0.8, 
                     edgecolor='white', linewidth=1.5)
    
    ax2.set_xlabel('Percentage of Total Samples', fontsize=11, fontweight='bold', color='#2c3e50')
    ax2.set_title('Holdout Set Distribution', fontsize=13, fontweight='bold', color='#2c3e50', pad=15)
    ax2.grid(True, alpha=0.3, axis='x', linestyle='-', color='#bdc3c7')
    ax2.set_axisbelow(True)
    ax2.set_facecolor('#f8f9fa')
    
    # Add value labels
    for bar, prop, count in zip(bars2, holdout_props, [holdout_counts.get(cls, 0) for cls in all_classes]):
        width = bar.get_width()
        ax2.text(width + max(holdout_props)*0.01, bar.get_y() + bar.get_height()/2, 
                f'{prop:.1f}% (n={count:,})', ha='left', va='center', fontweight='bold', 
                fontsize=10, color='#2c3e50')
    
    # Style both axes
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#bdc3c7')
        ax.spines['bottom'].set_color('#bdc3c7')
        ax.tick_params(axis='both', which='major', labelsize=11, colors='#2c3e50')
    
    # Add statistical test result as subtitle
    # significance_text = f"Statistical Test: χ² = {chi2_stat:.2f}, p = {p_value:.4f}"
    # if p_value < 0.05:
    #     significance_text += " (Significantly Different)"
    #     fig.suptitle(f'Training vs. Holdout Class Distribution Comparison\n{significance_text}', 
    #                 fontsize=14, fontweight='bold', color='#e74c3c', y=0.95)
    # else:
    #     significance_text += " (Not Significantly Different)"
    #     fig.suptitle(f'Training vs. Holdout Class Distribution Comparison\n{significance_text}', 
    #                 fontsize=14, fontweight='bold', color='#27ae60', y=0.95)
    
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Train-holdout comparison plot saved: {output_path}")
    
    return {
        'chi2_statistic': float(chi2_stat),
        'p_value': float(p_value),
        'degrees_of_freedom': int(dof),
        'is_significantly_different': bool(p_value < 0.05)
    }


def number_format(value):
    """Custom Jinja2 filter for number formatting."""
    if isinstance(value, (int, float)):
        return f"{value:,}"
    return value


def setup_jinja_environment(template_dir: str):
    """Set up and return Jinja2 environment with custom filters."""
    jinja_env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(['html', 'xml'])
    )
    
    # Add custom filters
    jinja_env.filters['number_format'] = number_format
    
    return jinja_env


def read_holdout_evaluation_results(modeling_dir: Path) -> List[Dict[str, Any]]:
    """
    Read holdout evaluation results from the modeling directory.
    
    Args:
        modeling_dir: Directory containing modeling outputs
    
    Returns:
        List of holdout evaluation results
    """
    results = []
    
    # Find all holdout evaluation JSON files
    json_files = glob.glob(str(modeling_dir / "holdoutEval*.json"))
    
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract model name from filename
            filename = os.path.basename(json_file)
            if '_Model_' in filename:
                model_name = filename.split('_Model_')[1].split('_results')[0]
            else:
                model_name = filename.replace('holdoutEval_', '').replace('_results.json', '')
            
            # Update plot paths to use plots subdirectory
            plot_fields = [
                'class_distribution_plot_path',
                'confusion_matrix_csv_path',
                'roc_curves_plot_path'
            ]
            
            for field in plot_fields:
                if field in data and data[field]:
                    filename = os.path.basename(data[field])
                    data[field] = f"plots/{filename}"
            
            result = {
                'model_name': model_name,
                'data': data
            }
            
            results.append(result)
            logger.info(f"Loaded holdout evaluation for model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error reading holdout evaluation file {json_file}: {e}")
    
    return results


def read_model_comparison_results(modeling_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Read model comparison results (xgbWinners output).
    
    Args:
        modeling_dir: Directory containing modeling outputs
    
    Returns:
        Model comparison data or None if not found
    """
    comparison_file = modeling_dir / "xgbWinners_results.json"
    
    if not comparison_file.exists():
        logger.warning(f"Model comparison file not found: {comparison_file}")
        return None
    
    try:
        with open(comparison_file, 'r') as f:
            data = json.load(f)
        
        # Update plot paths to use plots subdirectory
        plot_fields = [
            'class_distribution_plot_path',
            'parameter_search_plot_path'
        ]
        
        for field in plot_fields:
            if field in data and data[field]:
                filename = os.path.basename(data[field])
                data[field] = f"plots/{filename}"
        
        # Update CSV paths to use plots subdirectory
        csv_fields = [
            'classes_summary_path',
            'parameter_summary_csv_path'
        ]
        
        for field in csv_fields:
            if field in data and data[field]:
                filename = os.path.basename(data[field])
                data[field] = f"plots/{filename}"
        
        logger.info("Loaded model comparison results")
        return data
        
    except Exception as e:
        logger.error(f"Error reading model comparison file {comparison_file}: {e}")
        return None


def calculate_modeling_summary(holdout_results: List[Dict[str, Any]], 
                             comparison_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate summary statistics for modeling results.
    
    Args:
        holdout_results: List of holdout evaluation results
        comparison_data: Model comparison data
    
    Returns:
        Dictionary of summary statistics
    """
    summary = {
        'total_models_evaluated': len(holdout_results),
        'best_accuracy': 0.0,
        'best_f1_score': 0.0,
        'training_classes': 0,
        'class_imbalance_detected': False,
        'best_performing_class': 'N/A',
        'worst_performing_class': 'N/A'
    }
    
    if not holdout_results:
        return summary
    
    # Find best overall performance metrics
    best_accuracy = max([r['data'].get('accuracy', 0) for r in holdout_results])
    best_f1 = max([r['data'].get('f1_score', 0) for r in holdout_results])
    
    summary.update({
        'best_accuracy': best_accuracy,
        'best_f1_score': best_f1
    })
    
    # Use first model (typically the best) for class-level statistics
    first_model = holdout_results[0]['data']
    summary.update({
        'training_classes': first_model.get('n_classes', 0),
        'class_imbalance_detected': first_model.get('class_imbalance_detected', False),
        'best_performing_class': first_model.get('max_auc', {}).get('class_name', 'N/A'),
        'worst_performing_class': first_model.get('min_auc', {}).get('class_name', 'N/A')
    })
    
    return summary


def collect_modeling_data(modeling_dir: Path) -> Dict[str, Any]:
    """
    Collect all modeling data from the modeling directory.
    
    Args:
        modeling_dir: Directory containing modeling outputs
    
    Returns:
        Dictionary containing all modeling data
    """
    modeling_data = {
        'holdout_evaluations': [],
        'model_comparison': None,
        'summary_statistics': {},
        'total_models': 0
    }
    
    if not modeling_dir.exists():
        logger.error(f"Modeling directory not found: {modeling_dir}")
        return modeling_data
    
    # Read holdout evaluation results
    holdout_results = read_holdout_evaluation_results(modeling_dir)
    
    # Read model comparison results
    comparison_data = read_model_comparison_results(modeling_dir)
    
    # Calculate summary statistics
    summary_stats = calculate_modeling_summary(holdout_results, comparison_data)
    
    # Sort holdout results (First model first, Second model second, others last)
    def sort_models(model_eval):
        model_name = model_eval['model_name']
        if 'First' in model_name:
            return 0  # First priority
        elif 'Second' in model_name:
            return 1  # Second priority
        else:
            return 2  # Everything else last
    
    holdout_results.sort(key=sort_models)
    
    # Update data structure
    modeling_data.update({
        'holdout_evaluations': holdout_results,
        'model_comparison': comparison_data,
        'summary_statistics': summary_stats,
        'total_models': len(holdout_results)
    })
    
    logger.info(f"Collected modeling data: {len(holdout_results)} models evaluated")
    
    return modeling_data


def generate_section_report(modeling_data: Dict[str, Any],
                          output_file: Path,
                          jinja_env,
                          template_name: str = "modeling.html"):
    """
    Generate modeling section for inclusion in main report.
    
    Args:
        modeling_data: Dictionary containing modeling data
        output_file: Path to output HTML file
        jinja_env: Jinja2 environment
        template_name: Name of the template file
    """
    try:
        template = jinja_env.get_template(template_name)
        
        # Extract key metrics for easy template access
        summary = modeling_data.get('summary_statistics', {})
        
        # Prepare template context
        context = {
            'modeling_data': modeling_data,
            'holdout_accuracy': summary.get('best_accuracy', 0),
            'f1_score': summary.get('best_f1_score', 0),
            'training_classes': summary.get('training_classes', 0),
            'best_performing_class': summary.get('best_performing_class', 'N/A'),
            'worst_performing_class': summary.get('worst_performing_class', 'N/A')
        }
        
        # Render the template
        html_content = template.render(**context)
        
        # Write to output file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Modeling section generated: {output_file}")
        
        # Save summary metrics for final report
        model_summary = {
            'model_accuracy': round(summary.get('best_accuracy', 0) * 100, 1) if summary.get('best_accuracy', 0) > 0 else None,
            'training_cell_types': summary.get('training_classes', 0) if summary.get('training_classes', 0) > 0 else None
        }
        
        with open('model_summary.json', 'w') as f:
            json.dump(model_summary, f, indent=2)
        logger.info("Saved model summary metrics to model_summary.json")
        
    except Exception as e:
        logger.error(f"Error generating section report: {e}")
        raise


def export_modeling_json(modeling_data: Dict[str, Any], output_file: Path):
    """
    Export modeling data as JSON for use by other scripts.
    
    Args:
        modeling_data: Dictionary containing modeling data
        output_file: Path to output JSON file
    """
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(modeling_data, f, indent=2, default=str)
        logger.info(f"Modeling data exported to JSON: {output_file}")
    except Exception as e:
        logger.error(f"Error exporting JSON: {e}")


def main():
    """Main function to parse arguments and generate modeling report."""
    parser = argparse.ArgumentParser(
        description="Generate modeling report from pipeline outputs"
    )
    parser.add_argument(
        '--modeling-dir',
        type=Path,
        default=Path('./'),
        help='Directory containing modeling outputs (default: ./)'
    )
    parser.add_argument(
        '--output-file',
        type=Path,
        default=Path('./modeling_report.html'),
        help='Path to output HTML file (default: ./modeling_report.html)'
    )
    parser.add_argument(
        '--template-dir',
        type=Path,
        default=Path('templates'),
        help='Directory containing Jinja2 templates (default: templates)'
    )
    parser.add_argument(
        '--training-dataframe',
        type=Path,
        help='Path to training dataframe pickle (for comparison plot)'
    )
    parser.add_argument(
        '--holdout-dataframe', 
        type=Path,
        help='Path to holdout dataframe pickle (for comparison plot)'
    )
    parser.add_argument(
        '--class-column',
        type=str,
        help='Name of the class column'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Collect modeling data
    logger.info(f"Collecting modeling data from: {args.modeling_dir}")
    modeling_data = collect_modeling_data(args.modeling_dir)

    train_df = pd.read_pickle(args.training_dataframe)
    holdout_df = pd.read_pickle(args.holdout_dataframe)
    
    comparison_plot_path = "train_holdout_comparison.png"
    os.makedirs("plots", exist_ok=True)
    
    stats_results = create_train_holdout_comparison_plot(
        train_df, holdout_df, args.class_column, comparison_plot_path
    )
    
    # Add to modeling_data
    modeling_data['train_holdout_comparison'] = {
        'plot_path': f"plots/{comparison_plot_path}",
        'statistics': stats_results
    }
    
    if not modeling_data['holdout_evaluations']:
        logger.warning("No holdout evaluation results found. The report will have limited content.")
    
    # Set up Jinja2 environment
    jinja_env = setup_jinja_environment(str(args.template_dir))
    
    # Generate section report
    generate_section_report(modeling_data, args.output_file, jinja_env)
    
    # Export JSON for debugging/integration
    json_output = args.output_file.with_suffix('.json')
    export_modeling_json(modeling_data, json_output)
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL REPORT GENERATION COMPLETE")
    print("="*60)
    print(f"Output file: {args.output_file}")
    print(f"Models evaluated: {modeling_data['total_models']}")
    
    if modeling_data['summary_statistics']:
        stats = modeling_data['summary_statistics']
        print(f"\nSummary Statistics:")
        print(f"  - Best accuracy: {stats['best_accuracy']:.3f}")
        print(f"  - Best F1 score: {stats['best_f1_score']:.3f}")
        print(f"  - Training classes: {stats['training_classes']}")
        print(f"  - Best performing class: {stats['best_performing_class']}")
        print(f"  - Worst performing class: {stats['worst_performing_class']}")
    
    print("="*60)


if __name__ == "__main__":
    main()