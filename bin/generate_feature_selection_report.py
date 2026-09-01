#!/usr/bin/env python3
"""
Feature Selection Report Generator for Cell Type Classification Pipeline

This script reads feature selection outputs and generates a comprehensive HTML report 
section using Jinja2 templates.

Usage:
    python generate_feature_selection_report.py --fs-dir /path/to/feature_selection --output-file feature_selection_report.html
"""

import json
import glob
import argparse
from pathlib import Path
from typing import Dict, Any, List
import logging
import pandas as pd
import numpy as np

from jinja2 import Environment, FileSystemLoader, select_autoescape

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def read_feature_selection_results(fs_dir: Path) -> List[Dict[str, Any]]:
    """
    Read feature selection results from the feature_selection directory.
    
    Args:
        fs_dir: Directory containing feature selection outputs
    
    Returns:
        List of feature selection results for each cell type
    """
    results = []
    
    # Find all feature selection JSON files
    json_files = glob.glob(str(fs_dir / "feature_selection_*_results.json"))
    
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            celltype = data.get('celltype', 'Unknown')
            
            # Read the corresponding top features CSV if it exists
            safe_celltype = celltype.replace(' ', '_').replace('|', '_').replace('/', '')
            features_csv = fs_dir / f"top_rank_features_{safe_celltype}.csv"
            
            selected_features = []
            if features_csv.exists():
                try:
                    df = pd.read_csv(features_csv)
                    selected_features = df['Features'].tolist() if 'Features' in df.columns else []
                except Exception as e:
                    logger.warning(f"Could not read features CSV for {celltype}: {e}")
            
            # {feature: importance} for the template's per-feature score lookup.
            # Source is the ranked list written by rank_selected_features() in
            # generate_cell_type_selection.py (which also carries the sign).
            feature_importance = {r['feature']: r['importance']
                                  for r in data.get('feature_importance', []) or []}
            
            # Calculate warning threshold (2% of total features)
            warning_threshold = 0.02
            total_features = data.get('feature_selection_summary', {}).get('original_features', 100)
            optimal_features = data.get('optimal_n_features', 0)
            min_features_threshold = max(3, int(total_features * warning_threshold))
            
            # Add warning flag
            rfe_warning = optimal_features < min_features_threshold
            
            # Process stability data if available
            stability_data = data.get('feature_stability', {})
            
            # Process correlation data if available
            correlation_data = data.get('feature_correlations', {})
            
            # Get CV folds information
            cv_folds = data.get('cv_folds', data.get('n_folds', 'N/A'))
            
            result = {
                'celltype': celltype,
                'data': data,
                'selected_features': selected_features,
                'selected_features_count': len(selected_features),
                'feature_importance': feature_importance,
                'rfe_warning': rfe_warning,
                'min_features_threshold': min_features_threshold,
                'stability_data': stability_data,
                'correlation_data': correlation_data,
                'cv_folds': cv_folds
            }
            
            results.append(result)
            logger.info(f"Loaded feature selection results for {celltype}")
            
        except Exception as e:
            logger.error(f"Error reading feature selection file {json_file}: {e}")
    
    return results


def calculate_summary_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate summary statistics across all cell types.
    
    Args:
        results: List of feature selection results
    
    Returns:
        Dictionary of summary statistics
    """
    if not results:
        return {}
    
    total_celltypes = len(results)
    
    # Calculate unique feature counts instead of averages
    # For original features, use the maximum count as approximation since all cell types 
    # typically start with the same feature set
    max_original_features = max([
        r['data'].get('feature_selection_summary', {}).get('original_features', 0) 
        for r in results
    ], default=0)
    
    # Collect all unique selected features across cell types
    all_selected_features = set()
    for r in results:
        if r['selected_features']:
            all_selected_features.update(r['selected_features'])
    
    unique_original_features = max_original_features
    unique_selected_features = len(all_selected_features)
    
    avg_reduction_rate = np.mean([
        (1 - r['data'].get('optimal_n_features', 0) / 
         r['data'].get('feature_selection_summary', {}).get('original_features', 1)) * 100
        for r in results if r['data'].get('feature_selection_summary', {}).get('original_features', 0) > 0
    ])
    
    # Count warnings
    celltypes_with_warnings = sum(1 for r in results if r['rfe_warning'])
    
    # Get CV folds range
    cv_folds_list = [r['cv_folds'] for r in results if r['cv_folds'] != 'N/A']
    cv_folds_summary = f"{min(cv_folds_list)}-{max(cv_folds_list)}" if cv_folds_list else "N/A"
    
    summary = {
        'total_celltypes': total_celltypes,
        'unique_original_features': int(unique_original_features),
        'unique_selected_features': int(unique_selected_features),
        'avg_reduction_rate': round(avg_reduction_rate, 1),
        'celltypes_with_warnings': celltypes_with_warnings,
        'cv_folds_summary': cv_folds_summary
    }
    
    return summary


def collect_feature_selection_data(fs_dir: Path) -> Dict[str, Any]:
    """
    Collect all feature selection data from the feature_selection directory.
    
    Args:
        fs_dir: Directory containing feature selection outputs
    
    Returns:
        Dictionary containing all feature selection data
    """
    fs_data = {
        'feature_selection_results': [],
        'total_cell_types': 0,
        'summary_statistics': {}
    }
    
    if not fs_dir.exists():
        logger.error(f"Feature selection directory not found: {fs_dir}")
        return fs_data
    
    # Read feature selection results
    results = read_feature_selection_results(fs_dir)
    
    if not results:
        logger.warning("No feature selection results found.")
        return fs_data
    
    # Calculate summary statistics
    summary_stats = calculate_summary_statistics(results)
    
    # Update data structure
    fs_data.update({
        'feature_selection_results': results,
        'total_cell_types': len(results),
        'summary_statistics': summary_stats
    })
    
    logger.info(f"Collected feature selection data: {len(results)} cell types")
    
    return fs_data

def generate_section_report(fs_data: Dict[str, Any],
                          output_file: Path,
                          jinja_env,
                          template_name: str = "feature-selection.html"):
    """
    Generate feature selection section for inclusion in main report.
    
    Args:
        fs_data: Dictionary containing feature selection data
        output_file: Path to output HTML file
        jinja_env: Jinja2 environment
        template_name: Name of the template file
    """
    try:
        template = jinja_env.get_template(template_name)
        
        # Prepare template context
        context = {
            'feature_selection_data': fs_data
        }
        
        # Render the template
        html_content = template.render(**context)
        
        # Write to output file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Feature selection section generated: {output_file}")
        
    except Exception as e:
        logger.error(f"Error generating section report: {e}")
        raise


def main():
    """Main function to parse arguments and generate feature selection report."""
    parser = argparse.ArgumentParser(
        description="Generate feature selection report from pipeline outputs"
    )
    parser.add_argument(
        '--fs-dir',
        type=Path,
        default=Path('./'),
        help='Directory containing feature selection outputs (default: ./feature_selection)'
    )
    parser.add_argument(
        '--output-file',
        type=Path,
        default=Path('./feature_selection_report.html'),
        help='Path to output HTML file (default: ./feature_selection_report.html)'
    )
    parser.add_argument(
        '--template-dir',
        type=Path,
        default=Path('templates'),
        help='Directory containing Jinja2 templates (default: templates)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Collect feature selection data
    logger.info(f"Collecting feature selection data from: {args.fs_dir}")
    fs_data = collect_feature_selection_data(args.fs_dir)
    
    if not fs_data['feature_selection_results']:
        logger.warning("No feature selection results found. The report will be empty.")
    
    # Set up Jinja2 environment
    jinja_env = setup_jinja_environment(str(args.template_dir))
    
    generate_section_report(fs_data, args.output_file, jinja_env)

    # Print summary
    print("\n" + "="*60)
    print("FEATURE SELECTION REPORT GENERATION COMPLETE")
    print("="*60)
    print(f"Output file: {args.output_file}")
    print(f"Cell types processed: {fs_data['total_cell_types']}")
    
    if fs_data['summary_statistics']:
        stats = fs_data['summary_statistics']
        print(f"\nSummary Statistics:")
        print(f"  - Total unique original features: {stats['unique_original_features']}")
        print(f"  - Total unique selected features: {stats['unique_selected_features']}")
        print(f"  - Average reduction rate: {stats['avg_reduction_rate']}%")
        print(f"  - Cell types with RFE warnings: {stats['celltypes_with_warnings']}")
        print(f"  - CV folds used: {stats['cv_folds_summary']}")
    
    print("="*60)


if __name__ == "__main__":
    main()