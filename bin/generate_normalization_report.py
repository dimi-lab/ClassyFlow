#!/usr/bin/env python3
"""
Normalization Report Generator for Cell Type Classification Pipeline

This script reads normalization outputs from the four transformer scripts (Box-Cox, Quantile, MinMax, Log)
and generates a comprehensive HTML report section using Jinja2 templates.

Usage:
    python generate_normalization_report.py --norm-dir /path/to/norm/output --output-file normalization_report.html
"""

import os
import json
import glob
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from jinja2 import Environment, FileSystemLoader, select_autoescape

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def number_format(value):
    """Custom Jinja2 filter for number formatting."""
    if value is None:
        return "N/A"
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
    jinja_env.filters['default'] = lambda v, d="N/A": v if v is not None else d
    jinja_env.filters['format'] = lambda fmt, v: fmt % v if v is not None else "N/A"
    jinja_env.filters['join'] = lambda lst, sep=", ": sep.join(str(x) for x in lst) if lst else ""
    jinja_env.filters['title'] = lambda s: s.title() if s else ""
    
    return jinja_env


def read_transformation_results(norm_dir: Path, transformation_type: str) -> List[Dict[str, Any]]:
    """
    Read results from a specific transformation type.
    
    Args:
        norm_dir: Directory containing normalization outputs
        transformation_type: Type of transformation (boxcox, quantile, minmax, log)
    
    Returns:
        List of transformation results for each batch
    """
    results = []
    pattern = f"{transformation_type}_results_*.json"
    result_files = glob.glob(str(norm_dir / pattern))
    
    for file_path in sorted(result_files):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            batch_name = data.get('batch_name', 'Unknown')
            
            # Ensure transformation_type is set in data
            if 'transformation_type' not in data:
                data['transformation_type'] = transformation_type
            
            # Ensure required fields exist with defaults
            data.setdefault('total_cells', 0)
            data.setdefault('total_slides', 0)
            data.setdefault('total_markers', 0)
            data.setdefault('quantile_splits', 100)
            
            # Process Box-Cox specific metrics
            if transformation_type == 'boxcox' and 'boxcox_metrics' in data:
                metrics = data['boxcox_metrics']
                metrics.setdefault('successful_transforms', 0)
                metrics.setdefault('total_features', 0)
                metrics.setdefault('failed_transforms', 0)
                
                if 'lambda_stats' not in metrics:
                    metrics['lambda_stats'] = {
                        'mean_lambda': None,
                        'median_lambda': None
                    }
            
            # Process MinMax specific metrics
            if transformation_type == 'minmax' and 'minmax_metrics' in data:
                metrics = data['minmax_metrics']
                metrics.setdefault('total_features_transformed', data.get('total_markers', 0))
                
                if 'scaling_stats' not in metrics:
                    metrics['scaling_stats'] = {
                        'min_transformed_value': -2.0,
                        'max_transformed_value': 2.0
                    }
            
            # Process CV metrics
            if 'cv_metrics' in data:
                cv_metrics = data['cv_metrics']
                cv_metrics.setdefault('mean_cv_improvement', None)
                cv_metrics.setdefault('median_cv_improvement', None)
                cv_metrics.setdefault('markers_improved', 0)
                cv_metrics.setdefault('markers_worsened', 0)
            
            # Process worst performing markers for display
            worst_markers = data.get('worst_performing_markers', [])
            if worst_markers and len(worst_markers) > 3:
                worst_markers_display = ', '.join(worst_markers[:3]) + f" (+{len(worst_markers)-3} more)"
            else:
                worst_markers_display = ', '.join(worst_markers) if worst_markers else "N/A"
            
            # Get the HTML file path for all transformation plots
            html_file = f"pages/{transformation_type}_all_plots_{batch_name}.html"
            
            # Get the GMM gating report file path
            gmm_report_file = f"pages/gmm_gated_{batch_name}.html"
                    
            result = {
                'batch_name': batch_name,
                'data': data,
                'worst_performing_markers': worst_markers_display,
                'transformation_plots_html': html_file,
                'gmm_gating_report': gmm_report_file
            }
            
            results.append(result)
            logger.info(f"Loaded {transformation_type} results for batch: {batch_name}")
            
        except Exception as e:
            logger.error(f"Error reading {transformation_type} file {file_path}: {e}")
    
    return results


def collect_normalization_data(norm_dir: Path) -> Dict[str, Any]:
    """
    Collect all normalization data from the norm directory.
    
    Args:
        norm_dir: Directory containing normalization outputs
    
    Returns:
        Dictionary containing all normalization data organized by method
    """
    norm_data = {
        'normalization_results': [],
        'normalization_methods': [],
        'batch_names': set(),
        'total_batches': 0,
        'total_markers': 0,
        'primary_method': None,
        'primary_method_display': "N/A",
        'comparison_mode': False,
        'methods_summary': {}
    }
    
    if not norm_dir.exists():
        logger.error(f"Normalization directory not found: {norm_dir}")
        return norm_data
    
    # Read results from all transformation types
    transformation_types = ['boxcox', 'quantile', 'minmax', 'log']
    all_methods = []
    all_batches = set()
    total_markers = 0
    
    for trans_type in transformation_types:
        results = read_transformation_results(norm_dir, trans_type)
        
        if results:
            all_methods.append(trans_type)
            
            # Collect batch names and marker counts
            for result in results:
                all_batches.add(result['batch_name'])
                norm_data['normalization_results'].append(result)
                
                # Get total markers from the first result (should be same across all)
                if total_markers == 0 and 'total_markers' in result['data']:
                    total_markers = result['data']['total_markers']
            
            # Calculate summary statistics for this method
            cv_improvements = []
            success_rates = []
            
            for result in results:
                if 'cv_metrics' in result['data']:
                    cv_metrics = result['data']['cv_metrics']
                    if cv_metrics.get('mean_cv_improvement') is not None:
                        cv_improvements.append(cv_metrics['mean_cv_improvement'])
                
                # Calculate success rate for Box-Cox
                if trans_type == 'boxcox' and 'boxcox_metrics' in result['data']:
                    metrics = result['data']['boxcox_metrics']
                    total = metrics.get('total_features', 0)
                    successful = metrics.get('successful_transforms', 0)
                    if total > 0:
                        success_rates.append(successful / total)
            
            # Store method summary
            norm_data['methods_summary'][trans_type] = {
                'batch_count': len(results),
                'avg_cv_improvement': sum(cv_improvements) / len(cv_improvements) if cv_improvements else None,
                'avg_success_rate': sum(success_rates) / len(success_rates) if success_rates else None
            }
    
    # Determine primary method and display mode
    if len(all_methods) == 0:
        primary_method = None
        display_text = "No normalization applied"
    elif len(all_methods) == 1:
        primary_method = all_methods[0]
        display_text = all_methods[0].title()
    else:
        primary_method = "Multiple"
        # Create a formatted list of methods
        methods_list = [m.title() for m in all_methods]
        display_text = f"Multiple ({', '.join(methods_list)})"
    
    # Update summary information
    norm_data.update({
        'normalization_methods': all_methods,
        'batch_names': list(all_batches),
        'total_batches': len(all_batches) if all_batches else "N/A",
        'total_markers': total_markers if total_markers else "N/A",
        'primary_method': primary_method,
        'primary_method_display': display_text,
        'comparison_mode': len(all_methods) > 1
    })
    
    # Sort results by batch name and then by method for consistent display
    norm_data['normalization_results'].sort(key=lambda x: (x['batch_name'], x['data'].get('transformation_type', '')))
    
    logger.info(f"Collected normalization data: {len(all_methods)} methods, {len(all_batches)} batches, {total_markers} markers")
    
    return norm_data


def generate_section_report(norm_data: Dict[str, Any],
                          output_file: Path,
                          jinja_env,
                          template_name: str = "normalization.html"):
    """
    Generate normalization section for inclusion in main report.
    
    Args:
        norm_data: Dictionary containing normalization data
        output_file: Path to output HTML file
        jinja_env: Jinja2 environment
        template_name: Name of the template file
    """
    try:
        template = jinja_env.get_template(template_name)
        
        # Prepare template context
        context = {
            'normalization_data': norm_data
        }
        
        # Render the template
        html_content = template.render(**context)
        
        # Write to output file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Normalization section generated: {output_file}")
        
    except Exception as e:
        logger.error(f"Error generating section report: {e}")
        raise


def export_normalization_json(norm_data: Dict[str, Any], output_file: Path):
    """
    Export normalization data as JSON for use by other scripts.
    
    Args:
        norm_data: Dictionary containing normalization data
        output_file: Path to output JSON file
    """
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a serializable copy of the data
        serializable_data = json.loads(json.dumps(norm_data, default=str))
        
        with open(output_file, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        logger.info(f"Normalization data exported to JSON: {output_file}")
    except Exception as e:
        logger.error(f"Error exporting JSON: {e}")


def main():
    """Main function to parse arguments and generate normalization report."""
    parser = argparse.ArgumentParser(
        description="Generate normalization report from transformer outputs"
    )
    parser.add_argument(
        '--norm-dir',
        type=Path,
        default=Path('./'),
        help='Directory containing normalization outputs (default: ./norm)'
    )
    parser.add_argument(
        '--output-file',
        type=Path,
        default=Path('./normalization_report.html'),
        help='Path to output HTML file (default: ./normalization_report.html)'
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
    
    # Collect normalization data
    logger.info(f"Collecting normalization data from: {args.norm_dir}")
    norm_data = collect_normalization_data(args.norm_dir)
    
    if not norm_data['normalization_results']:
        logger.warning("No normalization results found. The report will be empty.")
    
    # Set up Jinja2 environment
    jinja_env = setup_jinja_environment(str(args.template_dir))
    
    # Generate the report
    generate_section_report(norm_data, args.output_file, jinja_env)
    
    # Export JSON for debugging
    json_output_path = args.output_file.parent / "normalization_results.json"
    export_normalization_json(norm_data, json_output_path)
    
    # Save summary metrics for final report
    normalization_summary = {
        'num_batches': norm_data.get('total_batches', 0) if norm_data.get('total_batches', 0) > 0 else None
    }
    
    with open('normalization_summary.json', 'w') as f:
        json.dump(normalization_summary, f, indent=2)
    print("Saved normalization summary to normalization_summary.json")
    
    # Print summary
    print("\n" + "="*60)
    print("NORMALIZATION REPORT GENERATION COMPLETE")
    print("="*60)
    print(f"Output file: {args.output_file}")
    print(f"Methods processed: {', '.join(norm_data['normalization_methods']) if norm_data['normalization_methods'] else 'None'}")
    print(f"Batches processed: {norm_data['total_batches']}")
    print(f"Total markers: {norm_data['total_markers']}")
    print(f"Total result sets: {len(norm_data['normalization_results'])}")
    
    if norm_data['comparison_mode']:
        print("\nComparison Mode: Multiple normalization methods detected")
        print("Method Summary:")
        for method, summary in norm_data['methods_summary'].items():
            print(f"  - {method}: {summary['batch_count']} batches")
            if summary['avg_cv_improvement'] is not None:
                print(f"    Average CV improvement: {summary['avg_cv_improvement']:.4f}")
    
    print("="*60)


if __name__ == "__main__":
    main()