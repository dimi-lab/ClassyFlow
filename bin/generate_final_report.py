#!/usr/bin/env python3
"""
HTML Report Generator for Cell Type Classification Pipeline

This script reads JSON outputs from various pipeline stages and generates
a comprehensive HTML report using Jinja2 templates.

Usage:
    python generate_report.py --output-dir /path/to/pipeline/output --report-dir /path/to/report/output
"""

import os
import json
import glob
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import base64
import mimetypes
import pandas as pd
from typing import Any, Dict, Union

from jinja2 import Environment, FileSystemLoader, select_autoescape

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """
    Convert an image file to a base64 data URI.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 data URI string or None if file doesn't exist/error
    """
    try:
        image_path = Path(image_path)
        if not image_path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return None
        
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if not mime_type or not mime_type.startswith('image/'):
            # Default to PNG if we can't determine the type
            mime_type = 'image/png'
        
        # Read and encode the image
        with open(image_path, 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Create data URI
        data_uri = f"data:{mime_type};base64,{encoded_image}"
        logger.info(f"Successfully embedded image: {image_path}")
        return data_uri
        
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return None

def replace_png_with_base64(data: Union[Dict, Any], base_path: str = "") -> Union[Dict, Any]:
    """
    Recursively traverse a nested dictionary and replace PNG filenames with base64 embeddings.
    
    Args:
        data: The nested dictionary or value to process
        base_path: Base directory path for resolving relative filenames
    
    Returns:
        The processed dictionary with PNG filenames replaced by base64 data URIs
    """
    if isinstance(data, dict):
        return {key: replace_png_with_base64(value, base_path) for key, value in data.items()}
    elif isinstance(data, list):
        return [replace_png_with_base64(item, base_path) for item in data]
    elif isinstance(data, str) and data.lower().endswith('.png'):
        try:
            # Resolve full path
            file_path = os.path.join(base_path, data) if base_path else data
            
            # Read and encode the PNG file
            with open(file_path, 'rb') as img_file:
                img_data = img_file.read()
                base64_str = base64.b64encode(img_data).decode('utf-8')
                return f"data:image/png;base64,{base64_str}"
        except (FileNotFoundError, IOError):
            # Return original filename if file can't be read
            return data
    else:
        return data

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


def read_normalization_data(pipeline_output_dir: Path) -> Dict[str, Any]:
    """Read normalization results from norm directory."""
    norm_dir = pipeline_output_dir / "norm"
    norm_template = {
        'normalization_results': []
    }
    
    if not norm_dir.exists():
        logger.error(f"Normalization directory not found: {norm_dir}")
    
    # Look for normalization result files
    norm_patterns = ['boxcox_results_*.json', 'log_results_*.json', 
                    'quantile_results_*.json', 'minmax_results_*.json']
    
    norm_files = []
    all_methods = set()
    all_batches = set()
    for pattern in norm_patterns:
        norm_files.extend(glob.glob(str(norm_dir / pattern)))
    
    if not norm_files:
        logger.error("No normalization result files found")
    
    # Process each normalization file
    for file_path in norm_files:
        try:
            with open(file_path, 'r') as f:
                norm_result = json.load(f)
            
            method = norm_result.get('transformation_type')
            batch_name = norm_result.get('batch_name')
            all_methods.add(method)
            all_batches.add(batch_name)

            batch_data = {
                'method': method,
                'batch_name': batch_name,
                'data': norm_result
            }
            norm_template['normalization_results'].append(batch_data)
            logger.info(f"Loaded normalization data from {file_path}")

        except Exception as e:
            logger.error(f"Error reading normalization file {file_path}: {e}")

   # Update summary statistics
    norm_template.update({
        'normalization_methods': list(all_methods),
        'batch_names': list(all_batches),
        'total_batches': len(all_batches),
        'primary_method': (list(all_methods)[0] if len(all_methods) ==1
                           else 'N/A')
    }) 
    
    return replace_png_with_base64(norm_template, "./norm")


def read_feature_selection_data(pipeline_output_dir: Path) -> Dict[str, Any]:
    """Read feature selection results from feature_selection directory."""
    
    fs_dir = pipeline_output_dir / "feature_selection"
    fs_data = {
        'cell_types_processed': [],
        'feature_selection_results': [],
        'total_cell_types': 0
    }
    
    if not fs_dir.exists():
        logger.error(f"Feature selection directory not found: {fs_dir}")
    
    # Look for feature selection result files
    fs_files = glob.glob(str(fs_dir / "feature_selection_*_results.json"))
    
    if not fs_files:
        logger.error("No feature selection result files found")
    
    for file_path in fs_files:
        try:
            with open(file_path, 'r') as f:
                fs_result = json.load(f)
            
            celltype = fs_result.get('celltype')
            
            fs_data['cell_types_processed'].append(celltype)
            fs_data['feature_selection_results'].append({
                'celltype': celltype,
                'data': fs_result  # Full JSON data with embedded PNGs
            })
            
            logger.info(f"Loaded feature selection data for {celltype}")
            
        except Exception as e:
            logger.error(f"Error reading feature selection file {file_path}: {e}")
    
    # Update metadata
    fs_data.update({
        'total_cell_types': len(fs_data['cell_types_processed'])
    })
    
    return replace_png_with_base64(fs_data, "./feature_selection")


def read_modeling_data(pipeline_output_dir: Path) -> Dict[str, Any]:
    """Read modeling results from modeling directory."""
    modeling_dir = pipeline_output_dir / "modeling"
    modeling_data = {
        'holdout_evaluations': [],
        'model_comparisons': [],
        'training_classes': 0,
        'holdout_accuracy': 0.0,
        'f1_score': 0.0,
        'class_imbalance': '',
        'best_class': '',
        'worst_class': ''

    }
    
    if not modeling_dir.exists():
        logger.error(f"Modeling directory not found: {modeling_dir}")
    
    # Read holdout evaluation files
    holdout_files = glob.glob(str(modeling_dir / "holdoutEval*.json"))

    for file_path in holdout_files:
        try:
            with open(file_path, 'r') as f:
                eval_result = json.load(f)
            
            model_name = file_path.split('_Model_')[1].split('_results')[0]
            
            modeling_data['holdout_evaluations'].append({
                'model_name': model_name,
                'data': eval_result
            })

            if model_name == "First":
                modeling_data.update({
                    'training_classes': eval_result.get("n_classes"),
                    'holdout_accuracy': eval_result.get("accuracy"),
                    'f1_score': eval_result.get("f1_score"),
                    'class_imbalance': eval_result.get("class_imbalance_detected"),
                    'best_class': eval_result["max_auc"]["class_name"],
                    'worst_class': eval_result["min_auc"]["class_name"]
                })
                
            
            logger.info(f"Loaded modeling evaluation from {file_path}")
            
        except Exception as e:
            logger.error(f"Error reading modeling file {file_path}: {e}")
    
    # Read model comparison files (xgbWinners, etc.)
    try:
        with open(str(modeling_dir / "xgbWinners_results.json"), 'r') as f:
            comparison_result = json.load(f)
        
        modeling_data['model_comparisons'].append(comparison_result)
        
        logger.info(f"Loaded model comparison from {file_path}")
        
    except Exception as e:
        logger.error(f"Error reading model comparison file {file_path}: {e}")


    def sort_models(model_eval):
        model_name = model_eval['model_name']
        if 'First' in model_name:
            return 0  # First priority
        elif 'Second' in model_name:
            return 1  # Second priority
        else:
            return 2  # Everything else last
    
    modeling_data['holdout_evaluations'].sort(key=sort_models)
    
    return replace_png_with_base64(modeling_data, "./modeling")


def collect_all_data(pipeline_output_dir: Path) -> Dict[str, Any]:
    """Collect all data from pipeline outputs."""
    logger.info("Collecting normalization data...")
    norm_data = read_normalization_data(pipeline_output_dir)
    
    logger.info("Collecting feature selection data...")
    fs_data = read_feature_selection_data(pipeline_output_dir)
    
    logger.info("Collecting modeling data...")
    modeling_data = read_modeling_data(pipeline_output_dir)
    
    return {
        'normalization': norm_data,
        'feature_selection': fs_data,
        'modeling': modeling_data
        }

def csv_to_dict(csv_file_path, max_rows=None):
    """Convert CSV to dictionary for Jinja2 templates."""
    try:
        data = {'headers': [], 'rows': []}
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data['headers'] = reader.fieldnames or []
            for i, row in enumerate(reader):
                if max_rows and i >= max_rows:
                    break
                data['rows'].append(row)
        return data
    except:
        return {'headers': [], 'rows': []}


def generate_report(all_data: Dict[str, Any], 
                   output_file: str, 
                   jinja_env, 
                   template_name: str = "base.html",
                   letterhead: Optional[str] = None):
    """
    Generate the HTML report with embedded images.
    
    Args:
        all_data: Dictionary containing all collected pipeline data
        output_file: Path to output HTML file
        jinja_env: Jinja2 environment
        template_name: Name of the Jinja2 template file
        header_logo_path: Optional path to header logo image
    """
    try:
        template = jinja_env.get_template(template_name)
        
        # Embed all images
        logger.info("Embedding images into report...")

        norm = all_data['normalization']
        
        template_data = {
            'total_cells': "123", #coming soon
            'total_batches': norm['total_batches'],
            'labeled_classes': "123", #coming soon
            'training_classes': all_data['modeling']["training_classes"],
            'normalization_method': norm["primary_method"],
            'holdout_accuracy': all_data['modeling']["holdout_accuracy"],
            'f1_score': all_data['modeling']["f1_score"],
            'normalization_data': norm,
            'feature_selection_data': all_data['feature_selection'],
            'modeling_data': all_data['modeling'],
            'letterhead': encode_image_to_base64(letterhead)

        }

        # Render the template
        html_content = template.render(**template_data)
        
        # Write to output file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Report generated successfully: {output_path}")
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise


def main():
    """Main function to parse arguments and generate report."""
    parser = argparse.ArgumentParser(
        description="Generate HTML report from pipeline outputs"
    )
    parser.add_argument(
        '--input-dir',
        default="./"
        help='Directory containing pipeline outputs (default: ./)'
    )
    parser.add_argument(
        '--report-dir',
        default='./',
        help='Directory to save the generated report (default: ./)'
    )
    parser.add_argument(
        '--template-dir',
        default='templates',
        help='Directory containing Jinja2 templates (default: templates)'
    )
    parser.add_argument(
        '--report-name',
        default='cell_classification_report.html',
        help='Name of the output report file (default: cell_classification_report.html)'
    )
    parser.add_argument(
        '--letterhead',
        help='Path to header logo image (will be embedded in report)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set up Jinja2 environment
    jinja_env = setup_jinja_environment(args.template_dir)
    
    # Collect all data
    all_data = collect_all_data(args.input_dir)
    
    # Generate report
    output_file = Path(args.report_dir) / args.report_name
    generate_report(
        all_data, 
        output_file, 
        jinja_env, 
        letterhead=args.letterhead,
    )
    
    print(f"Report generated: {output_file}")
    print("All images have been embedded - the report is completely standalone!")


if __name__ == "__main__":
    main()