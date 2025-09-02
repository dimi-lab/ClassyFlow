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


def read_config_file(config_path: str) -> Dict[str, Any]:
    """Read pipeline configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config_data
    except Exception as e:
        logger.error(f"Error reading config file {config_path}: {e}")
        return {}


def read_normalization_data(pipeline_output_dir: Path) -> Dict[str, Any]:
    """Read normalization results from norm directory."""
    norm_dir = Path(pipeline_output_dir, "norm")
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


def read_general_data(pipeline_output_dir: Path) -> Dict[str, Any]:
    general_dir = Path(pipeline_output_dir, "general")
    results = {"per_slide": []}

    if not general_dir.exists():
        logger.error(f"General Metrics directory not found: {general_dir}")
        return results

    # Read main abundance metrics file
    abundance_file = general_dir / "abundance_metrics.json"
    if abundance_file.exists():
        try:
            with open(abundance_file, 'r') as f:
                abundance_data = json.load(f)
            results.update(abundance_data)
            logger.info(f"Loaded abundance metrics from {abundance_file}")
        except Exception as e:
            logger.error(f"Error reading abundance metrics: {e}")
    
    # Read other general JSON files
    json_files = glob.glob(str(general_dir / "*.json"))
    for file_path in json_files:
        if Path(file_path).name == "abundance_metrics.json":
            continue  # Already processed above
            
        try:
            with open(file_path, 'r') as f:
                json_results = json.load(f)
            
            conflicting_keys = set(results.keys()) & set(json_results.keys())
            if conflicting_keys:
                logger.warning(f"Overwriting existing keys from {file_path}: {conflicting_keys}")

            results.update(json_results)
            logger.info(f"Loaded metrics from {file_path}")
            
        except Exception as e:
            logger.error(f"Error reading metrics file {file_path}: {e}")

    # Read per-slide classification results
    per_slide_dir = general_dir / "per_slide"
    if per_slide_dir.exists():
        json_files = glob.glob(str(per_slide_dir / "*_classified.json"))
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    json_results = json.load(f)
                
                results["per_slide"].append({
                    'sample': json_results["sample_name"],
                    'data': json_results
                })
                
                logger.info(f"Loaded per-slide metrics from {file_path}")
                
            except Exception as e:
                logger.error(f"Error reading per-slide metrics file {file_path}: {e}")
    
    # If we have per_sample_statistics in abundance data, use that for per_slide if per_slide is empty
    if not results["per_slide"] and "per_sample_statistics" in results:
        for sample_stat in results["per_sample_statistics"]:
            results["per_slide"].append({
                'sample': sample_stat["sample_name"],
                'data': sample_stat
            })
        logger.info(f"Used per_sample_statistics for per_slide data ({len(results['per_slide'])} samples)")
    
    print(f"Loaded general data with {len(results['per_slide'])} per-slide entries")
    return replace_png_with_base64(results, str(general_dir))

def collect_all_data(pipeline_output_dir: Path) -> Dict[str, Any]:
    """Collect all data from pipeline outputs."""
    logger.info("Collecting general metrics...")
    general_data = read_general_data(pipeline_output_dir)
        
    all_jsons = {
        'general': general_data,
        'normalization_report': "normalization_report.html",
        'feature_selection_data': "feature_selection_report.html",
        'modeling_data': "model_report.html"
        }

    with open(f'all_jsons.json', 'w') as f:
            json.dump(all_jsons, f, indent=2)

    return all_jsons

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
                   letterhead: Optional[str] = None,
                   pipeline_version: Optional[str] = None):
    """
    Generate the HTML report with embedded images.
    
    Args:
        all_data: Dictionary containing all collected pipeline data
        output_file: Path to output HTML file
        jinja_env: Jinja2 environment
        template_name: Name of the Jinja2 template file
        letterhead: Optional path to letterhead image
        pipeline_version: Pipeline version string
    """
    try:
        template = jinja_env.get_template(template_name)
        
        # Embed all images
        logger.info("Embedding images into report...")

        with open('normalization_report.html', 'r', encoding='utf-8') as f:
            normalization_html = f.read()

        with open('feature_selection_report.html', 'r', encoding='utf-8') as f:
            feature_selection_html = f.read()

        with open('model_report.html', 'r', encoding='utf-8') as f:
            model_html = f.read()

        template_data = {
            'pipeline_version': pipeline_version,
            'generation_date': datetime.now().strftime("%B %d, %Y"),
            'total_cells': all_data['general']['total_cells'], 
            'total_labels': all_data['general']["total_annotated_cells"],
            'letterhead': encode_image_to_base64(letterhead) if letterhead else None,
            'normalization_html_content': normalization_html,
            'feature_selection_html_content': feature_selection_html,
            'model_html_content': model_html
        }

        template_data.update(all_data)

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
        default="./",
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
        '--version',
        help='Pipeline version to be displayed in final report',
        default="N/A"
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
    
    # Generate main report
    output_file = Path(args.report_dir) / args.report_name
    generate_report(
        all_data=all_data, 
        output_file=output_file, 
        jinja_env=jinja_env, 
        letterhead=args.letterhead,
        pipeline_version=args.version
    )
    
    print(f"Main report generated: {output_file}")

if __name__ == "__main__":
    main()