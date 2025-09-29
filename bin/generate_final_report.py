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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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


def load_sample_jsons(json_directory):
    """
    Load all JSON files from a directory and organize by sample_name
    
    Args:
        json_directory (str): Path to directory containing JSON files
        
    Returns:
        dict: Dictionary with sample_name as keys and JSON data as values
    """
    samples = []
    
    # Find all JSON files in the directory
    json_pattern = os.path.join(json_directory, "*.json")
    json_files = glob.glob(json_pattern)
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as file:
                data = json.load(file)
                samples.append(data)
                
        except json.JSONDecodeError as e:
            print(f"Error reading {json_file}: {e}")
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    return samples

def load_metrics_json(json_file):
    """
    Load a single metrics JSON file
    
    Args:
        json_file (str): Path to the metrics JSON file
        
    Returns:
        dict: Dictionary containing the metrics data
    """
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
            print(f"Loaded metrics: {json_file}")
            return data
    except json.JSONDecodeError as e:
        print(f"Error reading {json_file}: {e}")
        return {}
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        return {}
    

def read_df(file_path):
    """Read cell count table from CSV file."""
    try:
        df = pd.read_csv(file_path)
        return {
            'headers': df.columns.tolist(),
            'rows': df.to_dict('records')
        }
    except Exception as e:
        logger.warning(f"Could not df: {e}")
        return {'headers': [], 'rows': []}
    
def read_html_chunk(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    return html_content


def collect_html_content(input_dir):
    input_dir = Path(input_dir)
    all_jsons = {
        'normalization_html_content': read_html_chunk(input_dir / "normalization_report.html"),
        'feature_selection_html_content': read_html_chunk(input_dir / "feature_selection_report.html"),
        'model_html_content': read_html_chunk(input_dir / "model_report.html"),
        'cell_count_table': read_df(input_dir / "cell_count_table.csv")
        }

    return all_jsons

def collect_metric_data(input_dir):
    input_dir = Path(input_dir)
    all_jsons = {
        'input_metrics': load_metrics_json(input_dir / "training_split_report.json"),
        'model_metrics': load_metrics_json(input_dir / "model_summary.json")
        }

    return all_jsons

def generate_prediction_content(pred_dir):
    
    df = read_prediction_files(pred_dir)

    results = {
        'prediction_metrics': collect_pred_results(df, pred_dir)
    }

    return results

def read_prediction_files(input_dir, file_pattern="*qPRED.tsv"):
    """Read all prediction files and combine into a single dataframe"""
    prediction_files = glob.glob(os.path.join(input_dir, file_pattern))
    
    if not prediction_files:
        raise ValueError(f"No files found matching pattern {file_pattern} in {input_dir}")
    
    all_data = []
    
    for file_path in prediction_files:
        try:
            df = pd.read_csv(file_path, sep='\t')
            
            # Extract sample name from filename
            sample_name = Path(file_path).stem.replace('.ome.tiff_qPRED', '')
            df['Sample'] = sample_name
            
            all_data.append(df)
            
            print(f"Loaded {len(df):,} cells from {sample_name}")
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No valid prediction files could be read")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df["Sample"] = combined_df["Image"].str.replace('.ome.tif', '')

    return combined_df

def collect_pred_results(df, input_dir):

    total_low_density_cells = 0
    # Add low density cells if available
    if 'low_bin_density' in df.columns:
        total_low_density_cells = df['low_bin_density'].sum()

    abundance_plot_name = "prediction_abundance_plot.png"
    # Create abundance plot
    print("Creating abundance visualization...")
    create_abundance_plot(df, abundance_plot_name)

    cell_type_counts = df['CellTypePrediction'].value_counts()
    cell_type_percentages = df['CellTypePrediction'].value_counts(normalize=True) * 100
    cell_type_percentages = cell_type_percentages.round(1)

    results = {
        'total_predicted_cells': len(df),
        'most_common_prediction': cell_type_counts.index[0] if len(cell_type_counts) > 0 else None,
        'most_common_percentage': cell_type_percentages.iloc[0] if len(cell_type_percentages) > 0 else None,
        'most_rare_prediction': cell_type_counts.index[-1] if len(cell_type_counts) > 0 else None,
        'most_rare_percentage': cell_type_percentages.iloc[-1] if len(cell_type_percentages) > 2 else None,
        'abundance_plot': abundance_plot_name,
        'total_low_density_cells': total_low_density_cells,
        'samples': load_sample_jsons(input_dir)
    }
    
    return results

def create_abundance_plot(df, output_file):
    """Create stacked bar plot with simplified dynamic sizing"""
    # Calculate dataset characteristics for sizing
    n_samples = df['Sample'].nunique()
    n_cell_types = df['CellTypePrediction'].nunique()
    
    # Enhanced dynamic sizing for large datasets
    plot_width = max(12, min(30, 10 + n_samples * 0.4))
    plot_height = max(8, min(12, 7 + n_cell_types * 0.15))
    
    # Calculate proportions and pivot
    proportions_list = []
    for sample in sorted(df['Sample'].unique()):
        sample_df = df[df['Sample'] == sample]
        proportions = sample_df['CellTypePrediction'].value_counts(normalize=True) * 100
        for cell_type, percentage in proportions.items():
            proportions_list.append({
                'Sample': sample,
                'CellType': cell_type,
                'Percentage': percentage
            })
    
    proportions_df = pd.DataFrame(proportions_list)
    pivot_df = proportions_df.pivot(index='Sample', columns='CellType', values='Percentage').fillna(0)
    print(pivot_df)
    # Get overall cell type order (most abundant first)
    overall_abundance = df['CellTypePrediction'].value_counts()
    cell_type_order = overall_abundance.index.tolist()
    
    # Reorder columns to match abundance order
    pivot_df = pivot_df.reindex(columns=cell_type_order, fill_value=0)
    
    # Create plot with standardized styling
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(plot_width, plot_height), dpi=300)
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.linewidth': 1
    })
    
    # Generate colors
    colors = sns.color_palette("Set2", n_cell_types)
    
    # Create stacked bars in order
    bottom = np.zeros(len(pivot_df))
    for i, cell_type in enumerate(pivot_df.columns):
        ax.bar(
            range(len(pivot_df)), 
            pivot_df[cell_type], 
            bottom=bottom,
            label=cell_type,
            color=colors[i],
            alpha=0.85,
            edgecolor='white',
            linewidth=0.8
        )
        bottom += pivot_df[cell_type]
        
    # Enhanced title with better positioning
    fig.suptitle('Predicted Cell Type Composition by Sample', fontsize=15, fontweight='bold', y=1.02)
    ax.set_xlabel('Sample', fontsize=11, fontweight='bold')
    ax.set_ylabel('Percentage of Cells (%)', fontsize=11, fontweight='bold')
    
    # X-axis labels - dynamic rotation based on sample count
    ax.set_xticks(range(len(pivot_df)))
    if n_samples <= 10:
        rotation = 30
        fontsize = 10
    elif n_samples <= 25:
        rotation = 45
        fontsize = 9
    else:
        rotation = 70
        fontsize = 8
    ax.set_xticklabels(pivot_df.index, rotation=rotation, ha='right', fontsize=fontsize)
    
    # Y-axis
    ax.set_ylim(0, 100)
    
    # Add sample counts above bars with improved rotation and positioning
    sample_counts = df.groupby('Sample').size()
    count_rotation = 0 if n_samples <= 15 else 30 if n_samples <= 30 else 45
    for i, sample in enumerate(pivot_df.index):
        ax.text(i, 104, f'n={sample_counts[sample]:,}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                rotation=count_rotation)
    
    # Legend (matching bar order - bottom to top)
    handles, labels = ax.get_legend_handles_labels()
    # Reverse to match visual stacking order (bottom to top)
    legend = ax.legend(
        reversed(handles), reversed(labels),
        bbox_to_anchor=(1.05, 1), 
        loc='upper left',
        frameon=True,
        title='Cell Type',
        title_fontsize=11,
        fontsize=10
    )
    
    # Clean styling
    ax.grid(True, alpha=0.3)
    sns.despine(top=True, right=True)
    ax.set_facecolor('#fafafa')
    
    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')

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

def generate_report(all_data, output_file, jinja_env, letterhead, pipeline_version, template_name = "base.html"):
    try:
        template = jinja_env.get_template(template_name)
        
        template_data = {
            'pipeline_version': pipeline_version,
            'generation_date': datetime.now().strftime("%B %d, %Y"),
            'letterhead': encode_image_to_base64(letterhead) if letterhead else None,
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
    parser = argparse.ArgumentParser(description="Generate HTML report from pipeline outputs")
    parser.add_argument('--input-dir', default="./", help='Directory containing pipeline outputs (default: ./)')
    parser.add_argument('--pred-dir', default='./', help='Directory to save the generated report (default: ./)')
    parser.add_argument('--template-dir', default='templates', help='Directory containing Jinja2 templates (default: templates)')
    parser.add_argument('--report-name', default='cell_classification_report.html', help='Name of the output report file (default: cell_classification_report.html)')
    parser.add_argument('--letterhead', help='Path to header logo image (will be embedded in report)')
    parser.add_argument('--version', help='Pipeline version to be displayed in final report', default="N/A")

    args = parser.parse_args()
    
    # Set up Jinja2 environment
    jinja_env = setup_jinja_environment(args.template_dir)
    
    # Collect all data
    all_data = collect_html_content(args.input_dir)

    all_data.update(collect_metric_data(args.input_dir))

    all_data.update(generate_prediction_content(args.pred_dir))

    generate_report(
        all_data=all_data, 
        output_file=args.report_name, 
        jinja_env=jinja_env, 
        letterhead=args.letterhead,
        pipeline_version=args.version
    )

    with open('output.json', 'w') as f:
        json.dump(all_data, f, indent=2, default=str)

if __name__ == "__main__":
    main()