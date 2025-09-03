#!/usr/bin/env python3
"""
Generate individual ROI detail pages with embedded plots and comprehensive metrics.

This script creates standalone HTML pages for each ROI containing:
- Complete ROI metrics and statistics
- Embedded barplot and spatial plot images
- Professional styling matching the main report
"""

import argparse
import os
import json
import glob
import base64
from pathlib import Path
from typing import Dict, List, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def encode_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 encoded string for embedding."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            # Determine MIME type based on file extension
            ext = Path(image_path).suffix.lower()
            if ext in ['.png']:
                mime_type = 'image/png'
            elif ext in ['.jpg', '.jpeg']:
                mime_type = 'image/jpeg'
            elif ext in ['.svg']:
                mime_type = 'image/svg+xml'
            else:
                mime_type = 'image/png'  # Default
            
            return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return ""

def read_html_file(html_path: str) -> str:
    """Read HTML file content."""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading HTML file {html_path}: {e}")
        return ""

def load_roi_data(abundance_dir: str, classified_dir: str) -> Dict[str, Dict]:
    """Load ROI data from abundance and classified results."""
    roi_data = {}
    
    # Load per-sample statistics from abundance results
    abundance_files = glob.glob(os.path.join(abundance_dir, "*_summary_stats.json"))
    
    for abundance_file in abundance_files:
        try:
            with open(abundance_file, 'r') as f:
                data = json.load(f)
                sample_name = data.get('sample_name')
                if sample_name:
                    roi_data[sample_name] = {
                        'stats': data,
                        'barplot_path': None,
                        'spatial_path': None
                    }
        except Exception as e:
            logger.error(f"Error reading abundance file {abundance_file}: {e}")
    
    # Find corresponding barplot and spatial plot files
    barplot_files = glob.glob(os.path.join(classified_dir, "*_celltype_barplot.png"))
    spatial_files = glob.glob(os.path.join(classified_dir, "*_spatial_plot.html"))
    
    for barplot_file in barplot_files:
        # Extract sample name from filename
        filename = Path(barplot_file).stem
        sample_name = filename.replace('_celltype_barplot', '')
        
        if sample_name in roi_data:
            roi_data[sample_name]['barplot_path'] = barplot_file
    
    for spatial_file in spatial_files:
        # Extract sample name from filename
        filename = Path(spatial_file).stem
        sample_name = filename.replace('_spatial_plot', '')
        
        if sample_name in roi_data:
            roi_data[sample_name]['spatial_path'] = spatial_file
    
    logger.info(f"Loaded data for {len(roi_data)} ROIs")
    return roi_data

def generate_roi_html_template() -> str:
    """Generate the HTML template for individual ROI pages."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ roi_name }} - ROI Detail Report</title>
    <style>
        /* Reset and base styles */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }

        /* Header */
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px 12px 0 0;
            text-align: center;
            margin-bottom: 0;
        }

        .header h1 {
            font-size: 2.5rem;
            font-weight: 300;
            margin-bottom: 0.5rem;
        }

        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        /* Navigation */
        .nav-back {
            background: white;
            padding: 1rem 2rem;
            border-radius: 0 0 12px 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }

        .nav-back a {
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
            font-size: 1rem;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.3s ease;
        }

        .nav-back a:hover {
            color: #5a6fd8;
            transform: translateX(-2px);
        }

        /* Main content */
        .content-container {
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
            overflow: hidden;
            margin-bottom: 2rem;
        }

        .section {
            padding: 2rem;
            border-bottom: 1px solid #e9ecef;
        }

        .section:last-child {
            border-bottom: none;
        }

        .section-title {
            font-size: 1.8rem;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #667eea;
        }

        /* Metrics grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin: 1.5rem 0;
        }

        .metric-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 0.5rem;
        }

        .metric-label {
            font-size: 0.9rem;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }

        /* Cell type distribution */
        .celltype-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }

        .celltype-card {
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            transition: all 0.3s ease;
        }

        .celltype-card:hover {
            border-color: #667eea;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
        }

        .celltype-name {
            font-weight: 600;
            color: #495057;
            margin-bottom: 0.5rem;
        }

        .celltype-count {
            font-size: 1.2rem;
            font-weight: 700;
            color: #28a745;
            margin-bottom: 0.25rem;
        }

        .celltype-percentage {
            font-size: 0.9rem;
            color: #6c757d;
        }

        /* Plot containers */
        .plot-container {
            text-align: center;
            margin: 2rem 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }

        .plot-container img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }

        .plot-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #495057;
            padding: 1rem;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
        }

        .spatial-container {
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }

        .spatial-container iframe {
            width: 100%;
            height: 600px;
            border: none;
        }

        /* Alert styles */
        .alert {
            padding: 1rem 1.5rem;
            margin: 1rem 0;
            border-radius: 8px;
            border-left: 4px solid;
        }

        .alert-info {
            background-color: #d1ecf1;
            border-color: #17a2b8;
            color: #0c5460;
        }

        .alert-warning {
            background-color: #fff3cd;
            border-color: #ffc107;
            color: #856404;
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem;
            color: #6c757d;
            font-size: 0.9rem;
        }

        /* Responsive design */
        @media (max-width: 768px) {
            body {
                padding: 1rem;
            }

            .header h1 {
                font-size: 2rem;
            }

            .metrics-grid {
                grid-template-columns: 1fr;
            }

            .section {
                padding: 1.5rem;
            }
        }
    </style>
</head>
<body>
    <!-- Header -->
    <div class="header">
        <h1>{{ roi_name }}</h1>
        <p>Detailed Cell Type Analysis Report</p>
    </div>

    <!-- Navigation -->
    <div class="nav-back">
        <a href="javascript:history.back()">← Back to ROI Summary</a>
    </div>

    <!-- Main Content -->
    <div class="content-container">
        <!-- ROI Overview -->
        <div class="section">
            <h2 class="section-title">ROI Overview</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{{ number_format(total_cells) }}</div>
                    <div class="metric-label">Total Cells</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ unique_classes }}</div>
                    <div class="metric-label">Unique Cell Types</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ most_common_class }}</div>
                    <div class="metric-label">Most Common Type</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ least_common_class }}</div>
                    <div class="metric-label">Least Common Type</div>
                </div>
            </div>

            <div class="alert alert-info">
                <strong>Key Statistics:</strong>
                Most abundant cell type is <strong>{{ most_common_class }}</strong> at {{ most_common_percentage }}%. 
                {% if second_common_class != 'N/A' %}
                Second most common is <strong>{{ second_common_class }}</strong> at {{ second_common_percentage }}%.
                {% endif %}
                {% if least_common_class != 'N/A' %}
                Least common type is <strong>{{ least_common_class }}</strong> at {{ least_common_percentage }}%.
                {% endif %}
            </div>
        </div>

        <!-- Cell Type Distribution -->
        <div class="section">
            <h2 class="section-title">Cell Type Distribution</h2>
            <div class="celltype-grid">
                {% for cell_type, count in cell_type_distribution.items() %}
                <div class="celltype-card">
                    <div class="celltype-name">{{ cell_type }}</div>
                    <div class="celltype-count">{{ number_format(count) }}</div>
                    <div class="celltype-percentage">{{ "%.1f" | format(percentage_distribution[cell_type]) }}%</div>
                </div>
                {% endfor %}
            </div>
        </div>

        <!-- Plots Section -->
        {% if barplot_image %}
        <div class="section">
            <h2 class="section-title">Cell Type Abundance</h2>
            <div class="plot-container">
                <div class="plot-title">Cell Type Distribution Bar Plot</div>
                <img src="{{ barplot_image }}" alt="Cell Type Distribution" />
            </div>
        </div>
        {% endif %}

        {% if spatial_content %}
        <div class="section">
            <h2 class="section-title">Spatial Distribution</h2>
            <div class="spatial-container">
                <div class="plot-title">Interactive Spatial Visualization</div>
                {{ spatial_content | safe }}
            </div>
        </div>
        {% elif spatial_image %}
        <div class="section">
            <h2 class="section-title">Spatial Distribution</h2>
            <div class="plot-container">
                <div class="plot-title">Spatial Distribution Plot</div>
                <img src="{{ spatial_image }}" alt="Spatial Distribution" />
            </div>
        </div>
        {% endif %}

        {% if not barplot_image and not spatial_content and not spatial_image %}
        <div class="section">
            <div class="alert alert-warning">
                <strong>Note:</strong> No visualization plots are available for this ROI.
            </div>
        </div>
        {% endif %}
    </div>

    <!-- Footer -->
    <div class="footer">
        <p>Generated by ClassyFlow Pipeline | ROI Detail Report</p>
    </div>
</body>
</html>
"""

def generate_roi_page(roi_name: str, roi_info: Dict, output_dir: str):
    """Generate individual ROI detail page."""
    from jinja2 import Template
    
    # Get the data
    stats = roi_info.get('stats', {})
    barplot_path = roi_info.get('barplot_path')
    spatial_path = roi_info.get('spatial_path')
    
    # Prepare template data
    template_data = {
        'roi_name': roi_name,
        'total_cells': stats.get('total_cells', 0),
        'unique_classes': stats.get('unique_classes', 0),
        'most_common_class': stats.get('most_common_class', 'N/A'),
        'most_common_percentage': f"{stats.get('most_common_percentage', 0):.1f}",
        'second_common_class': stats.get('second_common_class', 'N/A'),
        'second_common_percentage': f"{stats.get('second_common_percentage', 0):.1f}",
        'least_common_class': stats.get('least_common_class', 'N/A'),
        'least_common_percentage': f"{stats.get('least_common_percentage', 0):.1f}",
        'cell_type_distribution': stats.get('cell_type_distribution', {}),
        'percentage_distribution': stats.get('percentage_distribution', {}),
        'barplot_image': None,
        'spatial_content': None,
        'spatial_image': None
    }
    
    # Process barplot
    if barplot_path and os.path.exists(barplot_path):
        template_data['barplot_image'] = encode_image_to_base64(barplot_path)
    
    # Process spatial plot
    if spatial_path and os.path.exists(spatial_path):
        spatial_content = read_html_file(spatial_path)
        if spatial_content:
            # Extract the body content from the HTML (remove html, head tags)
            import re
            body_match = re.search(r'<body[^>]*>(.*?)</body>', spatial_content, re.DOTALL | re.IGNORECASE)
            if body_match:
                template_data['spatial_content'] = body_match.group(1)
            else:
                template_data['spatial_content'] = spatial_content
    
    # Create Jinja2 template with custom filters
    template = Template(generate_roi_html_template())
    template.globals['number_format'] = lambda x: f"{x:,}" if isinstance(x, (int, float)) else str(x)
    
    # Render the template
    html_content = template.render(**template_data)
    
    # Save the file
    output_file = os.path.join(output_dir, f"{roi_name}_detail.html")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Generated ROI detail page: {output_file}")

def main():
    """Main function to generate all ROI detail pages."""
    parser = argparse.ArgumentParser(description="Generate individual ROI detail pages")
    parser.add_argument('--abundance_dir', required=True, help='Directory containing abundance summary files')
    parser.add_argument('--classified_dir', required=True, help='Directory containing classified results (plots)')
    parser.add_argument('--output_dir', required=True, help='Output directory for ROI detail pages')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load ROI data
    roi_data = load_roi_data(args.abundance_dir, args.classified_dir)
    
    if not roi_data:
        logger.warning("No ROI data found. No pages will be generated.")
        return
    
    # Generate pages for each ROI and collect filename mapping
    roi_filename_mapping = {}
    for roi_name, roi_info in roi_data.items():
        try:
            generate_roi_page(roi_name, roi_info, args.output_dir)
            # Store the mapping of ROI name to generated filename
            roi_filename_mapping[roi_name] = f"{roi_name}_detail.html"
        except Exception as e:
            logger.error(f"Error generating page for {roi_name}: {e}")
    
    # Save the filename mapping as JSON for the template system
    mapping_file = os.path.join(args.output_dir, "roi_filename_mapping.json")
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(roi_filename_mapping, f, indent=2)
    
    logger.info(f"Generated {len(roi_data)} ROI detail pages in {args.output_dir}")
    logger.info(f"Saved ROI filename mapping to {mapping_file}")

if __name__ == "__main__":
    main()