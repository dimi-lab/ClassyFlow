# ClassyFlow User Guide

## Table of Contents
- [Installation & Setup](#installation--setup)
- [Input Data Requirements](#input-data-requirements)
- [Basic Usage](#basic-usage)
- [Understanding Outputs](#understanding-outputs)
- [Troubleshooting](#troubleshooting)

## Installation & Setup

### Requirements

- [Nextflow](https://www.nextflow.io/) ≥ 23.04.0
- [Python](https://www.python.org/) ≥ 3.10
- [Docker](https://www.docker.com/) or [Conda](https://docs.conda.io/) (recommended)

### Python Dependencies
```
fpdf 1.7.2
numpy 1.23.5
matplotlib 3.8.0
dataframe-image 0.2.3
pandas 2.2.0
seaborn 0.13.1
xgboost 1.6.2
scipy 1.12.0
scikit-learn 1.4.0
```

### Installation Options

#### Option 1: Local Installation
```bash
# Clone the repository
git clone https://github.com/dimi-lab/ClassyFlow.git
cd ClassyFlow

# Create Conda environment (optional but recommended)
conda create -n classyflow python=3.10
conda activate classyflow

# Install Python packages
pip install -r requirements.txt

# Test installation
nextflow run main.nf --help
```

#### Option 2: Docker/GCP
```bash
### Add documentation here
```

## Input Data Requirements

### File Organization
ClassyFlow expects measurement tables (quantification files) generated using QuPath 0.5+ from segmented single cell MxIF images. The pipeline supports multiple batches as input, organized as follows:

```
input_directory/
├── batch1/
│   ├── sample1_QUANT.tsv
│   ├── sample2_QUANT.tsv
│   └── ...
└── batch2/
    ├── sample1_QUANT.tsv
    ├── sample2_QUANT.tsv
    └── ...
```

### Quantification File Format

#### Required Columns

| Column Name | Type | Description | Example |
|-------------|------|-------------|---------|
| `Classification` | String | Cell type annotation (for training) | `T cell`, `B cell`, `Macrophage` |
| `Centroid X µm` | Float | X coordinate of cell centroid | `1245.67` |
| `Centroid Y µm` | Float | Y coordinate of cell centroid | `2890.34` |
| `[Marker Intensity]` | Float | Marker intensity values exported from QuPath either as *CellObject* or *DetectionObject* | `CD3: Cell: Mean` |

#### QuPath Object Types

ClassyFlow supports two QuPath object types that determine how marker intensities are measured and named:

##### CellObject
- **Components**: 4 measurement regions (Cell, Cytoplasm, Membrane, Nucleus)
- **Column Format**: `[Marker]: [Component]: [Statistic]`
- **Example**: `DAPI: Cell: Mean`, `CD3: Nucleus: Median`, `Ki67: Cytoplasm: Std Dev.`

##### DetectionObject  
- **Components**: 1 measurement region (either nucleus or whole cell)
- **Column Format**: `[Marker]: [Statistic]`
- **Example**: `DAPI: Mean`, `CD3: Median`, `Ki67: Std Dev.`


*Note: Don't forget to set the `qupath_object_type` with the appropriate value in your config file*

### Data Requirements
- **Training annotations**: Minimum 30-50 annotated cells per cell type
- **File format**: Tab-delimited (.tsv) files
- **Marker naming**: Follow QuPath convention (e.g., `CD3: Cell: Mean`)

### Supported Input Formats
- **QuPath exports**: Measurement tables from QuPath 0.5+
- **CellObject**: Includes Cell, Cytoplasm, Membrane, Nucleus measurements
- **DetectionObject**: Single whole cell or nucleus measurements

## Pipeline Usage - Under construction

### Quick Start
```bash
# Run with example configuration
nextflow run main.nf -c nextflow_ovtma.config

# Run with custom input directory
nextflow run main.nf --input_dirs "/path/to/your/data"

# Run with specific normalization
nextflow run main.nf --override_normalization "quantile"
```

### Execution Profiles

#### Local Execution (Default)
```bash
nextflow run main.nf
```

#### SLURM Cluster
```bash
nextflow run main.nf -profile slurm
```

#### Google Cloud Platform
```bash
nextflow run main.nf -profile gcp
```

## Understanding Outputs

### Output Directory Structure
```
classyflow_output/
├── normalization_reports/      # Data transformation QC reports (PDF)
├── celltype_reports/           # Feature selection analysis (PDF)
├── model_reports/              # Training & validation reports (PDF)
├── models/                     # Trained classifiers & encoders
├── celltypes/                  # Final cell type predictions (TSV)
└── clusters/                   # Optional clustering analysis
```

### Key Output Files

#### Quality Control Reports
- **normalization_reports/**: PDF reports comparing normalization methods
- **celltype_reports/**: Feature selection analysis per cell type
- **model_reports/**: Model training and validation summaries

#### Trained Models
- **models/XGBoost_Model_First.pkl**: Best performing model
- **models/XGBoost_Model_Second.pkl**: Second best model
- **models/classes.npy**: Label encoder for predictions

#### Final Results
- **celltypes/[ImageName]_PRED.tsv**: Per-image cell type predictions

### Interpreting Results

#### Prediction Files
Each prediction file contains:
```tsv
Image               Centroid X µm   Centroid Y µm   CellTypePrediction
TMA_Core_001.tif   1245.67         2890.34         T cell
TMA_Core_001.tif   1356.78         2901.45         B cell
```

#### Model Performance
- **Typical Accuracy**: 88-95% on holdout data
- **Model Reports**: Show feature importance and validation metrics
- **Holdout Evaluation**: Unbiased performance assessment

### Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/dimi-lab/ClassyFlow/issues)
- **Email Support**: your.email@mayo.edu
- **Parameter Questions**: See [Parameter Reference](parameter-reference.md)
- **Technical Details**: See [Pipeline Details](pipeline-details.md)