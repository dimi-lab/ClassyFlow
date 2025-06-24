# ClassyFlow User Guide

## Table of Contents
- [Input Data Requirements](#input-data-requirements)
- [Pipeline Usage](#pipeline-usage)
- [Understanding Outputs](#understanding-outputs)

## Installation
Please refer to Installation Instructions for requirements and dependencies

## Input Data Requirements
ClassyFlow expects measurement tables (quantification files) generated using QuPath 0.5+ from segmented single cell MxIF images. Each batch of samples should be placed in a separate directory. Additionally, a subset of cells must be annotated with cell type labels to be used for training and validation. It is recommended to have at least 30-50 annotations per cell type across multiple ROIs/samples.

### Quantification File Format

#### Required Columns

| Column Name | Type | Description | Example |
|-------------|------|-------------|---------|
| `Classification` | String | Cell type annotation (for training) | `T cell`, `B cell`, `Macrophage` |
| `Centroid X µm` | Float | X coordinate of cell centroid | `1245.67` |
| `Centroid Y µm` | Float | Y coordinate of cell centroid | `2890.34` |
| `[Marker Intensity]` | Float | Marker intensity values exported from QuPath either as *CellObject* or *DetectionObject* | `CD3: Cell: Mean` |

### QuPath Object Types

ClassyFlow supports two QuPath object types that determine how marker intensities are measured and named:

#### CellObject
- **Components**: 4 measurement regions (Cell, Cytoplasm, Membrane, Nucleus)
- **Column Format**: `[Marker]: [Component]: [Statistic]`
- **Example**: `DAPI: Cell: Mean`, `CD3: Nucleus: Median`, `Ki67: Cytoplasm: Std Dev.`

#### DetectionObject  
- **Components**: 1 measurement region (either nucleus or whole cell)
- **Column Format**: `[Marker]: [Statistic]`
- **Example**: `DAPI: Mean`, `CD3: Median`, `Ki67: Std Dev.`

*Note: Don't forget to set the `qupath_object_type` with the appropriate value in your config file*


## Pipeline Usage

### Running modes
```bash
# Run with example data
nextflow run main.nf

# Any configuration item can be overridden via the CLI. 
#  For example, running with custom input directory
nextflow run main.nf --input_dirs ["/path/to/your/data"]

# If re-running the pipeline, you may use the resume flag 
#  to take advantage of nextflow's caching
nextflow run main.nf -resume
```

### Execution Profiles
ClassyFlow may be run locally, on a slurm cluster, or on Google Cloud Platform using a docker container. Configurations for these profiles can be found in the `conf/` directory.

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
├── normalization_reports/      # Data transformation QC reports (HTML)
├── celltype_reports/           # Feature selection analysis (HTML)
├── model_reports/              # Training & validation reports (HTML)
├── models/                     # Trained classifiers & encoders
├── celltypes/                  # Final cell type predictions (TSV)
└── clusters/                   # Optional clustering analysis
```

### Key Output Files

#### Quality Control Reports
- **normalization_reports/**: HTML reports comparing normalization methods
- **celltype_reports/**: HTML selection analysis per cell type
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
These files can be loaded into your QuPath project to visualize and validate the predicted cell types.
