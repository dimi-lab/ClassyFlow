# ClassyFlow

**Automated Cell Type Classification Pipeline for Multiplex Imaging Data**

[![Nextflow](https://img.shields.io/badge/nextflow%20DSL2-%E2%89%A523.04.0-23aa62.svg)](https://www.nextflow.io/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

ClassyFlow is a comprehensive, automated pipeline for cell type classification from multiplex imaging data.

## Pipeline Architecture

```mermaid
flowchart TD
    A[Input Batches] --> B[Data Integration]
    B --> C[Panel QC] 
    C --> D[Harmonization]
    D --> E[Normalization]
    E --> F[Train/Test Split]
    F --> G[Feature Selection]
    F --> H[Model Training]
    G --> H
    H --> I[Prediction]
    E --> I
    I --> J[Results]
    
    classDef process fill:#6A1B9A,stroke:#4A148C,stroke-width:2px,color:#fff
    classDef workflow fill:#2E7D32,stroke:#1B5E20,stroke-width:2px,color:#fff
    classDef output fill:#D84315,stroke:#BF360C,stroke-width:2px,color:#fff
    
    class A,B,C,D process
    class E,F,G,H workflow  
    class I,J output
```

## Quick Start

### Requirements

- [Nextflow](https://www.nextflow.io/) ≥ 23.04.0
- [Python](https://www.python.org/) ≥ 3.10
    - fpdf 1.7.2
    - numpy 1.23.5
    - matplotlib 3.8.0
    - dataframe-image 0.2.3
    - pandas 2.2.0
    - seaborn 0.13.1
    - xgboost 1.6.2
    - scipy 1.12.0
    - scikit-learn 1.4.0
- [Docker](https://www.docker.com/)
- [Conda](https://docs.conda.io/) (recommended)


### Installation

#### Clone repository to your working directory
```bash
# Clone the repository
git clone https://github.com/yourusername/ClassyFlow.git
cd ClassyFlow
```

#### Install Python dependencies
##### Option 1: Installing locally
```bash
# Create Conda environment (optinal but recommended)
conda create -n classyflow python=3.10

# Install python packages
pip install -r requirements.txt
```

##### Option 2: Docker Alternative
```bash
# <<<Add docker related instructions here>>>

# Run pipeline in container
nextflow run main.nf -profile docker
```

### Basic Usage

```bash
# Run pipeline with example data
nextflow run main.nf -c nextflow_ovtma.config
```

## Input Data Format

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

The quantification files need to include some annotated classification labels to train the models. A minimum of 30-50 annotations per phenotype is recommended. 

### Required Columns

- `Classification`: Cell type annotation
- `Centroid X µm`: X coordinate of the cell's centroid
- `Centroid Y µm`: Y coordinate of the cell's centroid
- Marker intensities (e.g., `CD3: Cell: Mean`, `CD4: Cell: Mean`)

## Configuration

### Key Parameters -- Under construction

| Parameter | Description | Default |
|-----------|-------------|---------|
| `input_dirs` | Path to input directory | `"./input"` |
| `output_dir` | Output directory for results | `"./output"` |
| `bit_depth`  | Original Image Capture quality: 8-bit (pixel values will be 0-255) or 16-bit (pixel values will be 0-65,535) | `8-bit` |
| `qupath_object_type` | "CellObject" has two ROIs, jointly and 4 components [Cell, Cytoplasm, Membrane, Nucleus] from QuPath; 'DetectionObject' is Single Whole Cell or Nucleus only | `CellObject` |

### Execution Profiles

- **local**: Single machine execution (default)
- **slurm**: SLURM cluster execution  
- **gcp**: Google Cloud Platform

### Advanced Usage
```bash
# Run with SLURM cluster
nextflow run main.nf -c nextflow.config -profile slurm

# Run on GCP
nextflow run main.nf -c nextflow.config -profile gcp
```

## Output Structure

```
output/
├── normalization_reports/ # Data normalization comparison reports
├── celltype_reports/      # Feature analysis & training summaries
├── model_reports/         # Training & validation performance report
├── models/                # Trained classifiers
├── celltypes/             # Final cell type predictions
├── clusters/              # Optional clustering analysis using scimap
```

## Workflow Details

### Data Processing Pipeline

1. **Data Integration**: Merges quantification files from multiple batches
2. **Panel Design Check**: Analyzes marker presence across datasets
3. **Feature Harmonization**: Adds synthetic noise for missing markers
4. **Normalization Testing**: Evaluates Box-Cox, quantile, min-max, and log transformations
5. **Data Splitting**: Creates stratified train/holdout sets

### Feature Selection Strategy

1. **Binary Classification Setup**: One-vs-rest approach per cell type
2. **LASSO Regularization**: L1 penalty optimization via grid search
3. **Recursive Feature Elimination**: Cross-validated feature ranking
4. **Performance Evaluation**: Cell type-specific feature importance analysis
5. **Feature Integration**: Combines results across all cell types

### Model Development

1. **Parameter Space Definition**: Systematic grid search configuration
2. **Hyperparameter Tuning**: Cross-validated XGBoost optimization
3. **Model Training**: Multiple models with early stopping
4. **Holdout Validation**: Performance testing on reserved data
5. **Model Selection**: Best performer based on validation metrics

### Getting Help

- [Issue Tracker](https://github.com/yourusername/ClassyFlow/issues)
- Email: [your.email@mayo.edu](mailto:your.email@institution.edu)

## Citation

If you use ClassyFlow in your research, please cite:

```bibtex
@software{ClassyFlow2024,
  title={ClassyFlow: Automated Cell Type Classification Pipeline for Multiplex Imaging},
  author={Raymond Moore},
  year={2024},
  url={https://github.com/yourusername/ClassyFlow},
  version={1.0.0}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Links

- [Homepage](https://github.com/dimi-lab/ClassyFlow)
- [QuPath](https://qupath.github.io/)

---

**Scalable • Reproducible • Validated • Open Source**