# ClassyFlow

**Automated Cell Type Classification Pipeline for Multiplex Imaging Data**

[![Nextflow](https://img.shields.io/badge/nextflow%20DSL2-%E2%89%A523.04.0-23aa62.svg)](https://www.nextflow.io/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](docs/)

## Overview

ClassyFlow is a comprehensive, automated, Nextflow pipeline for supervised cell type classification from multiplex imaging data.

<img src="https://github.com/dimi-lab/ClassyFlow/blob/main/images/classyFlow_banner.PNG" width="600"/>

## Quick Start

### Requirements

- [Nextflow](https://www.nextflow.io/) ≥ 23.04.0
- [Python](https://www.python.org/) ≥ 3.10
- [Docker](https://www.docker.com/) or [Conda](https://docs.conda.io/) (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/dimi-lab/ClassyFlow.git
cd ClassyFlow

# Install Python dependencies
pip install -r requirements.txt
```

*For more advanced options, refer to [Installation Instructions](docs/installation.md)*
### Test with example data
```
nextflow run main.nf -c nextflow.config
```

### Running your own data

1. Organize input data. See [Input Data Requirements](docs/user-guide.md#input-data-requirements)
2. Adjust configurations in `nextflow.config`. See [Parameter Reference](docs/parameter-reference.md)
3. Run ClassyFlow

```bash
nextflow run main.nf -profile <>
```
*For more information on running ClassyFlow, see [Pipeline Usage](docs/user-guide.md#pipeline-usage)*

## Output Structure

```
output/
├── celltype_reports/      # Feature analysis & training summaries (PDF)
├── normalization_reports/ # Data transformation comparisons (PDF)
├── model_reports/         # Training & validation performance (PDF)
├── models/               # Trained classifiers & encoders (PKL)
├── celltypes/            # Final predictions (TSV)
├── clusters/             # Optional clustering analysis
└── normalization_files/  # Processed data files (TSV)
```

## What Does This Pipeline Do?

- **Automated Data Integration:** Merges and harmonizes quantification tables from QuPath or similar tools, handling missing markers and batch effects.
- **Flexible Normalization:** Supports multiple normalization strategies (Box-Cox, quantile, min-max, log) with automated selection of the best approach per batch.
- **Feature Engineering & Selection:** Identifies and selects optimal features for cell type classification using recursive feature elimination and Lasso-based methods.
- **Model Training & Hyperparameter Search:** Efficiently trains and tunes multiple machine learning models (e.g., XGBoost, Random Forest) with cross-validation and parameter grid search.
- **Holdout & Cross-Batch Evaluation:** Splits data into training and holdout sets, ensuring robust, unbiased model evaluation across batches.
- **Automated Reporting:** Generates comprehensive PDF and CSV reports for each step, including feature importance, model performance, and cell type distributions.
- **On-the-Fly Model Generation:** Designed for rapid, reproducible model building—simply point to your data and let the pipeline handle the rest.

## Why Use ClassyFlow?

- **Economical & Reproducible:** Minimizes manual intervention and scripting, reducing errors and ensuring reproducibility.
- **Scalable:** Easily adapts to new datasets, markers, or cell types—no need to rewrite code for each experiment.
- **Transparent:** Outputs all intermediate and final results, making it easy to audit and interpret every step of the modeling process.
- **Customizable:** Modular design allows users to swap in new normalization, feature selection, or modeling strategies as needed.

## Typical Use Case

1. **Input:** Exported single-cell quantification tables from QuPath (or similar), with cell type annotations.
2. **Run:** Launch the pipeline with a single command, specifying your input directories and desired output location.
3. **Output:** Ready-to-use, batch-corrected, and validated ML models for cell classification, along with detailed reports and predictions for all cells.

---

**ClassyFlow** is ideal for researchers and bioinformaticians seeking a fast, reliable, and transparent way to generate high-quality cell classification models from multiplexed imaging data.


## Documentation

📚 **[Complete Documentation](docs/)** - Comprehensive user guide and technical reference

- **[Installation Instructions](docs/installation.md)** - Installation options
- **[User Guide](docs/user-guide.md)** - Usage
- **[Parameter Reference](docs/parameter-reference.md)** - Complete configuration options
- **[Pipeline Details](docs/pipeline-details.md)** - Technical workflow documentation

## Getting Help

- 📖 [Documentation](docs/)
- 🐛 [Issue Tracker](https://github.com/dimi-lab/ClassyFlow/issues)

## Citation

If you use ClassyFlow in your research, please cite:

```bibtex
@software{ClassyFlow2024,
  title={ClassyFlow: Automated Cell Type Classification Pipeline for Multiplex Imaging},
  author={Raymond Moore},
  year={2024},
  url={https://github.com/dimi-lab/ClassyFlow},
  version={1.0.0}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
