# ClassyFlow

**Automated Cell Type Classification Pipeline for Multiplex Imaging Data**

[![Nextflow](https://img.shields.io/badge/nextflow%20DSL2-%E2%89%A523.04.0-23aa62.svg)](https://www.nextflow.io/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](docs/)

## Overview

ClassyFlow is a comprehensive, automated, Nextflow pipeline for supervised cell type classification from multiplex imaging data.


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
    
    class A,B,C,D process
    class E,F,G,H workflow  
    class I,J output
```

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
### Test with example data
```
nextflow run main.nf -c nextflow_ovtma.config
```

### Running your own data

1. Organize input data. See [Input Data Requirements](docs/user-guide.md#input-data-requirements)
2. Adjust configurations in `nextflow.config`. See [Parameter Reference](docs/parameter-reference.md)
3. Run ClassyFlow

```bash
nextflow run main.nf
```
For more information on running ClassyFlow, see [Pipeline Usage](docs/user-guide.md#pipeline-usage)

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


## Documentation

📚 **[Complete Documentation](docs/)** - Comprehensive user guide and technical reference

- **[User Guide](docs/user-guide.md)** - Installation and usage
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
