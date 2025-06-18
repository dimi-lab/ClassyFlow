<img src="https://github.com/dimi-lab/ClassyFlow/blob/main/images/classyFlow_banner.PNG" width="1000"/>


# ClassyFlow: Automated ML Pipeline for Multiplex Immunofluorescence Cell Classification

**ClassyFlow** is a robust, modular Nextflow pipeline designed to streamline and automate the process of building, evaluating, and deploying machine learning models for multiplex immunofluorescence (MxIF) single-cell image data. 

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



## Requirements/Dependencies

-   Nextflow 23.04.2 (requires: bash, java 11 [or later, up to 21] git and docker)
-   Python 3.10+ requirements file available.
     
------------------------------------------------------------------------

## Instructions

Note: This pipeline requires exported QuPath (0.5+) measurement tables (quantification files) generated from segmented single cell MxIF images. Those exported files need to include some annotated classification lables.

<img src="https://github.com/dimi-lab/ClassyFlow/blob/main/images/qupath_example_exporting.PNG" width="750"/>

1. 






### Configurable parameters

| Field               |Description   |
|-------------|-----------------------------------------------------------|
| bit_depth | Original Image Capture quality: 8-bit (pixel values will be 0-255) or 16-bit (pixel values will be 0-65,535) |
| qupath\_object\_type |  "CellObject" has two ROIs, jointly and 4 components [Cell, Cytoplasm, Membrane, Nucleus] from QuPath; 'DetectionObject' is Single Whole Cell or Nucleus only|
| classifed\_column_name| |
| exclude_markers | |
| nucleus_marker | |
| override_normalization | |
| downsample\_normalization_plots | |
| holdout_fraction | |
| filter_out\_junk\_celltype_labels | |
| minimum\_label_count | |
| max\_xgb_cv | |






