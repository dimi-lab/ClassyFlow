# ClassyFlow: Simplified ML Pipeline for Multiplex Immunofluorescence Cell Classification

**ClassyFlow** is a robust, modular Nextflow pipeline designed to streamline and automate the process of building, evaluating, and deploying machine learning models for multiplex immunofluorescence (MxIF) single-cell image data. 

<img src="https://github.com/dimi-lab/ClassyFlow/blob/main/images/classyFlow_banner.PNG" width="1000"/>

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

 
### Configurable Parameters

| Parameter                     | Small Data Defaults                        | Description                                                                                   |
|-------------------------------|--------------------------------------|-----------------------------------------------------------------------------------------------|
| `output_dir`                  | `classyflow_output`                  | Output directory for all pipeline results                                                     |
| `slide_contains_prefix`        | `True`                               | If true, assumes folder is a batch of slides & image name contains "_" delimited prefix       |
| `folder_is_slide`              | `False`                              | If true, assumes folder contains multiple ROIs for a single slide/sample                      |
| `quant_file_extension`         | `.tsv`                               | File extension for quantification tables                                                      |
| `quant_file_delimiter`         | `\t`                                 | Delimiter for quantification tables (`\t` for tab, `,` for comma)                             |
| `bit_depth`                    | `16-bit`                             | Image bit depth: `8-bit` (0-255) or `16-bit` (0-65535)                                        |
| `qupath_object_type`           | `DetectionObject`                    | QuPath object type: `CellObject` or `DetectionObject`                                         |
| `nucleus_marker`               | `DAPI_AF_R01`                        | Marker name for nucleus identification                                                        |
| `plot_fraction`                | `0.25`                               | Fraction of data to use for plotting                                                          |
| `classifed_column_name`        | `Classification`                     | Column name in quantification tables for cell type labels                                     |
| `exclude_markers`              | See config                           | Pipe-delimited list of marker names to exclude (regex supported)                              |
| `housekeeping_marker`          | `S6`                                 | Marker used as a housekeeping control                                                         |
| `override_normalization`       | `boxcox`                             | Normalization method: `minmax`, `boxcox`, `log`, `quantile`, or `null` for auto              |
| `downsample_normalization_plots`| `0.5`                               | Fraction of data to use for normalization plots                                               |
| `quantile_split`               | `1024`                               | Number of quantiles for quantile normalization (good for 16-bit images)                       |
| `max_xgb_cv`                   | `10`                                 | Maximum number of cross-validation folds for XGBoost                                          |
| `xgb_depth_start`              | `2`                                  | Starting value for XGBoost tree depth                                                         |
| `xgb_depth_stop`               | `6`                                  | Stopping value for XGBoost tree depth                                                         |
| `xgb_depth_step`               | `3`                                  | Step size for XGBoost tree depth                                                              |
| `xgb_learn_rates`              | `0.1`                                | Learning rates for XGBoost (comma-separated string)                                           |
| `predict_class_column`         | `CellType`                           | Column name for predicted cell type                                                           |
| `predict_le_encoder_file`      | `${params.output_dir}/models/classes.npy` | Path to label encoder file for predictions                                              |
| `predict_columns_to_export`    | `Centroid X µm,Centroid Y µm,Image,CellTypePrediction` | Columns to export in prediction output                                 |
| `predict_cpu_jobs`             | `16`                                 | Number of CPUs to use for prediction                                                          |
| `run_get_leiden_clusters`      | `false`                              | Whether to run Leiden clustering for feature engineering                                      |
| `scimap_resolution`            | `0.5`                                | Resolution parameter for scimap clustering                                                    |
| `holdout_fraction`             | `0.1`                                | Fraction of data per batch to withhold for holdout evaluation                                 |
| `filter_out_junk_celltype_labels`| `??,?,0,Negative,Ignore*`          | Cell type labels to filter out                                                                |
| `minimum_label_count`          | `20`                                 | Minimum number of cells per label for inclusion                                               |
| `min_rfe_nfeatures`            | `2`                                  | Minimum number of features for RFE                                                            |
| `max_rfe_nfeatures`            | `3`                                  | Maximum number of features for RFE                                                            |





