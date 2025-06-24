# Pipeline Details

Technical documentation of ClassyFlow's automated cell type classification workflow.

## Table of Contents
- [Pipeline Overview](#pipeline-overview)
- [Data Integration & Quality Control](#data-integration--quality-control)
- [Normalization](#normalization)
- [Feature Selection](#feature-selection)
- [Model Training](#model-training)
- [Final Prediction](#final-prediction)

## Pipeline Overview

ClassyFlow implements a systematic approach to cell type classification. Below are details of each step of the pipeline. For a visual representation of this pipeline, refer to the [ClassyFlow Workflow Diagram](detailed_workflow_diagram.mermaid)


<!-- ### Expected Processing Times
- **Small datasets** (< 10K cells): 30 minutes - 2 hours
- **Medium datasets** (10K-100K cells): 2-6 hours  
- **Large datasets** (> 100K cells): 6-24 hours -->


### 1. File merging and data filtering
Combines quantification files from multiple batches into unified datasets. This step will also apply any marker exclusion filters as defined in the configuration file

### 2. Panel Design Analysis and Feature Harmonization
ClassyFlow can work with multiple input batches. Since different batches may be coming from different projects, they may have different markers and study designs. First, the pipeline checks each batch and creates marker presence/absence matrix across all batches. For missing markers, the pipeline will generate synthetic, bakground noise using sklearn's [make_blobs](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html). This ensures that all batches have identical feature sets

### 3. Normalization
ClassyFlow supports multiple normalization approaches and generates comprehensive HTML reports to help the user choose the best normalization method for their dataset. The normalization method is specified in the `override_normalization` parameter in the config file.

#### Box-Cox Transformation (`boxcox`)
- **Purpose**: Variance stabilization and normality improvement
- **Method**: Power transformation with optimal lambda selection
- **Best for**: Imaging data with heteroscedastic variance

#### Quantile Normalization (`quantile`)
- **Purpose**: Batch effect reduction through distribution matching
- **Method**: Maps all features to uniform distributions
- **Best for**: Multi-batch studies with strong batch effects

#### Min-Max Scaling (`minmax`)
- **Purpose**: Feature scaling to consistent ranges
- **Method**: Linear scaling to [-2, 2] range
- **Best for**: Data with consistent distributions

#### Log Transformation (`log`)
- **Purpose**: Dynamic range compression
- **Method**: Natural log with +1 offset
- **Best for**: Exponentially distributed intensities

*Note: if another value is passed to the `override_normalization` parameter, the pipeline will run all 4 normalization methods and generate reports for each*

### 4. Generate Training and Holdout Datasets
Create a stratified training and holdout datasets from the user-provided annotations. The holdout dataset will be used for model validation later in the pipeline. This step will also generate an annotation_report.html summarizing the training/holdout split.

### 5. Feature Selection

ClassyFlow implements a sophisticated cell type-specific feature selection strategy. First, a one-vs-rest binary labels are created for each cell type. Next, a grid seaech is performed to identify the best-performing Lasso alpha value. A recursive feature elimination method is used to identify the optimal number of features to use for each cell type. An html report for each cell type is generated, including alpha optimization plots, feature importance rankings, and RFE performance curves. Finally, features selected for all cell types are consolidated to be used in  model training.

### 6. Model Training

First step in the model training module is a grid search for XGBoost hyperparameter optimization. These hyperparameters are used to train multiple XGBoost models, and each model performance is evaluated on a train/test split. The top two performing models are chosen as candidates for further evaluation, and a HTML report is generated to summarize the results. Next, the holdout dataset is used to perform a rigorous evaluation of the top two models. A HTML report is generated for each model, which includes statistical analysis, confusion matrices and ROC curves. Using the results from the holdout evaluaiton, the pipeline selects the best performing model for cell type prediction.

### 7. Cell Type Prediction
The best performing model is used to predict cell types across all samples, and outputs the results in a user-friendly format that can be loaded into QuPath and downstream analyses.

**Output Format**:
```tsv
Image               Centroid X µm   Centroid Y µm   CellTypePrediction
TMA_Core_001.tif   1245.67         2890.34         T cell
TMA_Core_001.tif   1356.78         2901.45         B cell
```

**Quality Indicators**:
- High holdout accuracy indicates good generalization
- Consistent performance across cell types
- Clear feature importance rankings
- Well-calibrated prediction confidence