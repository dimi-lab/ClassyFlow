# Parameter Reference

Complete configuration guide for ClassyFlow parameters.


## Core Pipeline Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `help` | `false` | Display help message and exit |
| `input_dir` | `"./input"` | Path to input directory |
| `output_dir` | `"./output"` | Output directory for all results |

## Input Data Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `slide_contains_prefix` | `"True"` | Image names contain "_" delimited slide prefix |
| `folder_is_slide` | `"False"` | Folder represents single slide with multiple ROIs |
| `quant_file_extension` | `".tsv"` | File extension for quantification files |
| `quant_file_delimiter` | `"\\t"` | Column delimiter (tab or comma) |
| `bit_depth` | `"16-bit"` | Original Image Capture quality: 8-bit (0-255) or 16-bit (0-65,535) |
| `qupath_object_type` | `"DetectionObject"` | QuPath object type: "CellObject" (4 components) or "DetectionObject" (single cell/nucleus) |

## Marker and Feature Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `nucleus_marker` | `"DAPI"` | Nuclear marker for normalization reference |
| `housekeeping_marker` | `"S6"` | Housekeeping gene marker |
| `classifed_column_name` | `"Classification"` | Column name containing cell type annotations |
| `exclude_markers` | `"Arg1\|BAD\|B2M\|..."` | Pipe-delimited markers to exclude (regex pattern) |
| `plot_fraction` | `0.25` | Fraction of data used for quality control plots |

### Excluded Markers (Default)
```
Arg1, BAD, B2M, CCAM5, CCR8, CD209, CTLA4, GATA3, Gal3, Gal9, GzB, 
Her2, ILT4, Ki67, PD1, PDL1, T-BET, TIGIT, TCF7, iNOS, CD103a, LAG3, 
HLAI, DAPI_R28, NKG7, TIM3, TenC, IDO
```

**Note**: Exclusion uses regex matching. Be careful with partial names (e.g., excluding "CD3" will also exclude "CD31", "CD33").

## Normalization Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `override_normalization` | `"boxcox"` | Normalization method for downstream analysis |
| `downsample_normalization_plots` | `0.5` | Fraction of data used for normalization plots |
| `quantile_split` | `1024` | Number of quantiles for QuantileTransformer (good for 16-bit, but not for 8-bit/DualBand/Hyperion) |

### Available Normalization Methods
- `"boxcox"` - Box-Cox power transformation (recommended for imaging data)
- `"quantile"` - Quantile normalization 
- `"minmax"` - Min-max scaling
- `"log"` - Log transformation
- `null` - Un-normalized values

## Data Splitting and Training
| Parameter | Default | Description |
|-----------|---------|-------------|
| `holdout_fraction` | `0.1` | Fraction of data reserved for final validation |
| `filter_out_junk_celltype_labels` | `"??,?,0,Negative,Ignore*"` | Cell types to exclude from training |
| `minimum_label_count` | `20` | Minimum annotations required per cell type for that cell type data to be included in the training set|

## Feature Selection
| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_rfe_nfeatures` | `5` | Minimum features for RFE evaluation |
| `max_rfe_nfeatures` | `20` | Maximum features for RFE evaluation |

## XGBoost Model Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_xgb_cv` | `5` | Maximum cross-validation iterations |
| `xgb_depth_start` | `2` | Minimum tree depth for grid search |
| `xgb_depth_stop` | `6` | Maximum tree depth for grid search |
| `xgb_depth_step` | `3` | Step size for tree depth search |
| `xgb_learn_rates` | `"0.1"` | Comma-separated learning rates to test |

## Prediction Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `predict_class_column` | `'CellType'` | Column name for predictions |
| `predict_le_encoder_file` | `"{output_dir}/models/classes.npy"` | Path to label encoder |
| `predict_columns_to_export` | `'Centroid X µm,Centroid Y µm,Image,CellTypePrediction'` | Columns in output files |
| `predict_cpu_jobs` | `16` | CPU cores for prediction |

## Optional Features
| Parameter | Default | Description |
|-----------|---------|-------------|
| `run_get_leiden_clusters` | `false` | Enable clustering-based feature augmentation |
| `scimap_resolution` | `0.5` | Resolution parameter for Leiden clustering |