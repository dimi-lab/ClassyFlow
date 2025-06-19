<img src="https://github.com/dimi-lab/ClassyFlow/blob/main/images/classyFlow_banner.PNG" width="1000"/>


# Supervised Classifier Workflow
Nextflow workflow specific to outlining approaches to building and testing dynamic ML classifiers

## Setup

Copy `.envrc.template` to `.envrc`. Fill in the values appropriately.

### Requirements/Dependencies

-   Nextflow 23.04.2 (requires: bash, java 11 [or later, up to 21] git and docker)
-   Python 3.10+
    - fpdf				1.7.2
    - numpy				1.23.5
    - matplotlib		3.8.0
    - dataframe-image	0.2.3
    - pandas			2.2.0
    - seaborn			0.13.1
    - xgboost			1.6.2
    - scipy				1.12.0
    - scikit-learn		1.4.0

Install to the user site-packages.

```
pip install --user -r requirements.txt
```

Install the chromium headless browser tools:

```
apt install chromium-driver
```

### Run the sample pipeline

To use the slurm backend, run this from the repository root:

```sh
nextflow run main.nf -c nextflow_ovtma.config
```

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






