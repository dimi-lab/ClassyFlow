<img src="https://github.com/dimi-lab/mxif_clustering_pipeline/images/classyFlow_banner.PNG" width="1000"/>


# Supervised Classifier Workflow
Nextflow workflow specific to outlining approaches to building and testing dynamic ML classifiers

## Requirements/Dependencies

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
     
------------------------------------------------------------------------

## Instructions

Note: This pipeline requires exported QuPath (0.5+) measurement tables (quantification files) generated from segmented single cell MxIF images. Those exported files need to include some annotated classification lables.









