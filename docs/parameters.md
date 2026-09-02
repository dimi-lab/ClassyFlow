# Parameters

Every setting below can be changed in two ways:

```bash
# on the command line, for one run
nextflow run main.nf -profile local --holdout_fraction 0.2

# or edit the params block in nextflow.config, to make it stick
```

If the same setting appears in more than one place, the command line wins, then
any file you pass with `-c`, then `nextflow.config`.

- [Core](#core)
- [Input files](#input-files)
- [Normalisation](#normalisation)
- [Holdout and label filtering](#holdout-and-label-filtering)
- [Feature selection](#feature-selection)
- [Model search](#model-search)
- [Prediction](#prediction)
- [Memory and scale](#memory-and-scale)
- [Assets](#assets)

---

## Core

| Parameter | Default | What it does |
| --- | --- | --- |
| `input_dirs` | the two folders under `data/` | List of folders, one per batch. The folder name becomes the batch name. Required. |
| `output_dir` | `"classyflow_output"` | Where results are written. A relative path is resolved from wherever you launched the run. |
| `report_mode` | `"both"` | Which report to build: `full`, `light`, or `both`. Any other value produces no report at all. |
| `help` | `false` | Print the usage message and stop. |

## Input files

| Parameter | Default | What it does |
| --- | --- | --- |
| `folder_is_slide` | `"True"` | Set when each folder holds several regions of one slide. |
| `slide_contains_prefix` | `"False"` | Set when each folder holds several slides and the slide name is the part of the image name before the first `_`. |
| `quant_file_extension` | `".tsv"` | Which files in each folder to read. |
| `quant_file_delimiter` | `"\\t"` | Column separator. Tab is `"\\t"`; use `","` for CSV. |
| `qupath_object_type` | `"DetectionObject"` | `DetectionObject` for one measurement set per cell (`CD4: Mean`). `CellObject` for four compartments per cell (`CD4: Nucleus: Mean`). |
| `classifed_column_name` | `"Classification"` | The column holding the hand-assigned cell type. (The spelling of the parameter name is missing an `i` — that is the real name.) |
| `batch_correct_column_names` | `false` | Rename marker columns to the canonical names in `assets/markers.yaml` before analysis. Turn this on when batches spell the same marker differently. |
| `exclude_markers` | a `\|`-separated list, see below | Markers to drop at read time. |
| `celltype_profile` | `null` | Path to a YAML file of expected markers per cell type. Adds a concordance check to the report; does not change the model. |

`folder_is_slide` and `slide_contains_prefix` are **text**, compared against the
string `"True"`. Writing `--folder_is_slide true` in lower case silently turns
it off. Set exactly one of the two.

The default `exclude_markers` is:

```
Bcl2|Ki67|HLA-A|ER|IDO1|Beta-actin|HLA-DR|PCNA|GzB|Podoplanin|CD45RO|ICOS|LAG3|Irr-61|Irr-66|Irr-96
```

This is matched against column names as a pattern, not as whole words, so a
short name will also match longer ones — `CD3` would remove `CD38` too. Check
your panel for overlaps before adding entries.

## Normalisation

| Parameter | Default | What it does |
| --- | --- | --- |
| `override_normalization` | `"boxcox"` | Which transform to apply: `boxcox`, `quantile`, `minmax`, `log`, or `none` to skip normalisation. |
| `plot_target_feature_suffix` | `"Mean"` | Which statistic to use for the gating and the normalisation plots. |
| `quantile_split` | `1024` | Number of quantiles, used only when `override_normalization = "quantile"`. |

The comment in `nextflow.config` lists `null` as an option, but only the text
`"none"` actually skips normalisation.

## Holdout and label filtering

| Parameter | Default | What it does |
| --- | --- | --- |
| `holdout_fraction` | `0.1` | Fraction of each batch set aside and never used for training or feature selection. The reported performance comes from these cells. |
| `minimum_label_count` | `18` | A cell type with fewer labelled cells than this is dropped. Too few examples cannot be learned or scored reliably. |
| `filter_out_junk_celltype_labels` | `"??,?,0,Negative,Ignore*"` | Label values thrown away before training. Add your own placeholder labels here. |

## Feature selection

For each cell type, ClassyFlow tries several feature-set sizes and keeps the
best. These three settings define the sizes it tries.

| Parameter | Default | What it does |
| --- | --- | --- |
| `min_rfe_nfeatures` | `2` | Smallest number of features to try. |
| `max_rfe_nfeatures` | `10` | Largest number of features to try. |
| `rfe_step` | `2` | Step between them. |

The defaults try 2, 4, 6, 8 and 10 features. A wider range or a smaller step
finds better feature sets but takes proportionally longer.

## Model search

ClassyFlow tries every combination of tree depth and learning rate, scores each
with cross-validation, and keeps the best two.

| Parameter | Default | What it does |
| --- | --- | --- |
| `xgb_depth_start` | `2` | First tree depth to try. |
| `xgb_depth_stop` | `9` | Stop before this depth. |
| `xgb_depth_step` | `3` | Step between depths. |
| `xgb_learn_rates` | `"0.1,0.7,1.0"` | Learning rates to try, comma-separated. |
| `max_xgb_cv` | `10` | Number of cross-validation folds. |

The defaults give depths 2, 5 and 8, times three learning rates — nine
combinations, each run as its own job. This is the slowest part of the
pipeline, so it is the first place to cut if you want a faster run.

> **The grid must contain at least two combinations.** ClassyFlow compares the
> best two configurations against each other, so a grid of one will fail.

## Prediction

| Parameter | Default | What it does |
| --- | --- | --- |
| `predict_class_column` | `"CellType"` | Name given to the prediction column. |
| `predict_columns_to_export` | `"Centroid X µm,Centroid Y µm,Image,CellTypePrediction"` | Which columns appear in the per-cell result files. Class probabilities and the QC flag are always added. |
| `predict_cpu_jobs` | `16` | How many cells to score in parallel. Lower this on a small machine — it is set independently of the CPUs the job requests. |

## Memory and scale

| Parameter | Default | What it does |
| --- | --- | --- |
| `target_splitting_size` | `55000` | Largest number of rows held in memory at once. Bigger batches are dealt into several pieces, processed separately, and rejoined before prediction. Set to `""` to turn splitting off, which needs a high-memory machine. |

This is the setting to reach for when a run is killed for using too much
memory. Lowering it makes each piece smaller and the run slower.

## Assets

These are set in `main.nf` rather than `nextflow.config`, so you will not find
them in the params block. They can still be overridden on the command line. The
files must exist — a missing one stops the run immediately.

| Parameter | Default | What it does |
| --- | --- | --- |
| `marker_vocabulary` | `assets/markers.yaml` | Canonical marker names and their aliases. Used when `batch_correct_column_names` is on. |
| `html_template` | `assets/html_templates` | Folder of report templates. |
| `letterhead` | `assets/images/Classyflow_banner_purple.png` | Banner image at the top of the reports. |
| `pipeline_version` | from `manifest.version` in `nextflow.config` | Version string stamped into the reports. Edit the manifest, not this param. |
| `config_file` | the first config file in use | Copied into the results as a record of the settings the run used. |

---

## Where to look when something needs tuning

| Situation | Try |
| --- | --- |
| Run killed for using too much memory | Lower `target_splitting_size` |
| Run is too slow | Shrink the model grid (`xgb_depth_step`, `xgb_learn_rates`) and narrow the feature range |
| A cell type is missing from the results | It had fewer labelled cells than `minimum_label_count` |
| Markers do not line up between batches | Set `batch_correct_column_names = true` and fill in `assets/markers.yaml` |
| Reported accuracy looks too good | Raise `holdout_fraction` so more cells are held back |
