#!/usr/bin/env nextflow
import groovy.json.JsonOutput

// Using DSL-2
nextflow.enable.dsl=2
println "Active profile: ${workflow.profile}"

//Static Assests for beautification
params.letterhead = file("${projectDir}/assets/images/Classyflow_banner_purple.png", checkIfExists: true)
params.html_template = file("${projectDir}/assets/html_templates", checkIfExists: true)
params.marker_vocabulary = file("${projectDir}/assets/markers.yaml", checkIfExists: true)
params.pipeline_version = "1.0"

params.config_file = file(workflow.configFiles[0], checkIfExists: true)

// Build Input List of Batches
Channel.fromList(params.input_dirs)
		.ifEmpty { error "No files found in ${params.input_dirs}" }
		.set { batchDirs }
			
// Import sub-workflows
include { normalization_wf } from './modules/normalizations'
include { featureselection_wf } from './modules/featureselections'
include { modelling_wf } from './modules/makemodels'


// -------------------------------------- //
// Function which prints help message text
def helpMessage() {
    println """
    ClassyFlow -- supervised cell-type classification for multiplex imaging.

    ClassyFlow reads one directory of QuPath quantification tables per batch,
    trains a classifier on the cells that are already labelled, predicts a cell
    type for every remaining cell, and writes an HTML report.

    Usage:
      nextflow run main.nf -profile local
      nextflow run main.nf -profile docker --input_dirs '["/data/batchA","/data/batchB"]'

    Common options:
      --input_dirs      List of input directories, one per batch. Required.
      --output_dir      Where results are written (default: classyflow_output)
      --report_mode     Which report to build: light, full or both (default: both)
      --override_normalization
                        minmax, boxcox, log, quantile or none (default: boxcox)
      --holdout_fraction
                        Fraction of each batch withheld for evaluation (default: 0.1)
      --help            Print this message and exit

      -profile          Configuration profile: local, docker, slurm or gcp
      -resume           Reuse results from a previous run

    Every parameter can also be set in nextflow.config. See docs/parameters.md
    for the full list, and the README for input format and outputs.
    """.stripIndent()
}

// Define a process to merge tab-delimited files and save as pickle
process MERGE_TAB_DELIMITED_FILES {
    tag { batchID }

	input:
    path subdir

    output:
    path("merged_dataframe_${batchID}*.pkl"), emit: batchtables

    script:
    batchID = subdir.baseName
    """
    merge_files.py \
        "$subdir" \
        "${params.exclude_markers}" \
        ${params.slide_contains_prefix == 'True' ? '--slide_by_prefix' : ''} \
        ${params.folder_is_slide == 'True' ? '--folder_is_slide' : ''} \
        --input_extension ${params.quant_file_extension} \
        --input_delimiter '${params.quant_file_delimiter}' \
        --batchID ${batchID} \
        ${params.target_splitting_size ? "--target_size ${params.target_splitting_size}" : ""}
    """
}

process COLUMN_FORMAT_AND_FIX {
    input:
    path tables_pkl
    path marker_vocabulary

    output:
    path("*_fx.pkl"), emit: batchtables

    script:
    """
    fixup_columns.py \
        --input_table ${tables_pkl} \
        --marker_vocabulary ${marker_vocabulary}
    """
}

/* 
 * For each input pickle file (merged quantification tables), extract all unique marker/channel names
 * (from columns containing 'Mean'), and generate a presence/absence matrix showing which markers
 * are present in each batch or panel. This matrix is saved as 'panel_design.csv' for downstream
 * comparison of panel designs across
 */
process CHECK_PANEL_DESIGN {
	input:
	path(tables_pkl_collected)

    output:
    path 'panel_design.csv', emit: paneldesignfile
    path 'input_batch_metrics.json', emit: input_metrics

    script:
    """
    compare_panel_designs.py \
        ${tables_pkl_collected.join(' ')} \
        -o panel_design.csv \
        -m input_batch_metrics.json \
        -t ${params.minimum_label_count} \
        -c ${params.classifed_column_name}
    """
}

/*
 * For each batch's merged quantification table, this process checks the panel design to identify any markers
 * that are missing from the data. For each missing marker, it generates a synthetic column of low-noise values
 * (using sklearn's make_blobs) to fill in the missing features, ensuring all batches have a consistent set of markers.
 * The modified table is saved for downstream normalization and modeling.
 */
process ADD_EMPTY_MARKER_NOISE {
    tag { batchID }

	input:
	tuple val(batchID), path(pickleTable)
	path designTable

    output:
    tuple val(batchID), path("merged_dataframe_${batchID}_mod.pkl"), emit: modbatchtables

    script:
    """
    add_empty_marker_noise.py \
        --objtype ${params.qupath_object_type} \
        --pickleTable ${pickleTable} \
        --batchID ${batchID} \
        --designTable ${designTable} \
    """
}

/*
 * This step combines all normalized annotation tables, filters out unwanted cell types,
 * and splits the data into training and holdout sets using stratified sampling based on batch and cell type.
 * It also generates a summary table and a PDF report showing the distribution of cell types in each set,
 * ensuring balanced and reproducible training/validation splits for downstream modeling.
 */
process GENERATE_TRAINING_N_HOLDOUT{
	publishDir(
        path: "${params.output_dir}/annotation",
        pattern: "*.csv",
        mode: "copy"
    )
    
	input:
	path(norms_pkl_collected)

	output:
    path("holdout_dataframe.pkl"), emit: holdout
    path("training_dataframe.pkl"), emit: training
	path("celltypes.csv"), emit: lableFile
    tuple path("training_split_report.json"), path("cell_count_table.csv"), emit: training_holdout_results
    path("per_batch_label_count.csv"), emit: per_batch_label_count

    script:
    """
    split_annotations_for_training.py \
        --classColumn ${params.classifed_column_name} \
        --holdoutFraction ${params.holdout_fraction} \
        --cellTypeNegative "${params.filter_out_junk_celltype_labels}" \
        --minimunHoldoutThreshold ${params.minimum_label_count} \
        --pickle_files "${norms_pkl_collected}"
    """

}

// Run model on everything make results
process PREDICT_ALL_CELLS_XGB{
    
	input:
	tuple path(model_path), path(leEncoderFile)
	tuple val(batchID), path(pickleTable)
	
	output:
	tuple val(batchID), path("*.tsv"), emit: predictions
	
	script:
    """
    predict_celltypes.py \
        --classColumn ${params.predict_class_column} \
        --leEncoderFile ${leEncoderFile} \
        --batchID ${batchID} \
        --infile ${pickleTable} \
        --modelfile ${model_path} \
        --columnsToExport "${params.predict_columns_to_export}" \
        --cpu_jobs ${params.predict_cpu_jobs} \
        --include_probabilities \
        --top_n_probs 2

    """
}

process QC_DENSITY {
   tag { sampleID }
    publishDir "${params.output_dir}/celltypes", pattern: "*_qPRED.tsv", mode: "copy", overwrite: true

    input:
    tuple val(batchID), val(sampleID), path(prediction_tsv)    

    output:
    tuple val(batchID), val(sampleID), path("*_qPRED.tsv"), emit: qc_predictions

    script:
    """
    calculate_bin_density.py --input_tsv ${prediction_tsv} \
        --bin_size 120 \
        --density_cutoff 3
    """
}

process CLASSIFIED_REPORT_PER_SLIDE {
    publishDir(
        path: "${params.output_dir}/final_reports/pages",
        pattern: "*_prediction_report.html",
        mode: "copy"
    )
    
    input:
    tuple val(batchID), val(sampleID), path(prediction_tsv)

    output:
    path("*_prediction_report.html")
    path("*counts.tsv"), emit: classified_counts

    script:
    """
    generate_classified_report.py \
        --input_tsv ${prediction_tsv} \
        --output_html ${sampleID}_prediction_report.html \
        --batch ${batchID}
    """
}

process GENERATE_FINAL_REPORT {
    // 'deep' hashing indexes template file contents so in-place edits inside the
    // staged html_templates/ directory correctly invalidate -resume.
    cache 'deep'
    publishDir "${params.output_dir}/final_reports", pattern: "*.html", mode: 'copy', overwrite: true
    publishDir "${params.output_dir}/final_reports/pages/", pattern: "nextflow.config", mode: 'copy', overwrite: true
    
    input:
    path(input_metrics_json)
    path(aggregated_counts)
    path(norm_html)
    path(fs_html) 
    path(model_html)
    path(template_dir)
    path(letterhead_file)
    path(nf_config, stageAs: "nextflow.config")

    output:
    path("classyflow_report.html"), emit: report_done
    path("nextflow.config")

    script:
    """
    generate_final_report.py \
        --input-metrics ${input_metrics_json} \
        --normalization-html ${norm_html} \
        --feature-selection-html ${fs_html} \
        --model-html ${model_html} \
        --counts-tsv ${aggregated_counts} \
        --template-dir ${template_dir} \
        --report-name classyflow_report.html \
        --letterhead ${letterhead_file} \
        --version ${params.pipeline_version}
    """

}


process GENERATE_LIGHT_REPORT {
    // 'deep' hashing indexes template file contents so in-place edits inside the
    // staged html_templates/ directory correctly invalidate -resume.
    cache 'deep'
    publishDir "${params.output_dir}/final_reports", pattern: "*.html", mode: 'copy', overwrite: true

    input:
    path(input_metrics_json)
    path(holdout_eval_files)   // holdoutEval_*_results.json + confusion/ROC/PR div HTMLs
    path(fs_files)             // feature_selection_*_results.json
    path(concordance_files)    // feature_concordance.csv (+ .json, unread); optional
    path(celltype_profile_file, stageAs: "celltype_profile.yaml")  // optional (may be empty)
    path(aggregated_counts)
    path(template_dir)
    path(letterhead_file)
    path(nf_config, stageAs: "nextflow.config")

    output:
    path("classyflow_report_light.html"), emit: report_done

    script:
    """
    mkdir -p plots
    concordance_flag=""
    if [ -f feature_concordance.csv ]; then
        concordance_flag="--concordance-csv feature_concordance.csv"
    fi
    profile_flag=""
    if [ -f celltype_profile.yaml ]; then
        profile_flag="--celltype-profile celltype_profile.yaml"
    fi

    generate_light_report.py \
        --input-metrics ${input_metrics_json} \
        --holdout-eval-dir . \
        --fs-dir . \
        \$concordance_flag \
        \$profile_flag \
        --counts-tsv ${aggregated_counts} \
        --plots-dir plots \
        --template-dir ${template_dir} \
        --report-name classyflow_report_light.html \
        --letterhead ${letterhead_file} \
        --version ${params.pipeline_version} \
        --input-dirs "${params.input_dirs.join(',')}" \
        --normalization "${params.override_normalization ?: ''}" \
        --holdout-fraction ${params.holdout_fraction} \
        --min-label-count ${params.minimum_label_count} \
        --exclude-markers "${params.exclude_markers}" \
        --config-file nextflow.config
    """

}


process MERGE_BACK_LARGE_TABLES {
    tag { mergedID }
    input:
    tuple val(mergedID), path(files_to_merge)

    output:
    tuple val(mergedID), path("full_data_${mergedID}.tsv"), emit: merged_tables

    script:
    """
    merge_back_large_tables.py \
        --input_files ${files_to_merge.join(' ')} \
        --output_file full_data_${mergedID}.tsv
    """
}


// -------------------------------------- //


workflow {
    if ( params.help || !params.input_dirs ) {
        helpMessage()
        exit 1
    } else {
        // 1. Merge tab-delimited files
        merged_pkl_ch = MERGE_TAB_DELIMITED_FILES(batchDirs)

        // 2. Optionally fix columns if enabled
        if (params.batch_correct_column_names) {
            // Pass static asset as path, like params.letterhead
            marker_vocab_path = file(params.marker_vocabulary, checkIfExists: true)
            fixed_pkl_ch = COLUMN_FORMAT_AND_FIX(merged_pkl_ch.flatten(), marker_vocab_path)
            input_for_panel_design = fixed_pkl_ch
        } else {
            input_for_panel_design = merged_pkl_ch
        }

        // 3. CHECK_PANEL_DESIGN expects a list of files
        input_for_panel_design_list = input_for_panel_design.collect()
        CHECK_PANEL_DESIGN(input_for_panel_design_list)

        // 4. Create namedBatchtables channel from whichever was used
        namedBatchtables = input_for_panel_design
            .flatten()
            .map { file ->
                def base = file.getBaseName()
                def batchID = base.replaceFirst(/^merged_dataframe_/, '').replaceFirst(/(_fx)?.pkl$/, '')
                tuple(batchID, file)
            }

        // 5. Downstream unchanged
        ADD_EMPTY_MARKER_NOISE(namedBatchtables, CHECK_PANEL_DESIGN.output.paneldesignfile)
        normalized_output = normalization_wf(ADD_EMPTY_MARKER_NOISE.output.modbatchtables)
        normalizedDataFrames = normalized_output.normalized
        labledDataFrames = GENERATE_TRAINING_N_HOLDOUT(normalizedDataFrames.map{ it[1] }.collect())
        feature_selection_results = featureselection_wf(labledDataFrames.training, labledDataFrames.lableFile)
        selectFeatures = feature_selection_results.mas_results
        modeling_results = modelling_wf(labledDataFrames.training, labledDataFrames.holdout, selectFeatures, labledDataFrames.lableFile)
        bestModel = modeling_results.best_model_results
        merged_groups = normalizedDataFrames
            .map { item ->
                def key = item[0].replaceFirst(/-[^-]{5}(_fx)?$/, '')
                [key, item[1]]
            }
            .groupTuple()
        mergeResult = MERGE_BACK_LARGE_TABLES(merged_groups)
        normalizedDataFrames = mergeResult.merged_tables
        prediction_results = PREDICT_ALL_CELLS_XGB(bestModel, normalizedDataFrames)
        prediction_results.predictions
            .flatMap { batchID, files ->
                def fileList = files instanceof List ? files : [files]
                fileList.collect { file ->
                    def sampleID = file.getBaseName().split('\\.')[0]
                    [batchID, sampleID, file]
                }
            }
            .set { prediction_tuples }
        qc_density = QC_DENSITY(prediction_tuples)
        predictions_for_report = qc_density.qc_predictions
        CLASSIFIED_REPORT_PER_SLIDE(predictions_for_report)
        aggregated_counts = CLASSIFIED_REPORT_PER_SLIDE.out.classified_counts
            .collectFile(
                name: 'all_cell_counts.tsv',
                keepHeader: true,
                skip: 1
            )
        // Select which final report(s) to build. Nothing is removed from the
        // existing output; the light report is an additional condensed summary.
        def report_mode = params.report_mode ?: 'both'

        if (report_mode in ['full', 'both']) {
            GENERATE_FINAL_REPORT(
                CHECK_PANEL_DESIGN.output.input_metrics,
                aggregated_counts,
                normalized_output.report,
                feature_selection_results.report,
                modeling_results.report,
                params.html_template,
                params.letterhead,
                params.config_file
            )
        }

        if (report_mode in ['light', 'both']) {
            GENERATE_LIGHT_REPORT(
                CHECK_PANEL_DESIGN.output.input_metrics,
                modeling_results.holdout_evals,
                feature_selection_results.fs_results,
                feature_selection_results.concordance.ifEmpty { [] },
                params.celltype_profile ? file(params.celltype_profile) : [],
                aggregated_counts,
                params.html_template,
                params.letterhead,
                params.config_file
            )
        }
    }
}
