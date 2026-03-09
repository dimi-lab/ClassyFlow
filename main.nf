#!/usr/bin/env nextflow
import groovy.json.JsonOutput

// Using DSL-2
nextflow.enable.dsl=2
println "Active profile: ${workflow.profile}"

// Users can override this in their own config or with --input_dirs
params.output_dir = "${workflow.projectDir}/output"
//Static Assests for beautification
<<<<<<< HEAD
params.letterhead = file("${projectDir}/assets/images/Classyflow_banner_purple.png", checkIfExists: true)
params.html_template = file("${projectDir}/assets/html_templates", checkIfExists: true)
params.rename_yaml = file("${projectDir}/assets/rename_columns.yaml", checkIfExists: true)
=======
params.letterhead = file("${projectDir}/assets/images/Classyflow_banner_1600_220px.png", checkIfExists: true)
params.html_template = file("${projectDir}/assets/html_templates", checkIfExists: true)
>>>>>>> origin/expansion_add_tabnet_model
params.pipeline_version = "1.0"
params.reports_dir = "${params.output_dir}/final_reports"

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
    """
    This pipeline processes batches of images, where the list of input directories is specified in the configuration file (nextflow.config) using the 'input_dirs' parameter. 
    By default, all output will be written to the 'output' directory within the Nextflow working directory, unless an alternative output directory is specified in the configuration file.

    Usage:
      nextflow run main.nf

    Options:
      --input_dirs      List of input directories containing image batches (set in nextflow.config)
      --outdir          Output directory for results (default: ./output, can be overridden in nextflow.config)
      -profile          Chose configuration profile to use [local, slurm, gcp] (default: local)

    For more details, see the
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

<<<<<<< HEAD
process COLUMN_FORMAT_AND_FIX {
    input:
    path tables_pkl
    path rename_yaml

    output:
    path("*_fx.pkl"), emit: batchtables

    script:
    """
    fixup_columns.py \
        --input_table ${tables_pkl} \
        --rename_yaml ${rename_yaml}
    """
}

=======
>>>>>>> origin/expansion_add_tabnet_model
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
        --bitDepth ${params.bit_depth} \
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
<<<<<<< HEAD
        path: "${params.output_dir}/annotation",
        pattern: "*.csv",
=======
        path: "${params.output_dir}/celltype_reports",
        pattern: "*.pdf",
>>>>>>> origin/expansion_add_tabnet_model
        mode: "copy"
    )
    
	input:
	path(norms_pkl_collected)

	output:
    path("holdout_dataframe.pkl"), emit: holdout
    path("training_dataframe.pkl"), emit: training
	path("celltypes.csv"), emit: lableFile
    tuple path("training_split_report.json"), path("cell_count_table.csv"), emit: training_holdout_results
<<<<<<< HEAD
    path("per_batch_label_count.csv"), emit: per_batch_label_count
=======
>>>>>>> origin/expansion_add_tabnet_model

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


process ZIP_PUBLISHED {
    tag "zipping published dir"
    publishDir "${params.output_dir}", pattern: "final_reports.zip", mode: 'copy', overwrite: true

    input:
    val trigger
    path(final_dir)

    output:
    path "final_reports.zip"

    script:
    """
    zip -r final_reports.zip $final_dir
    """
}
// -------------------------------------- //


<<<<<<< HEAD
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
            rename_yaml_path = file(params.rename_yaml, checkIfExists: true)
            fixed_pkl_ch = COLUMN_FORMAT_AND_FIX(merged_pkl_ch.flatten(), rename_yaml_path)
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
        merged_groups.view()
        mergeResult = MERGE_BACK_LARGE_TABLES(merged_groups)
        normalizedDataFrames = mergeResult.merged_tables
        prediction_results = PREDICT_ALL_CELLS_XGB(bestModel, normalizedDataFrames)
        prediction_results.predictions
            .flatMap { batchID, files ->
=======

workflow {
    // Show help message if the user specifies the --help flag at runtime
    // or if any required params are not provided
    if ( params.help || !params.input_dirs ){
        // Invoke the function above which prints the help message
        helpMessage()
        // Exit out and do not run anything else
        exit 1
    } else {
        // Pull channel object `batchDirs` from nextflow env - see top of file.
        MERGE_TAB_DELIMITED_FILES(batchDirs)
        CHECK_PANEL_DESIGN(MERGE_TAB_DELIMITED_FILES.output.batchtables.collect())  

        // Create namedBatchtables channel from batchtables
        namedBatchtables = MERGE_TAB_DELIMITED_FILES.output.batchtables
            .flatten()
            .map { file ->
                def base = file.getBaseName()
                def batchID = base.replaceFirst(/^merged_dataframe_/, '').replaceFirst(/\.pkl$/, '')
                tuple(batchID, file)
            }
        
        ADD_EMPTY_MARKER_NOISE(namedBatchtables, CHECK_PANEL_DESIGN.output.paneldesignfile)
        /*
         * - Subworkflow to handle all Normalization/Standardization Tasks - 
         */ 
        normalized_output = normalization_wf(ADD_EMPTY_MARKER_NOISE.output.modbatchtables)
        normalizedDataFrames = normalized_output.normalized
        
        labledDataFrames = GENERATE_TRAINING_N_HOLDOUT(normalizedDataFrames.map{ it[1] }.collect())
        
        /*
        * - Subworkflow to examine Cell Type Specific interpetability & Feature Selections - 
        */ 
        feature_selection_results = featureselection_wf(labledDataFrames.training, labledDataFrames.lableFile)
        selectFeatures = feature_selection_results.mas_results
        
        /*
        * - Subworkflow to generate models and then check them against the holdout - 
        */ 
        modeling_results = modelling_wf(labledDataFrames.training, labledDataFrames.holdout, selectFeatures, labledDataFrames.lableFile)
        bestModel = modeling_results.best_model_results
        
        // If large file splitting is enabled, merge back the normalized tables
        // Group normalizedDataFrames by removing hyphen and last 5 chars from key
        merged_groups = normalizedDataFrames
        .map { item ->
            def key = item[0].replaceFirst(/-[^-]{5}$/, '')
            [key, item[1]]
        }
        .groupTuple()  

        // merged_groups.view()
        mergeResult = MERGE_BACK_LARGE_TABLES(merged_groups)
        normalizedDataFrames = mergeResult.merged_tables

        // Run the best model on the full input batches/files 
        prediction_results = PREDICT_ALL_CELLS_XGB(bestModel, normalizedDataFrames)

        prediction_results.predictions
            .flatMap { batchID, files -> 
>>>>>>> origin/expansion_add_tabnet_model
                def fileList = files instanceof List ? files : [files]
                fileList.collect { file ->
                    def sampleID = file.getBaseName().split('\\.')[0]
                    [batchID, sampleID, file]
                }
            }
            .set { prediction_tuples }
<<<<<<< HEAD
        qc_density = QC_DENSITY(prediction_tuples)
        predictions_for_report = qc_density.qc_predictions
        CLASSIFIED_REPORT_PER_SLIDE(predictions_for_report)
        aggregated_counts = CLASSIFIED_REPORT_PER_SLIDE.out.classified_counts
            .collectFile(
                name: 'all_cell_counts.tsv',
                keepHeader: true,
                skip: 1
            )
=======


        qc_density = QC_DENSITY(prediction_tuples)
        // Overwrite predictions with QC-augmented files for downstream steps
        predictions_for_report = qc_density.qc_predictions
    
        // Generate a comprehensive HTML report for each prediction file
        CLASSIFIED_REPORT_PER_SLIDE(predictions_for_report)

        aggregated_counts = CLASSIFIED_REPORT_PER_SLIDE.out.classified_counts
            .collectFile(
                name: 'all_cell_counts.tsv',
                keepHeader: true, 
                skip: 1
            )

        // Pass all to reporting including summary JSONs
>>>>>>> origin/expansion_add_tabnet_model
        final_report = GENERATE_FINAL_REPORT(
            CHECK_PANEL_DESIGN.output.input_metrics,
            aggregated_counts,
            normalized_output.report,
<<<<<<< HEAD
            feature_selection_results.report,
=======
            feature_selection_results.report, 
>>>>>>> origin/expansion_add_tabnet_model
            modeling_results.report,
            params.html_template,
            params.letterhead,
            params.config_file
        )
        // ZIP_PUBLISHED(final_report.report_done.map {"done"}, file("${params.output_dir}/final_reports"))
    }
<<<<<<< HEAD
=======
    
>>>>>>> origin/expansion_add_tabnet_model
}