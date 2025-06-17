#!/usr/bin/env nextflow

// Using DSL-2
nextflow.enable.dsl=2

// All of the default parameters are being set in `nextflow.config`
params.input_dirs = [
    "${workflow.projectDir}/data/TMA1990",
    "${workflow.projectDir}/data/TMAS1_4xB2"
]
// Users can override this in their own config or with --input_dirs
<<<<<<< Updated upstream
params.output_dir = "${projectDir}/output"
=======
params.output_dir = "${workflow.projectDir}/output"
>>>>>>> Stashed changes

//Static Assests for beautification
params.letterhead = "${workflow.projectDir}/images/ClassyFlow_Letterhead.PNG"

// Build Input List of Batches
<<<<<<< Updated upstream
Channel.from("${params.input_dirs}", type: 'dir')
			.ifEmpty { error "No files found in ${params.input_dirs}" }
			.set { batchDirs }
=======
Channel.fromList(params.input_dirs)
		.ifEmpty { error "No files found in ${params.input_dirs}" }
		.set { batchDirs }
>>>>>>> Stashed changes
			
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
<<<<<<< Updated upstream
      -profile          Configuration profile to use
=======
      -profile          Chose configuration profile to use [local, slurm, gcp] (default: local)
>>>>>>> Stashed changes

    For more details, see the
    """.stripIndent()
}

// Define a process to merge tab-delimited files and save as pickle
process mergeTabDelimitedFiles {
	input:
    path subdir
    
    output:
    tuple val(batchID), path("merged_dataframe_${batchID}.pkl"), emit: namedBatchtables
    path("merged_dataframe_${batchID}.pkl"), emit: batchtables

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
        --batchID ${batchID}
    """
}

/* 
 * For each input pickle file (merged quantification tables), extract all unique marker/channel names
 * (from columns containing 'Mean'), and generate a presence/absence matrix showing which markers
 * are present in each batch or panel. This matrix is saved as 'panel_design.csv' for downstream
 * comparison of panel designs across
 */
process checkPanelDesign {
	input:
	path(tables_pkl_collected)

    output:
    path 'panel_design.csv', emit: paneldesignfile

    script:
    """
    compare_panel_designs.py ${tables_pkl_collected.join(' ')}
    """
}

/*
 * For each batch's merged quantification table, this process checks the panel design to identify any markers
 * that are missing from the data. For each missing marker, it generates a synthetic column of low-noise values
 * (using sklearn's make_blobs) to fill in the missing features, ensuring all batches have a consistent set of markers.
 * The modified table is saved for downstream normalization and modeling.
 */
 process addEmptyMarkerNoise {
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
process generateTrainingNHoldout{
	publishDir(
        path: "${params.output_dir}/celltype_reports",
        pattern: "*.pdf",
        mode: "copy"
    )
    
	input:
	path(norms_pkl_collected)

	output:
    path("holdout_dataframe.pkl"), emit: holdout
    path("training_dataframe.pkl"), emit: training
	path("celltypes.csv"), emit: lableFile
	path("annotation_report.pdf")

    script:
    """
    split_annotations_for_training.py \
        --classColumn ${params.classifed_column_name} \
        --holdoutFraction ${params.holdout_fraction} \
        --cellTypeNegative "${params.filter_out_junk_celltype_labels}" \
        --minimunHoldoutThreshold ${params.minimum_label_count} \
        --pickle_files "${norms_pkl_collected}" \
        --letterhead "${params.letterhead}"
    """

}

// Run model on everything make results
process predictAllCells_xgb{
	publishDir(
        path: "${params.output_dir}/celltypes",
        pattern: "*_PRED.tsv",
        mode: "copy"
    )
    
	input:
	tuple val(model_name), path(model_path)
	tuple val(batchID), path(pickleTable)
	
	output:
	path("*.tsv")
	
	script:
    """
    predict_celltypes.py \
        --classColumn ${params.predict_class_column} \
        --leEncoderFile ${params.predict_le_encoder_file} \
        --batchID ${batchID} \
        --infile ${pickleTable} \
        --modelfile ${model_path} \
        --columnsToExport "${params.predict_columns_to_export}" \
        --cpu_jobs ${params.predict_cpu_jobs}
    """
}
// -------------------------------------- //



workflow {
    // Show help message if the user specifies the --help flag at runtime
    // or if any required params are not provided
    if ( params.help || params.input_dir == false ){
        // Invoke the function above which prints the help message
        helpMessage()
        // Exit out and do not run anything else
        exit 1
    } else {

		// Pull channel object `batchDirs` from nextflow env - see top of file.
    	mergeTabDelimitedFiles(batchDirs)
    
    	checkPanelDesign(mergeTabDelimitedFiles.output.batchtables.collect())  
    	
    	//modify the pickle files to account for missing features...
    	addEmptyMarkerNoise(mergeTabDelimitedFiles.output.namedBatchtables, checkPanelDesign.output.paneldesignfile)
    	   
    	/*
    	 * - Subworkflow to handle all Normalization/Standardization Tasks - 
    	 */ 
    	normalizedDataFrames = normalization_wf(addEmptyMarkerNoise.output.modbatchtables)
    	
    	labledDataFrames = generateTrainingNHoldout(normalizedDataFrames.map{ it[1] }.collect())
    	
		/*
    	 * - Subworkflow to examine Cell Type Specific interpetability & Feature Selections - 
    	 */ 
		selectFeatures = featureselection_wf(labledDataFrames.training, labledDataFrames.lableFile)
    	
    	/*
    	 * - Subworkflow to generate models and then check them against the holdout - 
    	 */ 
		bestModel = modelling_wf(labledDataFrames.training, labledDataFrames.holdout, selectFeatures)
		
		
    	// Run the best model on the full input batches/files 
    	predictAllCells_xgb(bestModel, normalizedDataFrames)
    	
    }
    
}
