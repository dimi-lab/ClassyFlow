#!/usr/bin/env nextflow

// Using DSL-2
nextflow.enable.dsl=2

// All of the default parameters are being set in `nextflow.config`
params.input_dirs = [
    "${workflow.projectDir}/data/TMA1990",
    "${workflow.projectDir}/data/TMAS1_4xB2"
]
// Users can override this in their own config or with --input_dirs
params.output_dir = "${projectDir}/output"

//Static Assests for beautification
params.letterhead = "${projectDir}/images/ClassyFlow_Letterhead.PNG"

// Build Input List of Batches
Channel.from("${params.input_dirs}", type: 'dir')
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
      -profile          Configuration profile to use

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
    exMarks = "${params.exclude_markers}"
    batchID = subdir.baseName
    template 'merge_files.py'
}

// Identify 
process checkPanelDesign {
	input:
	path(tables_pkl_collected)

    output:
    path 'panel_design.csv', emit: paneldesignfile

    script:
    template 'compare_panel_designs.py'
}

//Add back empty Markers, low noise (16-bit or 8-bit)
process addEmptyMarkerNoise {
	input:
	tuple val(batchID), path(pickleTable)
	path designTable

    output:
    tuple val(batchID), path("merged_dataframe_${batchID}_mod.pkl"), emit: modbatchtables

    script:
    template 'add_empty_marker_noise.py'
}
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
    template 'split_annotations_for_training.py'

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
    template 'predict_celltypes.py'

}
// -------------------------------------- //




// Main workflow
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
