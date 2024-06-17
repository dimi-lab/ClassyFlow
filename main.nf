#!/usr/bin/env nextflow

// Using DSL-2
nextflow.enable.dsl=2

// All of the default parameters are being set in `nextflow.config`
params.input_dir = "${projectDir}/data"
params.output_dir = "${projectDir}/output"

// Build Input List of Batches
Channel.fromPath("${params.input_dir}/*/", type: 'dir')
			.ifEmpty { error "No subdirectories found in ${params.input_dir}" }
			.set { batchDirs }
			
// Import sub-workflows
include { normalization_wf } from './modules/normalizations'




// -------------------------------------- //
// Function which prints help message text
def helpMessage() {
    log.info"""
Usage:

nextflow run main.nf <ARGUMENTS>

Required Arguments:

  Input Data:
  --input_dir        Folder containing subfolders of QuPath's Quantification Exported Measurements,
                        each dir containing Quant files belonging to a common batch of images.
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
    tuple val(batchID), path("merged_dataframe_${batchID}.pkl"), emit: modbatchtables

    script:
    template 'add_empty_marker_noise.py'
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
    	
    	checkPanelDesign(mergeTabDelimitedFiles.output.batchtables.collect())  // "merged_dataframe_SET01.pkl    merged_dataframe_SET02.pkl    merged_dataframe_SET03.pkl"
    	
    	//modify the pickle files to account for missing features...
    	addEmptyMarkerNoise(mergeTabDelimitedFiles.output.namedBatchtables, checkPanelDesign.output.paneldesignfile)
    	   
    	normalization_wf(addEmptyMarkerNoise.output.modbatchtables)
    	
    	
    }
    
    
    
}
