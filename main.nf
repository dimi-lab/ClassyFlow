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
    path 'merged_dataframe.pkl'

    script:
    template 'merge_files.py'
}


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
		// Pull channel object `batchDirs` from env - see top of file.
    	mergeTabDelimitedFiles(batchDirs)
    
    }
    
    
    
}
