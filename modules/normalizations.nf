// Produce Batch based normalization - boxcox
process boxcox {
	tag { batchID }
    label 'normalization_parallel'

	publishDir(
        path: "${params.output_dir}/normalization_reports",
        pattern: "*.pdf",
        mode: "copy"
    )
	
	input:
	tuple val(batchID), path(pickleTable)
	
	output:
	tuple val(batchID), path("boxcox_transformed_${batchID}.tsv"), emit: norm_df
	path("boxcox_report_${batchID}.pdf")
	
	script:
	"""
	boxcox_transformer.py \
		--pickleTable ${pickleTable} \
		--batchID ${batchID} \
		--quantType ${params.qupath_object_type} \
		--nucMark ${params.nucleus_marker} \
		--plotFraction ${params.plot_fraction} \
		--letterhead ${params.letterhead}
	"""
}
    
    
// Produce Batch based normalization - quantile
process quantile {
	tag { batchID }

    publishDir(
        path: "${params.output_dir}/normalization_reports",
        pattern: "*.pdf",
        mode: "copy"
    )

    input:
    tuple val(batchID), path(pickleTable)

    output:
    tuple val(batchID), path("quantile_transformed_${batchID}.tsv"), emit: norm_df
    path("quantile_report_${batchID}.pdf")

    script:
    """
    quantile_transformer.py \
        --pickleTable ${pickleTable} \
        --batchID ${batchID} \
        --quantType ${params.qupath_object_type} \
        --nucMark ${params.nucleus_marker} \
        --plotFraction ${params.plot_fraction} \
        --quantileSplit ${params.quantile_split} \
        --letterhead ${params.letterhead}
    """
}

// Produce Batch based normalization - min/max scaling
process minmax {
	tag { batchID }
	executor "slurm"
    memory "50G"
    queue "cpu-short"
    time "24:00:00"
    
	publishDir(
        path: "${params.output_dir}/normalization_reports",
        pattern: "*.pdf",
        mode: "copy"
    )
	
	input:
	tuple val(batchID), path(pickleTable)
	
	output:
	tuple val(batchID), path("minmax_transformed_${batchID}.tsv"), emit: norm_df
	path("minmax_report_${batchID}.pdf")
	
	script:
	template 'minmax_transformer.py'

}

process logscale {
	tag { batchID }
	executor "slurm"
    memory "50G"
    queue "cpu-short"
    time "24:00:00"
    
	publishDir(
        path: "${params.output_dir}/normalization_reports",
        pattern: "*.pdf",
        mode: "copy"
    )
	
	input:
	tuple val(batchID), path(pickleTable)
	
	output:
	tuple val(batchID), path("log_transformed_${batchID}.tsv"), emit: norm_df
	path("log_report_${batchID}.pdf")
	
	script:
	template 'log_transformer.py'

}


// Look at all of the normalizations within a batch and attempt to idendity the best approach
process IDENTIFY_BEST{
	publishDir(
        path: "${params.output_dir}/normalization_reports",
        pattern: "*.pdf",
        mode: "copy"
    )

	publishDir(
        path: "${params.output_dir}/normalization_files",
        pattern: "normalized_*.tsv",
        mode: "copy"
    )
    
	input:
	tuple val(batchID), path(all_possible_tables)
	
	output:
	tuple val(batchID), path("normalized_${batchID}.pkl"), emit: norm_df
	path("multinormalize_report_${batchID}.pdf")
	path("normalized_*_${batchID}.tsv")

	script:
	template 'characterize_normalization.py'
}


process AUGMENT_WITH_LEIDEN_CLUSTERS{
	publishDir(
        path: "${params.output_dir}/clusters",
        pattern: "*.pdf",
        mode: "copy"
    )
    
	input:
	tuple val(batchID), path(norms_pkl)

	output:
    path("x_dataframe.pkl"), emit: norm_df
	path("*report.pdf")

    script:
    template 'scimap_clustering.py'
}
// -------------------------------------- //


workflow normalization_wf {
    take:
    batchPickleTable

    main:
	def best_ch
    if (params.override_normalization == "boxcox") {
        def bc = boxcox(batchPickleTable)
        best_ch = bc.norm_df
    }
    else if (params.override_normalization == "quantile") {
        def qt = quantile(batchPickleTable)
        best_ch = qt.norm_df
    }
    else if (params.override_normalization == "minmax") {
        def mm = minmax(batchPickleTable)
        best_ch = mm.norm_df
    }
    else if (params.override_normalization == "logscale") {
        def lg = logscale(batchPickleTable)
        best_ch = lg.norm_df
    }
    else {
        def bc = boxcox(batchPickleTable).norm_df
        def qt = quantile(batchPickleTable).norm_df
        def mm = minmax(batchPickleTable).norm_df
        def lg = logscale(batchPickleTable).norm_df

        def mxchannels = batchPickleTable.mix(bc, qt, mm, lg).groupTuple()
        mxchannels.dump(tag: 'debug_normalization_channels', pretty: true)

        def best = IDENTIFY_BEST(mxchannels)
        best_ch = best.norm_df
    }

    if (params.run_get_leiden_clusters) {
        def leiden_augmented = AUGMENT_WITH_LEIDEN_CLUSTERS(best_ch)
        best_ch = leiden_augmented.norm_df
    }

    emit:
    normalized = best_ch
}

