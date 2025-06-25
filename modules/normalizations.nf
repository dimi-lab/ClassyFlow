// Produce Batch based normalization - boxcox
process BOXCOX {
	tag { batchID }

	publishDir(
        path: "${params.output_dir}/normalization",
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
process QUANTILE {
	tag { batchID }

    publishDir(
        path: "${params.output_dir}/normalization",
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
process MINMAX {
	tag { batchID }
	publishDir(
        path: "${params.output_dir}/normalization",
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

process LOGSCALE {
	tag { batchID }
	publishDir(
        path: "${params.output_dir}/normalization",
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
        path: "${params.output_dir}/normalization",
        pattern: "*.pdf",
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

process GMM_GATING {
    tag { batchID }
    publishDir(
        path: "${params.output_dir}/normalization",
        pattern: "*.html",
        mode: "copy"
    )
    input:
    tuple val(batchID), path(norm_table)

    output:
    tuple val(batchID), path("gmm_gated_${batchID}.tsv"), emit: norm_df
    path("gmm_gated_${batchID}.html")
    
    script:
    """
    gmm_gating.py \
        --input ${norm_table} \
        --output gmm_gated_${batchID}.tsv \
        --html_report gmm_gated_${batchID}.html
    """
}

// -------------------------------------- //


workflow normalization_wf {
    take:
    batchPickleTable

    main:
	def best_ch
    if (params.override_normalization == "boxcox") {
        def bc = BOXCOX(batchPickleTable)
        best_ch = bc.norm_df
    }
    else if (params.override_normalization == "quantile") {
        def qt = QUANTILE(batchPickleTable)
        best_ch = qt.norm_df
    }
    else if (params.override_normalization == "minmax") {
        def mm = MINMAX(batchPickleTable)
        best_ch = mm.norm_df
    }
    else if (params.override_normalization == "logscale") {
        def lg = LOGSCALE(batchPickleTable)
        best_ch = lg.norm_df
    }
    else {
        def bc = BOXCOX(batchPickleTable).norm_df
        def qt = QUANTILE(batchPickleTable).norm_df
        def mm = MINMAX(batchPickleTable).norm_df
        def lg = LOGSCALE(batchPickleTable).norm_df

        def mxchannels = batchPickleTable.mix(bc, qt, mm, lg).groupTuple()
        mxchannels.dump(tag: 'debug_normalization_channels', pretty: true)

        def best = IDENTIFY_BEST(mxchannels)
        best_ch = best.norm_df
    }

    // Insert GMM gating after normalization
    def gmm_gated = GMM_GATING(best_ch)
    best_ch = gmm_gated.norm_df

    if (params.run_get_leiden_clusters) {
        def leiden_augmented = AUGMENT_WITH_LEIDEN_CLUSTERS(best_ch)
        best_ch = leiden_augmented.norm_df
    }

    // ## future add flag column for bin density ##

    // ## add column for sig sum ##

    emit:
    normalized = best_ch
}

