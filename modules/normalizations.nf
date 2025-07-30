// Produce Batch based normalization - boxcox
process BOXCOX {
    tag { batchID }
    
    input:
    tuple val(batchID), path(pickleTable)
    
    output:
    tuple val(batchID), path("boxcox_transformed_${batchID}.tsv"), emit: norm_df
    tuple val(batchID), path ("boxcox_results_${batchID}.json"), path("boxcox_*.png"), emit: boxcox_results
    
    script:
    """
    boxcox_transformer.py \
        --pickleTable ${pickleTable} \
        --batchID ${batchID} \
        --quantType ${params.qupath_object_type} \
        --nucMark ${params.nucleus_marker} \
        --plotFraction ${params.plot_fraction}
    """
}
    
    
// Produce Batch based normalization - quantile
process QUANTILE {
    tag { batchID }

    input:
    tuple val(batchID), path(pickleTable)

    output:
    tuple val(batchID), path("quantile_transformed_${batchID}.tsv"), emit: norm_df
    tuple val(batchID), path ("quantile_results_${batchID}.json"), path("quantile_*.png"), emit: quantile_results

    script:
    """
    quantile_transformer.py \
        --pickleTable ${pickleTable} \
        --batchID ${batchID} \
        --quantType ${params.qupath_object_type} \
        --nucMark ${params.nucleus_marker} \
        --plotFraction ${params.plot_fraction} \
        --quantileSplit ${params.quantile_split}
    """
}


// Produce Batch based normalization - min/max scaling
process MINMAX {
    tag { batchID }
    
    input:
    tuple val(batchID), path(pickleTable)
    
    output:
    tuple val(batchID), path("minmax_transformed_${batchID}.tsv"), emit: norm_df
    tuple val(batchID), path ("minmax_results_${batchID}.json"), path("minmax_*.png"), emit: minmax_results
    
    script:
    """
    minmax_transformer.py \
        --pickleTable ${pickleTable} \
        --batchID ${batchID} \
        --quantType ${params.qupath_object_type} \
        --nucMark ${params.nucleus_marker} \
        --plotFraction ${params.plot_fraction}
    """

}

process LOGSCALE {
    tag { batchID }
    
    input:
    tuple val(batchID), path(pickleTable)
    
    output:
    tuple val(batchID), path("log_transformed_${batchID}.tsv"), emit: norm_df
    tuple val(batchID), path ("log_results_${batchID}.json"), path("log_*.png"), emit: log_results
    
    script:
    """
    log_transformer.py \
        --pickleTable ${pickleTable} \
        --batchID ${batchID} \
        --quantType ${params.qupath_object_type} \
        --nucMark ${params.nucleus_marker} \
        --plotFraction ${params.plot_fraction}
    """
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
        pattern: '*.{html,png}',
        mode: 'copy'
    )

    input:
    tuple val(batchID), path(norms_pkl)

    output:
    tuple val(batchID), path("scimap_clusters_${batchID}.tsv"), emit: norm_df
    path("*.png"), optional: true
    path("*.html")

    script:
    """
    scimap_clustering.py \
        --input_tsv ${norms_pkl} \
        --roi_name ${batchID} \
        --resolution ${params.scimap_resolution} \
        --label_fraction ${params.scimap_label_fraction} \
        --qupath_object_type ${params.qupath_object_type} \
        --classifed_column_name ${params.classifed_column_name}
    """
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
    // Declare variables for normalization results and channels
    def best_ch
    def bc
    def lg
    def qt
    def mm
    def mxchannels

    // Step 1: Choose normalization method based on override parameter
    if (params.override_normalization == "boxcox") {
        // Use BoxCox normalization
        bc = BOXCOX(batchPickleTable)
        best_ch = bc.norm_df
    }
    else if (params.override_normalization == "quantile") {
        // Use Quantile normalization
        qt = QUANTILE(batchPickleTable)
        best_ch = qt.norm_df
    }
    else if (params.override_normalization == "minmax") {
        // Use MinMax normalization
        mm = MINMAX(batchPickleTable)
        best_ch = mm.norm_df
    }
    else if (params.override_normalization == "logscale") {
        // Use LogScale normalization
        lg = LOGSCALE(batchPickleTable)
        best_ch = lg.norm_df
    }
    else {
        // Run all normalization methods and group results for comparison
        bc = BOXCOX(batchPickleTable).norm_df
        qt = QUANTILE(batchPickleTable).norm_df
        mm = MINMAX(batchPickleTable).norm_df
        lg = LOGSCALE(batchPickleTable).norm_df

        // Mix all normalization results and group them for best selection
        mxchannels = batchPickleTable.mix(bc, qt, mm, lg).groupTuple()
        mxchannels.dump(tag: 'debug_normalization_channels', pretty: true)

        // Identify the best normalization approach
        def best = IDENTIFY_BEST(mxchannels)
    }

    // Step 2: Apply GMM gating to the normalized data
    def gmm_gated = GMM_GATING(best_ch)
    best_ch = gmm_gated.norm_df

    // Step 3: Optionally augment with Leiden clusters if enabled
    if (params.run_get_leiden_clusters) {
        def leiden_augmented = AUGMENT_WITH_LEIDEN_CLUSTERS(best_ch)
        best_ch = leiden_augmented.norm_df
    }

    // Step 4: Emit final results and normalization outputs
    emit:
    normalized = best_ch
    boxcox_results = bc ? bc.boxcox_results : Channel.empty()
    quantile_results = qt ? qt.quantile_results : Channel.empty()
    minmax_results = mm ? mm.minmax_results : Channel.empty()
    log_results = lg ? lg.log_results : Channel.empty()
}

