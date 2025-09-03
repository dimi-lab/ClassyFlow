// Produce Batch based normalization - boxcox
process BOXCOX {
    tag { batchID }
    publishDir "${params.output_dir}/final_reports/pages", pattern: "boxcox_*.html", mode: 'copy'
    publishDir "${params.output_dir}/temp/", pattern: "boxcox_*.json", mode: 'copy'
    
    input:
    tuple val(batchID), path(pickleTable)
    
    output:
    tuple val(batchID), path("boxcox_transformed_${batchID}.tsv"), emit: norm_df
    tuple val(batchID), path ("boxcox_results_${batchID}.json"), path("boxcox_all_plots_${batchID}.html"), emit: boxcox_results
    
    script:
    """
    boxcox_transformer.py \
        --pickleTable ${pickleTable} \
        --batchID ${batchID} \
        --quantType ${params.qupath_object_type} \
        --nucMark ${params.nucleus_marker} \
        --plotFraction ${params.plot_fraction} \
        --target-feature ${params.plot_target_feature_suffix}
    """
}
    
    
// Produce Batch based normalization - quantile
process QUANTILE {
    tag { batchID }

    publishDir "${params.output_dir}/final_reports/pages", pattern: "quantile_*.html", mode: 'copy'
    publishDir "${params.output_dir}/temp/", pattern: "quantile_*.json", mode: 'copy'

    input:
    tuple val(batchID), path(pickleTable)

    output:
    tuple val(batchID), path("quantile_transformed_${batchID}.tsv"), emit: norm_df
    tuple val(batchID), path ("quantile_results_${batchID}.json"), path("quantile_all_plots_${batchID}.html"), emit: quantile_results

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

    publishDir "${params.output_dir}/final_reports/pages", pattern: "minmax_*.html", mode: 'copy'
    publishDir "${params.output_dir}/temp/", pattern: "minmax_*.json", mode: 'copy'
    
    input:
    tuple val(batchID), path(pickleTable)
    
    output:
    tuple val(batchID), path("minmax_transformed_${batchID}.tsv"), emit: norm_df
    tuple val(batchID), path ("minmax_results_${batchID}.json"), path("minmax_all_plots_${batchID}.html"), emit: minmax_results
    
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

    publishDir "${params.output_dir}/final_reports/pages", pattern: "log_*.html", mode: 'copy'
    publishDir "${params.output_dir}/temp/", pattern: "log_*.json", mode: 'copy'
    
    input:
    tuple val(batchID), path(pickleTable)
    
    output:
    tuple val(batchID), path("log_transformed_${batchID}.tsv"), emit: norm_df
    tuple val(batchID), path ("log_results_${batchID}.json"), path("log_all_plots_${batchID}.html"), emit: log_results
    
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
    tuple val(batchID), path("scimap_extended_${batchID}.tsv"), emit: norm_df
    path("*.png"), optional: true
    path("*.html")

    script:
    """
    scimap_clustering.py \
        --input_tsv ${norms_pkl} \
        --roi_name ${batchID} \
        --resolution ${params.scimap_resolution} \
        --label_fraction ${params.scimap_label_fraction} \
        --perc_top_features ${params.scimap_top_feature_prec} \
        --qupath_object_type ${params.qupath_object_type} \
        --classifed_column_name ${params.classifed_column_name}
    """
}

process GMM_GATING {
    tag { batchID }
    publishDir(
        path: "${params.output_dir}/final_reports/pages",
        pattern: "*.html",
        mode: "copy"
    )
    input:
    tuple val(batchID), path(norm_table)
    val(target_feature)

    output:
    tuple val(batchID), path("gmm_gated_${batchID}.tsv"), emit: norm_df
    tuple val(batchID), path("gmm_gated_${batchID}.html"), emit: gmm_html
    
    script:
    """
    gmm_gating.py \
        --input ${norm_table} \
        --output gmm_gated_${batchID}.tsv \
        --html_report gmm_gated_${batchID}.html \
        --target-feature "${target_feature}" \
        --batch-name "${batchID}"

    """
}

process GENERATE_NORM_REPORT {
    //publishDir "${params.output_dir}/final_reports/", pattern: "normalization_report.html", mode: 'copy'

    input:
    path(norm_files)
    tuple val(batchIDs), path(html_files)
    path(html_template)

    output:
    path("normalization_report.html"), emit: norm_html
    path("normalization_summary.json"), emit: norm_summary

    script:
    """
    generate_normalization_report.py \
            --output-file normalization_report.html \
            --template-dir $html_template
    """
}

// -------------------------------------- //


workflow normalization_wf {
    take:
    batchPickleTable

    main:
    // Initialize empty channels for conditional results
    bc_results = Channel.empty()
    qt_results = Channel.empty()
    mm_results = Channel.empty()
    lg_results = Channel.empty()
    norm_outputs_list = []
    
    // Step 1: Choose normalization method based on override parameter
    if (params.override_normalization == "boxcox") {
        // Use BoxCox normalization
        bc_results = BOXCOX(batchPickleTable)
        best_ch = bc_results.norm_df

        norm_outputs_list.add(bc_results.boxcox_results.map { it -> it[1..-1] }.ifEmpty([]))
    }
    else if (params.override_normalization == "quantile") {
        // Use Quantile normalization
        qt_results = QUANTILE(batchPickleTable)
        best_ch = qt_results.norm_df

        norm_outputs_list.add(qt_results.quantile_results.map { it -> it[1..-1] }.ifEmpty([]))
    }
    else if (params.override_normalization == "minmax") {
        // Use MinMax normalization
        mm_results = MINMAX(batchPickleTable)
        best_ch = mm_results.norm_df

        norm_outputs_list.add(mm_results.minmax_results.map { it -> it[1..-1] }.ifEmpty([]))
    }
    else if (params.override_normalization == "logscale") {
        // Use LogScale normalization
        lg_results = LOGSCALE(batchPickleTable)
        best_ch = lg_results.norm_df

        norm_outputs_list.add(lg_results.log_results.map { it -> it[1..-1] }.ifEmpty([]))
    }
    else {
        // Run all normalization methods and compare results
        bc_results = BOXCOX(batchPickleTable)
        qt_results = QUANTILE(batchPickleTable)
        mm_results = MINMAX(batchPickleTable)
        lg_results = LOGSCALE(batchPickleTable)

        // Mix all normalization results and group them for comparison
        mxchannels = batchPickleTable
            .mix(bc_results.norm_df,qt_results.norm_df, mm_results.norm_df, lg_results.norm_df)
            .groupTuple()
        
        mxchannels.dump(tag: 'debug_normalization_channels', pretty: true)

        // Identify the best normalization approach
        best_selection = IDENTIFY_BEST(mxchannels)
        best_ch = best_selection.norm_df

        // All other values (including null/empty) - add all outputs
        norm_outputs_list.add(bc_results.boxcox_results.map { it -> it[1..-1] }.ifEmpty([]))
        norm_outputs_list.add(qt_results.quantile_results.map { it -> it[1..-1] }.ifEmpty([]))
        norm_outputs_list.add(mm_results.minmax_results.map { it -> it[1..-1] }.ifEmpty([]))
        norm_outputs_list.add(lg_results.log_results.map { it -> it[1..-1] }.ifEmpty([]))
    }

    // Step 2: Apply GMM gating to the normalized data
    gmm_gated = GMM_GATING(best_ch, params.plot_target_feature_suffix)
    gated_ch = gmm_gated.norm_df
    gated_html = gmm_gated.gmm_html.collect(flat: false)
                        .map { it.transpose() }

    // Step 3: Optionally augment with Leiden clusters if enabled
    if (params.run_get_leiden_clusters) {
        leiden_augmented = AUGMENT_WITH_LEIDEN_CLUSTERS(gated_ch)
        final_ch = leiden_augmented.norm_df
    } else {
        final_ch = gated_ch
    }

    // Collect normalization outputs for reporting
    // Only mix channels that actually exist
    if (norm_outputs_list.size() > 0) {
        norm_outputs = Channel.empty()
            .mix(*norm_outputs_list)
            .flatten()
            .collect()
    } else {
        norm_outputs = Channel.empty().collect()
    }

    norm_report = GENERATE_NORM_REPORT(norm_outputs, gated_html, params.html_template)

    emit:
    normalized = final_ch
    report = norm_report.norm_html
    norm_summary = norm_report.norm_summary
}

