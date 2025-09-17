// Produce Batch based normalization - boxcox
process NORMALIZATION {
    tag { batchID }

    publishDir "${params.output_dir}/final_reports/pages", pattern: "*.html", mode: 'copy'
    
    input:
    tuple val(batchID), path(pickleTable), val(method)
    
    output:
    tuple val(batchID), path("*_transformed_${batchID}.tsv"), emit: norm_df
    tuple val(batchID), path ("*_results_${batchID}.json"), path("*_all_plots_${batchID}.html"), optional: true, emit: norm_results
    
    script:
    """
    quant_transformer.py \
        --method ${method} \
        --pickleTable ${pickleTable} \
        --batchID ${batchID} \
        --quantileSplit ${params.quantile_split} \
        --target-feature ${params.plot_target_feature_suffix}
    """
}


// Look at all of the normalizations within a batch and attempt to idendity the best approach
process IDENTIFY_BEST{
    publishDir "${params.output_dir}/norm_reports", pattern: "*.html", mode: 'copy'
    publishDir "${params.output_dir}/norm_reports", pattern: "*.csv", mode: 'copy'
    publishDir "${params.output_dir}/norm_reports", pattern: "*.png", mode: 'copy'

    input:
    val(batchIDs)
    path(files)
    
    output:
    path("*.csv")
    path("*.png")

    script:
    """
    characterize_normalization.py --batch-ids ${batchIDs.join(',')} --target-features ${params.plot_target_feature_suffix}
    """
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
            ${params.override_normalization == 'none' ? '--no-normalization' : ''}
            --output-file normalization_report.html \
            --template-dir ${html_template}
    """
}

// -------------------------------------- //


workflow normalization_wf {
    take:
    batchPickleTable

    main:
    // Initialize empty channels for conditional results
    norm_results = Channel.empty()
    norm_outputs_list = []
    
    // Step 1: Choose normalization method based on override parameter
    if (params.override_normalization == "all") {
        // Run all normalization methods and compare results
        all_methods = Channel.of("boxcox", "quantile", "minmax", "log", "none")

        combined_ch = batchPickleTable
                        .combine(all_methods)
        
        norm_results = NORMALIZATION(combined_ch)
        
        // Mix all normalization results and group them for comparison
        batch_ch = norm_results.norm_df.map { batchID, file -> batchID }.collect()
        files_ch = norm_results.norm_df.map { batchID, file -> file }.collect()

        mxchannels = batch_ch.combine(files_ch)

        // Identify the best normalization approach
        best_selection = IDENTIFY_BEST(batch_ch, files_ch)

        return

    } else if (params.override_normalization in ["boxcox", "quantile", "minmax", "log", "none"]) {
        // Use BoxCox normalization
        norm_results = NORMALIZATION(batchPickleTable, params.override_normalization)
        best_ch = norm_results.norm_df

        norm_outputs_list.add(norm_results.norm_results.map { it -> it[1..-1] }.ifEmpty([]))

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
}

