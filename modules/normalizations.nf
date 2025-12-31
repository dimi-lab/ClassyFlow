// Produce Batch based normalization - boxcox
process NORMALIZATION {
    tag { batchID }
    publishDir "${params.output_dir}/final_reports/pages", pattern: "*.html", mode: 'copy'
    
    input:
    tuple val(batchID), path(pickleTable)
    
    output:
    tuple val(batchID), path("*_transformed_${batchID}.tsv"), emit: norm_df
    path ("*_results_${batchID}.json"), emit: norm_results
    path("*_all_plots_${batchID}.html")
    
    script:
    """
    pick_a_transformer.py \
        --method ${params.override_normalization} \
        --pickleTable ${pickleTable} \
        --batchID ${batchID} \
        --quantileSplit ${params.quantile_split} \
        --target-feature "${params.plot_target_feature_suffix}"
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

    output:
    tuple val(batchID), path("gmm_gated_${batchID}.tsv"), emit: norm_df
    tuple val(batchID), path("gmm_gated_${batchID}.html"), emit: gmm_html
    path("gmm_results_${batchID}.json"), emit: gmm_json
    
    script:
    """
    gmm_gating.py \
        --input ${norm_table} \
        --output gmm_gated_${batchID}.tsv \
        --html_report gmm_gated_${batchID}.html \
        --target-feature "${params.plot_target_feature_suffix}" \
        --batch-name "${batchID}"

    """
}

process GENERATE_NORM_REPORT {
    //publishDir "${params.output_dir}/final_reports/", pattern: "normalization_report.html", mode: 'copy'

    input:
    path(norm_files)
    path(gmm_files)
    path(html_template)

    output:
    path("normalization_report.html"), emit: norm_html

    script:
    """
    echo "Template inside process ------> $html_template"
    generate_normalization_report.py \
            --output-file normalization_report.html \
            --template-dir ${html_template} ${params.override_normalization == 'none' ? '--no-normalization' : ''}
    """
}

// -------------------------------------- //


workflow normalization_wf {
    take:
    batchPickleTable

    main:    
    // Step 1: Choose normalization method based on override parameter
    //if (params.override_normalization in ["boxcox", "quantile", "minmax", "log", "none"]) 

    norm_results = NORMALIZATION(batchPickleTable)
    //norm_results.norm_df.view()
    // Step 2: Apply GMM gating to the normalized data
    gmm_gated = GMM_GATING(norm_results.norm_df)
    gated_ch = gmm_gated.norm_df

    // Step 3: Optionally augment with Leiden clusters if enabled
    if (params.run_get_leiden_clusters) {
        leiden_augmented = AUGMENT_WITH_LEIDEN_CLUSTERS(gated_ch)
        final_ch = leiden_augmented.norm_df
    } else {
        final_ch = gated_ch
    }

    //norm_outputs = norm_results.norm_results.map { batchID, file -> file }.collect()
    norm_outputs = norm_results.norm_results.collect()
    gmm_outputs = gmm_gated.gmm_json.collect()
    norm_report = GENERATE_NORM_REPORT(norm_outputs, gmm_outputs, "${params.html_template}")

    emit:
    normalized = final_ch
    report = norm_report.norm_html
}

