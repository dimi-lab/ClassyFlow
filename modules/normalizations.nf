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
    final_ch = gmm_gated.norm_df

    //norm_outputs = norm_results.norm_results.map { batchID, file -> file }.collect()
    norm_outputs = norm_results.norm_results.collect()
    gmm_outputs = gmm_gated.gmm_json.collect()
    norm_report = GENERATE_NORM_REPORT(norm_outputs, gmm_outputs, "${params.html_template}")

    emit:
    normalized = final_ch
    report = norm_report.norm_html
}

