//Split training data and binarize by class label
process TOP_LABEL_SPLIT {
	input:
	path(trainingDataframe)
	val(celltype)

	output:
	tuple val(celltype), path("binary_df*"), optional: true
	
	script:
    """
    split_cell_type_labels.py \
        --trainingDataframe ${trainingDataframe} \
        --celltype "${celltype}" \
        --classColumn ${params.classifed_column_name} \
        --varThreshold 0.01 \
        --mim_class_label_threshold ${params.minimum_label_count} \
        --ifSubsetData True \
        --subSet_n 3000
    """
}

process SEARCH_FOR_ALPHAS {
	
	input:
	tuple val(celltype), path(binary_dataframe), val(logspace_chunk)
    
    output:
    tuple val(celltype), path("alphas_params*"), emit: alphas
    
    script:
    """
    search_all_alphas.py \
        --logspace_chunk "${logspace_chunk}" \
        --binary_dataframe "${binary_dataframe}" \
        --celltype "${celltype}" \
        --n_folds 10
    """
}

process MERGE_ALPHAS_SEARCH_CSV_FILES {
    input:
    tuple val(celltype), path(csv_files)

    output:
    tuple val(celltype), path("merged_alphas_*.csv")

    script:
    // Remove spaces from the original string
    cleanedString = celltype.replaceAll(/[\s\/]+/, '')
    cleanedString = cleanedString.replaceAll(/\|/, '_')
    """
    # Concatenate all CSV files, sort by mean_test_score
    head -n 1 ${csv_files[0]} > merged_alphas_${cleanedString}.csv
    tail -n +2 -q ${csv_files.join(' ')} | sort -t, -k1,1nr >> merged_alphas_${cleanedString}.csv
    """
}

process SELECT_BEST_ALPHA {
    input:
    tuple val(celltype), path(merged_csv)

    output:
    tuple val(celltype), stdout

    shell:
    """
    # Extract the best_alpha where mean_test_score is the highest
    best_alpha=\$(awk -F, 'NR==2 {best=\$3} END {print best}' ${merged_csv})
    echo -n \$best_alpha
    """
}


process RUN_ALL_RFE{

	input:
	tuple val(celltype), path(binary_dataframe), val(best_alpha), val(n_feats)
	    
    output:
    tuple val(celltype), path("rfe_scores*"), emit: feature_scores
    
    script:
    """
    calculate_RFE.py \
        --binary_dataframe ${binary_dataframe} \
        --celltype "${celltype}" \
        --best_alpha ${best_alpha} \
        --n_feats ${n_feats} \
        --n_splits 2 \
        --n_folds 3 \
        --lasso_max_iteration 1000 \
        --parallel_cpus 8
    """
}


process MERGE_RFE_SCORE_CSV_FILES {
    input:
    tuple val(celltype), path(csv_files)

    output:
    tuple val(celltype), path("merged_rfe_scores_*.csv")

    script:
    // Remove spaces from the original string
    cleanedString = celltype.replaceAll(/[\s\/]+/, '')
    cleanedString = cleanedString.replaceAll(/\|/, '_')
    """
    # Concatenate all CSV files, sort by mean_test_score
    head -n 1 ${csv_files[0]} > merged_rfe_scores_${cleanedString}.csv
    tail -n +2 -q ${csv_files.join(' ')} | sort -t, -k1,1nr >> merged_rfe_scores_${cleanedString}.csv
    """
}


// Need to generate a comma seperated list of Celltype labels from Pandas
process EXAMINE_CLASS_LABEL{
    publishDir "${params.output_dir}/final_reports/plots", pattern: "feature_selection_*.png", mode: 'copy'

	input:
	tuple val(celltype), path(trainingDataframe), val(best_alpha), path(rfe_scores), path(alpha_scores)
    	
	output:
	path("top_rank_features_*.csv"), emit: feature_list
	tuple path("feature_selection_*_results.json"), path("feature_selection_*.csv"), path("feature_selection_*.png"), emit: feature_selection_results
    
    script:
    """
    generate_cell_type_selection.py \
        --trainingDataframe ${trainingDataframe} \
        --celltype "${celltype}" \
        --rfe_scores ${rfe_scores} \
        --best_alpha ${best_alpha} \
        --alpha_scores ${alpha_scores} \
        --classColumn "${params.classifed_column_name}" \
        --varThreshold 0.01 \
        --n_features_to_RFE 20 \
        --n_folds 12 \
        --ifSubsetData True \
        --max_workers 8 \
        --mim_class_label_threshold 20 \
        --n_alphas_to_search 8 \
    """
}

process MERGE_AND_SORT_CSV {
    input:
    path csv_files

    output:
    path("selected_features.csv")

    script:
    """
    head -n 1 ${csv_files[0]} > selected_features.csv
    tail -n +2 -q ${csv_files.join(' ')} | sort >> selected_features.csv
    """
}

process GENERATE_FS_REPORT {
    input:
    path(fs_files)
    path(features_list)
    path(html_template)

    output:
    path("feature_selection_report.html"), emit: fs_html

    script:
    """
    generate_feature_selection_report.py \
            --output-file feature_selection_report.html \
            --template-dir $html_template
    """
}


// -------------------------------------- //


workflow featureselection_wf {
    take: 
    trainingPickleTable
    celltypeCsv
    
    main:
    // Step 1: Split the celltype CSV into individual cell type labels
    list_channel = celltypeCsv
        .splitCsv(header: false, sep: ',').flatten()
    list_channel.dump(tag: 'markers', pretty: true)

    // Step 2: Generate binary data frames for each cell type label
    bls = TOP_LABEL_SPLIT(trainingPickleTable, list_channel)

    // Step 3: Generate logarithmically spaced alpha values for regularization search
    logspace_values_channel = Channel.from(
        (0..<96).collect { idx -> 
            Math.exp(-5.1 + idx * (Math.log(10) * (-0.00004 - (-5.1)) / 95)) / 100 
        }
    ).collate(6).map{ list -> list.join(',') }.flatten()
        
    // Step 4: Combine binary data frames and alpha values for parameter search
    combined_channel = bls.combine(logspace_values_channel).map { lbl, binary_df, logspace_values_chunk ->
        tuple( lbl, binary_df, logspace_values_chunk )
    }
    combined_channel.dump(tag: 'alpha_searching', pretty: true)
    
    // Step 5: Search for best alpha parameters for each cell type
    sfa = SEARCH_FOR_ALPHAS(combined_channel)
    
    // Step 6: Merge alpha search results CSV files for each cell type
    merged_csv = MERGE_ALPHAS_SEARCH_CSV_FILES(sfa.alphas.groupTuple())

    // Step 7: Select the best alpha value from merged CSVs
    best_alpha_channel = SELECT_BEST_ALPHA(merged_csv)
    
    // Step 8: Combine binary data frames and best alpha values for downstream analysis
    labelWithAlphas = bls
        .combine(best_alpha_channel, by: 0)
    // labelWithAlphas.view() // Check the structure of the combined tuples
    labelWithAlphas.dump(tag: 'labelWithAlphas', pretty: true)
    
    // Step 9: Generate a channel of feature counts for RFE (Recursive Feature Elimination)
    ref_counts = Channel.from(params.min_rfe_nfeatures..params.max_rfe_nfeatures)
    // Combine labelWithAlphas and feature counts for RFE
    scatter2_channel = labelWithAlphas.combine(ref_counts)
    //scatter2_channel.view()
    scatter2_channel.dump(tag: 'alpha_and_rfe', pretty: true)
    rfeRez = RUN_ALL_RFE(scatter2_channel)
    refScores = MERGE_RFE_SCORE_CSV_FILES(rfeRez.feature_scores.groupTuple())
    
    // Step 10: Join all relevant results for final feature selection and reporting
    labelWithEverything = labelWithAlphas.join(refScores, by: 0).join(merged_csv, by: 0)
    //labelWithEverything.view()    
    labelWithEverything.dump(tag: 'feat_sec_everything', pretty: true)
    
    // Step 11: Run feature selection and generate outputs
    fts = EXAMINE_CLASS_LABEL(labelWithEverything)
    mas = MERGE_AND_SORT_CSV(fts.feature_list.collect())

    final_results = fts.feature_selection_results
            .flatten()
            .collect()
    feature_list = fts.feature_list
            .flatten()
            .collect()

    fs_report = GENERATE_FS_REPORT(final_results, feature_list, params.html_template)
    
    // Step 12: Emit final results
    emit:
    mas_results = mas
    report = fs_report.fs_html
}
