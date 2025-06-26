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
// -------------------------------------- //


workflow featureselection_wf {
	take: 
	trainingPickleTable
	celltypeCsv
	
	main:
	// Step1. Split the list into individual elements
	list_channel = celltypeCsv
		.splitCsv(header: false, sep: ',').flatten()
	list_channel.dump(tag: 'markers', pretty: true)

    //Step 2. Generate binary data frames for each label    
    bls = TOP_LABEL_SPLIT(trainingPickleTable, list_channel)

    //lgVals = logspace_values  //.collate(2)   //[a,b,c,d] = [[a,b],[c,d]]
    logspace_values_channel = Channel.from(
    (0..<96).collect { idx -> 
        Math.exp(-5.1 + idx * (Math.log(10) * (-0.0004 - (-5.1)) / 95)) 
        }
    ).collate(6).map{ list -> list.join(',') }.flatten()
        
    // Step 3: Combine bls and logspace_values_channel
    combined_channel = bls.combine(logspace_values_channel).map { lbl, binary_df, logspace_values_chunk ->
        tuple( lbl, binary_df, logspace_values_chunk )
    }
    combined_channel.dump(tag: 'alpha_searching', pretty: true)
    
    // Step 4. Search many parameters to determine best alpha per label
    sfa = SEARCH_FOR_ALPHAS(combined_channel)
    
    // Step 5. Merge CSV files into one per label
    merged_csv = MERGE_ALPHAS_SEARCH_CSV_FILES(sfa.alphas.groupTuple())

    // Step 6. Sort and Select the best alpha from the merged CSV
    best_alpha_channel = SELECT_BEST_ALPHA(merged_csv)
    
    
    //bls.view()    
    //best_alpha_channel.view()
    
    
    // Debugging intermediate outputs
    labelWithAlphas = bls
    .combine(best_alpha_channel, by: 0)
    //    labelWithAlphas.view() // Check the structure of the combined tuples
    labelWithAlphas.dump(tag: 'labelWithAlphas', pretty: true)
    
    ref_counts = Channel.from(params.min_rfe_nfeatures..params.max_rfe_nfeatures)
    //ref_counts.view()
    //labelWithAlphas.view()
    // Combine the `labelWithAlphas` with `ref_counts`
    scatter2_channel = labelWithAlphas.combine(ref_counts)
    //scatter2_channel.view()
       scatter2_channel.dump(tag: 'alpha_and_rfe', pretty: true)
    rfeRez = RUN_ALL_RFE(scatter2_channel)
    
    refScores = MERGE_RFE_SCORE_CSV_FILES(rfeRez.feature_scores.groupTuple())
    
    //labelWithEverything = labelWithAlphas.join(refScores, by: 0).map { labelAlpha, rfe_tuple ->
    //    def (celltype1, binary_df, best_alpha) = labelAlpha
    //    def (celltype2, rfe_csv) = rfe_tuple
    //    return tuple(celltype1, binary_df, best_alpha, rfe_csv)
    //}
    labelWithEverything = labelWithAlphas.join(refScores, by: 0).join(merged_csv, by: 0)
    //labelWithEverything.view()    
    labelWithEverything.dump(tag: 'feat_sec_everything', pretty: true)
	
	fts = EXAMINE_CLASS_LABEL(labelWithEverything)
		
	mas = MERGE_AND_SORT_CSV(fts.feature_list.collect())
	
	emit:
	mas

}
