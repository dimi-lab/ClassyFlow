//Split training data and binarize by class label
process topLabelSplit {
	executor "slurm"
	memory "10G"
	queue "cpu-short"
	time "5:00:00"

	input:
	path(trainingDataframe)
	val(celltype)

	output:
	tuple val(celltype), path("binary_df*"), optional: true
	
	script:
	template 'split_cell_type_labels.py'
}

process search_for_alphas {
	executor "slurm"
	memory "14G"
	queue "cpu-short"
	time "5:00:00"
	
	input:
	tuple val(celltype), path(binary_dataframe), val(logspace_chunk)
    
    output:
    tuple val(celltype), path("alphas_params*"), emit: alphas
    
    script:
    template 'search_all_alphas.py'
}

process merge_alphas_search_csv_files {
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

process select_best_alpha {
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


process runAllRFE{
	executor "slurm"
    cpus 8
	memory "5G"
	queue "cpu-short"
	time "6:50:00"

	input:
	tuple val(celltype), path(binary_dataframe), val(best_alpha), val(n_feats)
	    
    output:
    tuple val(celltype), path("rfe_scores*"), emit: feature_scores
    
    script:
    template 'calculate_RFE.py'
}


process merge_rfe_score_csv_files {
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
process examineClassLabel{
	executor "slurm"
    memory "20G"
    queue "cpu-short"
    time "4:00:00"

	publishDir(
        path: "${params.output_dir}/celltype_reports",
        pattern: "*.pdf",
        mode: "copy"
    )
    
	input:
	tuple val(celltype), path(trainingDataframe), val(best_alpha), path(rfe_scores), path(alpha_scores)
	
	output:
	path("top_rank_features_*.csv"), emit: feature_list
	path("*_Features.pdf")
    
    script:
	template 'generate_cell_type_selection.py'
}

process mergeAndSortCsv {
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
	bls = topLabelSplit(trainingPickleTable, list_channel)

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
	sfa = search_for_alphas(combined_channel)
	
	// Step 5. Merge CSV files into one per label
    merged_csv = merge_alphas_search_csv_files(sfa.alphas.groupTuple())

    // Step 6. Sort and Select the best alpha from the merged CSV
    best_alpha_channel = select_best_alpha(merged_csv)
	
	
	//bls.view()	
	//best_alpha_channel.view()
	
	
	// Debugging intermediate outputs
	labelWithAlphas = bls
    .combine(best_alpha_channel, by: 0)
	//	labelWithAlphas.view() // Check the structure of the combined tuples
    labelWithAlphas.dump(tag: 'labelWithAlphas', pretty: true)
    
	ref_counts = Channel.from(params.min_rfe_nfeatures..params.max_rfe_nfeatures)
	//ref_counts.view()
	//labelWithAlphas.view()
    // Combine the `labelWithAlphas` with `ref_counts`
    scatter2_channel = labelWithAlphas.combine(ref_counts)
    //scatter2_channel.view()
   	scatter2_channel.dump(tag: 'alpha_and_rfe', pretty: true)
	rfeRez = runAllRFE(scatter2_channel)
	
	refScores = merge_rfe_score_csv_files(rfeRez.feature_scores.groupTuple())
	
	//labelWithEverything = labelWithAlphas.join(refScores, by: 0).map { labelAlpha, rfe_tuple ->
    //    def (celltype1, binary_df, best_alpha) = labelAlpha
    //    def (celltype2, rfe_csv) = rfe_tuple
    //    return tuple(celltype1, binary_df, best_alpha, rfe_csv)
    //}
    labelWithEverything = labelWithAlphas.join(refScores, by: 0).join(merged_csv, by: 0)
    //labelWithEverything.view()	
    labelWithEverything.dump(tag: 'feat_sec_everything', pretty: true)
	
	fts = examineClassLabel(labelWithEverything)
		
	mas = mergeAndSortCsv(fts.feature_list.collect())
	
	emit:
	mas

}
