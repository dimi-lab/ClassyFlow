//Split training data and binarize by class label
process topLabelSplit {
	executor "slurm"
	memory "10G"
	queue "cpu-short"
	time "24:00:00"

	input:
	path(trainingDataframe)
	val(celltype)

	output:
	tuple val(celltype), path("binary_df*")
	//path("binary_count_table*"), emit: binary_cnts
	//path("non_variant_features*"), emit: nonvars

	script:
	template 'split_cell_type_labels.py'
}


process search_for_alphas {
	executor "slurm"
	memory "14G"
	queue "cpu-short"
	time "24:00:00"
	
	input:
	tuple val(celltype), path(binary_dataframe), val(logspace_chunk)
    
    output:
    val(celltype)
    path("alphas_params*"), emit: alphas
    
    script:
    template 'search_all_alphas.py'

}

process merge_alphas_search_csv_files {
    input:
    path csv_files

    output:
    path "merged_alphas.csv"

    script:
    """
    # Concatenate all CSV files, sort by mean_test_score
    head -n 1 ${csv_files[0]} > merged_alphas.csv
    tail -n +2 -q ${csv_files.join(' ')} | sort -t, -k1,1nr >> merged_alphas.csv
    """
}

process select_best_alpha {
    input:
    path merged_csv

    output:
    stdout

    shell:
    """
    # Extract the best_alpha where mean_test_score is the highest
    best_alpha=\$(awk -F, 'NR==2 {best=\$3} END {print best}' ${merged_csv})
    echo \$best_alpha
    """
}
process runAllRFE{
	executor "slurm"
	memory "14G"
	queue "cpu-short"
	time "24:00:00"

	input:
	tuple val(celltype), path(binary_dataframe), val(n_feats)
	val(best_alpha)
	    
    output:
    val(celltype)
    path("ref_scores*"), emit: alphas
    
    script:
    template 'calculate_RFE.py.py'
}




// Need to generate a comma seperated list of Celltype labels from Pandas
process examineClassLabel{
	executor "slurm"
    cpus 16
    memory "40G"
    queue "cpu-short"
    time "24:00:00"

	publishDir(
        path: "${params.output_dir}/celltype_reports",
        pattern: "*.pdf",
        mode: "copy"
    )
    
	input:
	tuple val(celltype), path(trainingDataframe)
	val(bestAlpha)
	
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
    path "selected_features.csv"

    script:
    """
    head -n 1 ${csv_files[0]} > selected_features.csv
    tail -n +2 -q ${csv_files.join(' ')} | sort >> selected_features.csv
    """
}
// -------------------------------------- //

//Need to generate list of possible alpha values to attempt
//logspace_values = (0..<24).collect { idx -> Math.exp(-5.1 + idx * (Math.log(10) * (-0.0004 - (-5.1)) / 23)) }

workflow featureselection_wf {
	take: 
	trainingPickleTable
	celltypeCsv
	
	main:
	// Split the list into individual elements
	list_channel = celltypeCsv
		.splitCsv(header: false, sep: ',')
	//     list_channel.view()

	//Generate binary data frames	
	bls = topLabelSplit(trainingPickleTable, list_channel)

	// Debugging views (Remove these if not needed)
    // bls.lbl.view()
    // bls.binary_df.view()
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
	
	sfa = search_for_alphas(combined_channel)
	
	// Merge CSV files into one
    merged_csv = merge_alphas_search_csv_files(sfa.alphas.collect())

    // Select the best alpha from the merged CSV
    best_alpha = select_best_alpha(merged_csv)
	best_alpha.view()
	
	
	ref_counts = Channel.from(2..12)
	combined_channel2 = bls.combine(ref_counts).map { lbl, binary_df, rfeIdx ->
		tuple( lbl, binary_df, rfeIdx )
	}
	
	rfeRez = runAllRFE(combined_channel2, best_alpha)
	
	fts = examineClassLabel(bls, best_alpha)
		
	mas = mergeAndSortCsv(fts.feature_list.collect())
	
	emit:
	mas

}
