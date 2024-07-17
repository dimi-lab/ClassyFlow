// Need to generate a comma seperated list of Celltype labels from Pandas
process examineClassLabel{
	executor "slurm"
    cpus 8
    memory "40G"
    queue "cpu-short"
    time "24:00:00"

	publishDir(
        path: "${params.output_dir}/celltype_reports",
        pattern: "*.pdf",
        mode: "copy"
    )
    
	input:
	path(trainingDataframe)
	val(celltype)
	
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


workflow featureselection_wf {
	take: 
	trainingPickleTable
	celltypeCsv
	
	main:
	// Split the list into individual elements
	list_channel = celltypeCsv
		.splitCsv(header: false, sep: ',')

	fts = examineClassLabel(trainingPickleTable, list_channel)
	mas = mergeAndSortCsv(fts.feature_list.collect())
	
	emit:
	mas

}
