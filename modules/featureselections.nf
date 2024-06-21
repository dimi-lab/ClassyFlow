// Need to generate a comma seperated list of Celltype labels from Pandas
process examineClassLabel{
	publishDir(
        path: "${params.output_dir}/celltype_reports",
        pattern: "*.pdf",
        mode: "copy"
    )
    
	input:
	path(trainingDataframe)
	val(celltype)
	
	output:
	tuple val(celltype), path("normalized_${batchID}.pkl"), emit: norm_df
	path("multinormalize_report_${batchID}.pdf")

    
    script:
	template 'generate_cell_type_selection.py'
}







workflow featureselection_wf {
	take: 
	trainingPickleTable
	celltypeCsv
	
	main:
	// Split the list into individual elements
	list_channel = celltypeCsv
		.splitCsv(header: false, sep: ',')


	examineClassLabel(trainingPickleTable, list_channel)

}
