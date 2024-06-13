// Produce Batch based normalization - boxcox
process boxcox {
	publishDir(
        path: "${params.output_dir}/normalization_reports",
        pattern: "*.pdf",
        mode: "copy"
    )
	
	input:
	path pickleTable
	path batchDir
	
	output:
	path "boxcox_transformed_${batchID}.tsv", emit: bc_table
	path "boxcox_report_${batchID}.pdf"
	
	script:
    batchID = batchDir.baseName
	template 'boxcox_transformer.py'

}
    
    
// Produce Batch based normalization - boxcox
process quantile {
	publishDir(
        path: "${params.output_dir}/normalization_reports",
        pattern: "*.pdf",
        mode: "copy"
    )
	
	input:
	path pickleTable
	path batchDir
	
	output:
	path "quantile_transformed_${batchID}.tsv", emit: qt_table
	path "quantile_report_${batchID}.pdf"
	
	script:
    batchID = batchDir.baseName
	template 'quantile_transformer.py'

}
    
    
workflow normalization_wf{
	take: 
	batchPickleTable
	originalDir

	main:
	//Examine Transformations
	boxcox(batchPickleTable, originalDir)
	
	quantile(batchPickleTable, originalDir)

}
