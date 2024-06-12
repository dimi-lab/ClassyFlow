// Align reads with BWA MEM
process boxcox {
	publishDir(
        path: "${params.output_dir}/output_reports",
        pattern: "*.html",
        mode: "copy"
    )
	
	input:
	path pickleTable
	path batchDir
	
	output:
	path "boxcox_transformed_${idx}.tsv", emit: bc_table
	path "boxcox_report_${idx}.pdf"
	
	script:
    batchID = batchDir.baseName
	template 'boxcox_transformer.py'

}
    
workflow normalization_wf{
	take: 
	batchPickleTable
	originalDir

	main:
	//Examine BoxCox Transformation + Zscore (best so far)
	boxcox(batchPickleTable, originalDir)

}
