// Produce Batch based normalization - boxcox
process boxcox {
	publishDir(
        path: "${params.output_dir}/normalization_reports",
        pattern: "*.pdf",
        mode: "copy"
    )
	
	input:
	tuple val(batchID), path(pickleTable)
	
	output:
	path "boxcox_transformed_${batchID}.tsv", emit: bc_table
	path "boxcox_report_${batchID}.pdf"
	
	script:
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
	tuple val(batchID), path(pickleTable)
	
	output:
	path "quantile_transformed_${batchID}.tsv", emit: qt_table
	path "quantile_report_${batchID}.pdf"
	
	script:
	template 'quantile_transformer.py'

}

// Produce Batch based normalization - boxcox
process minmax {
	publishDir(
        path: "${params.output_dir}/normalization_reports",
        pattern: "*.pdf",
        mode: "copy"
    )
	
	input:
	tuple val(batchID), path(pickleTable)
	
	output:
	path "minmax_transformed_${batchID}.tsv", emit: mm_table
	path "minmax_report_${batchID}.pdf"
	
	script:
	template 'minmax_transformer.py'

}
    
    
workflow normalization_wf{
	take: 
	batchPickleTable
	
	main:
	//Examine Transformations
	boxcox(batchPickleTable)
	
	quantile(batchPickleTable)

	minmax(batchPickleTable)
	
	

}
