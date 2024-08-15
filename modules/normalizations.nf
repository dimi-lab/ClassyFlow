// Produce Batch based normalization - boxcox
process boxcox {
	tag { batchID }
	executor "slurm"
    memory "50G"
    queue "cpu-short"
    time "24:00:00"
	
	publishDir(
        path: "${params.output_dir}/normalization_reports",
        pattern: "*.pdf",
        mode: "copy"
    )
	
	input:
	tuple val(batchID), path(pickleTable)
	
	output:
	tuple val(batchID), path("boxcox_transformed_${batchID}.tsv"), emit: bc_table
	path("boxcox_report_${batchID}.pdf")
	
	script:
	template 'boxcox_transformer.py'

}
    
    
// Produce Batch based normalization - quantile
process quantile {
	tag { batchID }
	executor "slurm"
    memory "50G"
    queue "cpu-short"
    time "24:00:00"
	
	publishDir(
        path: "${params.output_dir}/normalization_reports",
        pattern: "*.pdf",
        mode: "copy"
    )
	
	input:
	tuple val(batchID), path(pickleTable)
	
	output:
	tuple val(batchID), path("quantile_transformed_${batchID}.tsv"), emit: qt_table
	path("quantile_report_${batchID}.pdf")
	
	script:
	template 'quantile_transformer.py'

}

// Produce Batch based normalization - min/max scaling
process minmax {
	tag { batchID }
	executor "slurm"
    memory "40G"
    queue "cpu-short"
    time "24:00:00"
    
	publishDir(
        path: "${params.output_dir}/normalization_reports",
        pattern: "*.pdf",
        mode: "copy"
    )
	
	input:
	tuple val(batchID), path(pickleTable)
	
	output:
	tuple val(batchID), path("minmax_transformed_${batchID}.tsv"), emit: mm_table
	path("minmax_report_${batchID}.pdf")
	
	script:
	template 'minmax_transformer.py'

}

process logscale {
	tag { batchID }
	executor "slurm"
    memory "30G"
    queue "cpu-short"
    time "24:00:00"
    
	publishDir(
        path: "${params.output_dir}/normalization_reports",
        pattern: "*.pdf",
        mode: "copy"
    )
	
	input:
	tuple val(batchID), path(pickleTable)
	
	output:
	tuple val(batchID), path("log_transformed_${batchID}.tsv"), emit: lg_table
	path("log_report_${batchID}.pdf")
	
	script:
	template 'log_transformer.py'

}


// Look at all of the normalizations within a batch and attempt to idendity the best approach
process identify_best{
	publishDir(
        path: "${params.output_dir}/normalization_reports",
        pattern: "*.pdf",
        mode: "copy"
    )
    
	input:
	tuple val(batchID), path(all_possible_tables)
	
	output:
	tuple val(batchID), path("normalized_${batchID}.pkl"), emit: norm_df
	path("multinormalize_report_${batchID}.pdf")

	script:
	template 'characterize_normalization.py'
}
// -------------------------------------- //
    
    
    
workflow normalization_wf {
	take: 
	batchPickleTable
	
	main:
	bc = boxcox(batchPickleTable)
	qt = quantile(batchPickleTable)
	mm = minmax(batchPickleTable)
	lg = logscale(batchPickleTable)
	
	mxchannels = batchPickleTable.mix(bc.bc_table,qt.qt_table,mm.mm_table,lg.lg_table).groupTuple()
	mxchannels.dump(tag: 'debug_normalization_channels', pretty: true)
	
	bestN = identify_best(mxchannels)
	
	/// add multi-batch synchro later...
	
	emit:
	normalized = bestN.norm_df
	
}
