process createXGBParams {
	output:
	path("xgb_iterate_params.csv"), emit: params

	script:
	template 'get_xgboost_parameter_search.py'
}
process xgboostingModel {
	executor "slurm"
    cpus 16
    memory "30G"
    queue "cpu-short"
    time "8:00:00"

	input:
	path(trainingDataframe)
	path(select_features_csv)
	tuple val(cv_c), val(depth_d), val(eta_l)
	
	output:
	path("parameters_found_*.csv"), emit: behavior
	
	script:
	template 'get_xgboost.py'

}
process mergeXgbCsv {
    // Define the input and output
    input:
    path csv_files

    output:
    path 'merged_xgb_performance_output.csv', emit: table

    // Script block
    script:
    """
    # Extract the header from the first CSV file
    head -n 1 \$(ls parameters_found_*.csv | head -n 1) > merged_xgb_performance_output.csv

    # Concatenate all CSV files excluding their headers
    for file in parameters_found_*.csv; do
        tail -n +2 "\$file" >> merged_xgb_performance_output.csv
    done
    """
}

process xgboostingFinalModel {
	executor "slurm"
    memory "30G"
    queue "cpu-short"
    time "8:00:00"
    
	publishDir(
        path: "${params.output_dir}/model_reports",
        pattern: "*.pdf",
        mode: "copy"
    )
    publishDir(
        path: "${params.output_dir}/models",
        pattern: "*_Model_*.pkl",
        overwrite: true,
        mode: "copy"
    )
    publishDir(
        path: "${params.output_dir}/models",
        pattern: "classes.npy",
        overwrite: true,
        mode: "copy"
    )
	
	input:
	path(trainingDataframe)
	path(select_features_csv)
	path(model_performance_table)
	
	output:
	path("XGBoost_Model_First.pkl"), emit: m1
	path("XGBoost_Model_Second.pkl"), emit: m2
	path("Model_Development_Xgboost.pdf")
	path("classes.npy")
	
	script:
	template 'get_xgboost_winners.py'

}
process holdOutXgbEvaluation{
	executor "slurm"
    memory "30G"
    queue "cpu-short"
    time "8:00:00"
    
	publishDir(
        path: "${params.output_dir}/model_reports",
        pattern: "*.pdf",
        mode: "copy"
    )
	
	input:
	path(holdoutDataframe)
	path(select_features_csv)
	path(model_pickle)
	
	output:
	path("holdout_*.csv"), emit: eval
	path("Holdout_on_*.pdf")
	
	script:
	template 'get_holdout_evaluation.py'
}
process mergeHoldoutCsv {
	publishDir(
        path: "${params.output_dir}/models",
        pattern: "merged_holdout_performance.csv",
        overwrite: true,
        mode: "copy"
    )
    // Define the input and output
    input:
    path csv_files

    output:
    path 'merged_holdout_performance.csv', emit: table

    // Script block
    script:
    """
    # Extract the header from the first CSV file
    head -n 1 \$(ls holdout_*.csv | head -n 1) > merged_holdout_performance.csv

    # Concatenate all CSV files excluding their headers
    for file in holdout_*.csv; do
        tail -n +2 "\$file" >> merged_holdout_performance.csv
    done
    """
}

process selectBestModel {
	input:
	path csv_file

	output:
	path("best_model_info.txt"), emit: outfile

	script:
	"""
	best_model=\$(awk -F, 'NR > 1 { if(\$2 > max) { max=\$2; model=\$1 } } END { print model }' ${csv_file})
	best_model_path="${params.output_dir}/models/\${best_model}"
	echo "\$best_model,\$best_model_path" > best_model_info.txt
	"""
}


workflow modelling_wf {
	take: 
	trainingPickleTable
	holdoutPickleTable
	featuresCSV

	main:
	// Gerneate all the permutations for xgboost parameter search
	xgbconfig = createXGBParams()
	params_channel = xgbconfig.params.splitCsv( header: true, sep: ',' )
	
	// params_channel.subscribe { println "Params: $it" }
	xgbHyper = xgboostingModel(trainingPickleTable, featuresCSV, params_channel)
	paramSearch = mergeXgbCsv(xgbHyper.behavior.collect())
	
	xgbModels = xgboostingFinalModel(trainingPickleTable, featuresCSV, paramSearch.table)
	
	/// Able to add more modelling modules here
	
	//allModelsTrained = Channel.fromPath( "${params.output_dir}/models/XGBoost_Model*.pkl" )
	allModelsTrained = xgbModels.m1.concat(xgbModels.m2).flatten()
	allModelsTrained.subscribe { println "Model: $it" }
	//allModelsTrained.view()
	
	allHoldoutResults = holdOutXgbEvaluation(holdoutPickleTable, featuresCSV, allModelsTrained)
	
	holdoutEval = mergeHoldoutCsv(allHoldoutResults.eval.collect())
	selected = selectBestModel(holdoutEval.table)
	
	// Step 5: Emit the best model name and path
    best_model_info = selected.outfile.map { line -> 
        def (name, path) = line.text.split(',')
        tuple(name.trim(), file(path.trim()))
    }
    
    // Print the best model info
    best_model_info.subscribe { println "Best Model: $it" }
	
	emit:
	best_model_info
	
}	
	
