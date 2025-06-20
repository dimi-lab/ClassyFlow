process createXGBParams {
    output:
    path("xgb_iterate_params.csv"), emit: params

    script:
    """
    get_xgboost_parameter_search.py \
        --max_cv ${params.max_xgb_cv} \
        --depth_start ${params.xgb_depth_start} \
        --depth_stop ${params.xgb_depth_stop} \
        --depth_step ${params.xgb_depth_step} \
        --learnRates "${params.xgb_learn_rates}"
    """
}
process xgboostingModel {
	input:
	path(trainingDataframe)
	path(select_features_csv)
	tuple val(cv_c), val(depth_d), val(eta_l)
	
	output:
	path("parameters_found_*.csv"), emit: behavior
	
	script:
    """
    get_xgboost.py \
        --classColumn ${params.classifed_column_name} \
        --cpu_jobs 16 \
        --uTaskID ${task.index} \
        --mim_class_label_threshold ${params.minimum_label_count} \
        --depth_d ${depth_d} \
        --eta_l ${eta_l} \
        --cv_c ${cv_c} \
        --trainingDataframe ${trainingDataframe} \
        --select_features_csv ${select_features_csv}
    """

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
	path(letterhead)
	
	output:
	path("XGBoost_Model_First.pkl"), emit: m1
	path("XGBoost_Model_Second.pkl"), emit: m2
	path("Model_Development_Xgboost.pdf")
	path("classes.npy"), emit: classes
	
	script:
    """
    get_xgboost_winners.py \
        --classColumn ${params.classifed_column_name} \
        --cpu_jobs 16 \
        --mim_class_label_threshold ${params.minimum_label_count} \
        --letterhead "${letterhead}" \
        --model_performance_table ${model_performance_table} \
        --trainingDataframe ${trainingDataframe} \
        --select_features_csv ${select_features_csv}
    """
}
process holdOutXgbEvaluation{  
	publishDir(
        path: "${params.output_dir}/model_reports",
        pattern: "*.pdf",
        mode: "copy"
    )
	
	input:
	path(holdoutDataframe)
	path(select_features_csv)
	path(model_pickle)
	path(leEncoderFile)
	path(letterhead)
	
	output:
	path("holdout_*.csv"), emit: eval
	path("Holdout_on_*.pdf")
	
	script:
    """
    get_holdout_evaluation.py \
        --classColumn ${params.classifed_column_name} \
        --leEncoderFile ${leEncoderFile} \
        --letterhead "${letterhead}" \
        --model_pickle ${model_pickle} \
        --holdoutDataframe ${holdoutDataframe} \
        --select_features_csv ${select_features_csv}
    """
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
	letterhead

	main:
	// Gerneate all the permutations for xgboost parameter search
	xgbconfig = createXGBParams()
	params_channel = xgbconfig.params.splitCsv( header: true, sep: ',' )
	
	// params_channel.subscribe { println "Params: $it" }
	xgbHyper = xgboostingModel(trainingPickleTable, featuresCSV, params_channel)
	paramSearch = mergeXgbCsv(xgbHyper.behavior.collect())
	
	xgbModels = xgboostingFinalModel(trainingPickleTable, featuresCSV, paramSearch.table, letterhead)
	
	/// Able to add more modelling modules here
	
	allModelsTrained = xgbModels.m1.concat(xgbModels.m2).flatten()
	allModelsTrained.subscribe { println "Model: $it" }
	//allModelsTrained.view()
	
	allHoldoutResults = holdOutXgbEvaluation(
        holdoutPickleTable, 
        featuresCSV, 
        allModelsTrained, 
        xgbModels.classes,
        letterhead
    )
	
	holdoutEval = mergeHoldoutCsv(allHoldoutResults.eval.collect())
	selected = selectBestModel(holdoutEval.table)
	
	selected.outfile.subscribe { println "Selected outfile: $it" }
	xgbModels.classes.subscribe { println "Classes file: $it" }

    // Step 4: Create a channel that emits the best model info
    // This will emit a tuple of (model_name, model_path, leEncoderFile)
    // where model_name is the name of the best model,
    // model_path is the path to the best model file,
    // and leEncoderFile is the label encoder file.
    
    // Note: Assuming `xgbModels.classes` contains the label encoder file path
    // and `selected.outfile` contains the best model information.
    best_model_info = selected.outfile.map { line -> 
        def (name, path) = line.text.split(',')
        tuple(name.trim(), file(path.trim()), xgbModels.classes.value)
    }
    // Print the best model info
    best_model_info.subscribe { println "Best Model: $it" }

	emit:
	best_model_info
}
