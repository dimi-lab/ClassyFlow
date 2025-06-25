process CREATE_XGB_PARAMS {
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

process XGBOOSTING_MODEL {
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

process MERGE_XGB_CSV {
    input:
    path csv_files

    output:
    path 'merged_xgb_performance_output.csv', emit: table

    script:
    """
    head -n 1 \$(ls parameters_found_*.csv | head -n 1) > merged_xgb_performance_output.csv
    for file in parameters_found_*.csv; do
        tail -n +2 "\$file" >> merged_xgb_performance_output.csv
    done
    """
}

process XGBOOSTING_FINAL_MODEL {
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
	path("classes.npy"), emit: classes
    tuple path("xgbWinners_*.png"), path("xgbWinners_parameter_summary.csv"), path("xgbWinners_results.json"), emit: xgboost_output
	
	script:
    """
    get_xgboost_winners.py \
        --classColumn ${params.classifed_column_name} \
        --cpu_jobs 16 \
        --mim_class_label_threshold ${params.minimum_label_count} \
        --model_performance_table ${model_performance_table} \
        --trainingDataframe ${trainingDataframe} \
        --select_features_csv ${select_features_csv}
    """
}

process HOLDOUT_XGB_EVALUATION {  
	input:
	path(holdoutDataframe)
	path(select_features_csv)
	path(model_pickle)
	path(leEncoderFile)

	output:
	tuple path("XGBoost_Model_*.png"), path("XGBoost_Model_*_auc_rankings.csv"), path("XGBoost_Model_*_results.json"), emit: holdoutEval_output
    path("holdout_*.csv"), emit: eval

	
	script:
    """
    get_holdout_evaluation.py \
        --classColumn ${params.classifed_column_name} \
        --leEncoderFile ${leEncoderFile} \
        --model_pickle ${model_pickle} \
        --holdoutDataframe ${holdoutDataframe} \
        --select_features_csv ${select_features_csv}
    """
}

process MERGE_HOLDOUT_CSV {
    publishDir(
        path: "${params.output_dir}/models",
        pattern: "merged_holdout_performance.csv",
        overwrite: true,
        mode: "copy"
    )
    input:
    path csv_files

    output:
    path 'merged_holdout_performance.csv', emit: table

    script:
    """
    head -n 1 \$(ls holdout_*.csv | head -n 1) > merged_holdout_performance.csv
    for file in holdout_*.csv; do
        tail -n +2 "\$file" >> merged_holdout_performance.csv
    done
    """
}

process SELECT_BEST_MODEL {
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
    xgbconfig = CREATE_XGB_PARAMS()
    params_channel = xgbconfig.params.splitCsv( header: true, sep: ',' )
    
    xgbHyper = XGBOOSTING_MODEL(trainingPickleTable, featuresCSV, params_channel)
    paramSearch = MERGE_XGB_CSV(xgbHyper.behavior.collect())
    
    xgbModels = XGBOOSTING_FINAL_MODEL(trainingPickleTable, featuresCSV, paramSearch.table)
    
    allModelsTrained = xgbModels.m1.concat(xgbModels.m2).flatten()
    allModelsTrained.subscribe { println "Model: $it" }
    
    allHoldoutResults = HOLDOUT_XGB_EVALUATION(
        holdoutPickleTable, 
        featuresCSV, 
        allModelsTrained, 
        xgbModels.classes
    )
    
    holdoutEval = MERGE_HOLDOUT_CSV(allHoldoutResults.eval.collect())
    selected = SELECT_BEST_MODEL(holdoutEval.table)
    
    selected.outfile.subscribe { println "Selected outfile: $it" }
    xgbModels.classes.subscribe { println "Classes file: $it" }

    best_model_info = selected.outfile.map { line -> 
        def (name, path) = line.text.split(',')
        tuple(name.trim(), file(path.trim()), xgbModels.classes.value)
    }
    best_model_info.subscribe { println "Best Model: $it" }

    emit:
    best_model_info
}
