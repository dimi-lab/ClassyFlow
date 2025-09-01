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
    publishDir(
        path: "${params.output_dir}/final_reports/plots",
        pattern: "xgbWinners_*.png",
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
    tuple path("xgbWinners_*.png"), path("xgbWinners_*.csv"), path("xgbWinners_results.json"), emit: xgboost_results
	
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
publishDir(
        path: "${params.output_dir}/final_reports/plots",
        pattern: "holdoutEval_XGBoost_Model_*.png",
        overwrite: true,
        mode: "copy"
    )
    
	input:
	path(holdoutDataframe)
	path(select_features_csv)
	path(model_pickle)
	path(leEncoderFile)

	output:
	tuple path("holdoutEval_XGBoost_Model_*.png"), path("holdoutEval_XGBoost_Model_*_auc_rankings.csv"), path("holdoutEval_XGBoost_Model_*_results.json"), emit: holdoutEval_results
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

process GLMTRANS_MODEL {
    input:
    path(trainingDataframe)
    path(select_features_csv)
    path(holdoutDataframe)
    tuple val(mdl_name), path(best_model_info), path(classes_encoded)
    val(celltype)

    output:
    path("GLMtrans_model.pkl"), emit: glm_model
    path("GLMtrans_eval.json"), emit: glm_eval
    path("*.png"), emit: glm_coeffs

    script:
    """
    train_glmtrans.py \
        --train_data ${trainingDataframe} \
        --features_csv ${select_features_csv} \
        --holdout_data ${holdoutDataframe} \
        --celltype "${celltype}" \
        --output_model GLMtrans_model.pkl \
        --output_eval GLMtrans_eval.json \
        --output_coeffs GLMtrans_coefficients.png \
        --compare_xgb ${best_model_info} \
        --output_comparison GLMtrans_vs_xgb_comparison.png
    """
}

process GENERATE_MODEL_REPORT {
    publishDir(
        path: "${params.output_dir}/final_reports/plots",
        pattern: "train_holdout_comparison.png",
        overwrite: true,
        mode: "copy"
    )
    input:
    path(xgb_files)
    path(holdout_files)
    path(training_pickle)
    path(holdout_pickle)
    path(html_template)

    output:
    path("model_report.html"), emit: model_html
    path("train_holdout_comparison.png")

    script:
    """
    generate_model_report.py \
            --output-file model_report.html \
            --training-dataframe $training_pickle \
            --holdout-dataframe $holdout_pickle \
            --template-dir $html_template \
            --class-column ${params.classifed_column_name}
    """

}



workflow modelling_wf {
    take: 
    trainingPickleTable
    holdoutPickleTable
    featuresCSV
    celltypeCsv

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

    list_channel = celltypeCsv
        .splitCsv(header: false, sep: ',').flatten()

    list_channel.subscribe { println "Label: $it" }


    // glmtrans_results = GLMTRANS_MODEL(
    //     trainingPickleTable,
    //     featuresCSV,
    //     holdoutPickleTable,
    //     best_model_info,
    //     list_channel
    // )

    xgb_results = xgbModels.xgboost_results
        .flatten()
        .collect()
    
    holdout_evals = allHoldoutResults.holdoutEval_results
        .flatten()
        .collect()

    model_report = GENERATE_MODEL_REPORT(xgb_results, holdout_evals, trainingPickleTable, holdoutPickleTable, params.html_template)

    emit:
    best_model_results = best_model_info
    report = model_report.model_html
}
