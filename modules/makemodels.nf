
process xgboostingModel {
	publishDir(
        path: "${params.output_dir}/model_reports",
        pattern: "*.pdf",
        mode: "copy"
    )
	
	input:
	path(trainingDataframe)
	path(select_features_csv)
	
	output:
	tuple val('XGBoost'), path("xgb.pkl"), emit: model
	path("Model_Development_Xgboost.pdf")
	
	script:
	template 'get_xgboost.py'

}
process holdOutEvaluation{
	publishDir(
        path: "${params.output_dir}/model_reports",
        pattern: "*.pdf",
        mode: "copy"
    )
	
	input:
	path(holdoutDataframe)
	path(select_features_csv)
	tuple val(modelType), path("xgb.pkl")
	
	output:
	path("Holdout_${modelType}.json"), emit: eval
	path("Holdout_Evaluation_${modelType}.pdf")
	
	script:
	template 'get_holdout_evaluation.py'
	

}







workflow modelling_wf {
	take: 
	trainingPickleTable
	holdoutPickleTable
	featuresCSV

	main:
	xgb = xgboostingModel(trainingPickleTable, featuresCSV)
	
}	
	
