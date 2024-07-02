
process xgboostingModel {
	publishDir(
        path: "${params.output_dir}/model_reports",
        pattern: "*.pdf",
        mode: "copy"
    )
	
	input:
	path(trainingDataframe)
	
	output:
	tuple val('XGBoost'), path("xgb.pkl"), emit: model
	path("log_report_${batchID}.pdf")
	
	script:
	template 'get_xgboost.py'

}








workflow featureselection_wf {
	take: 
	trainingPickleTable
	holdoutPickleTable

	main:
	xgb = xgboostingModel(batchPickleTable)
	
}	
	
