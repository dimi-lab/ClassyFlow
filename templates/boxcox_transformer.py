#!/usr/bin/env python3

import sys, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.backends.backend_pdf
from scipy.stats import boxcox


def collect_and_transform(df, pdfOUT, qTyp):
	## Print original values figure
	df['Image'] = [e.replace('.ome.tiff', '') for e in df['Image'].tolist() ]

	smTble = df.groupby('Slide').apply(lambda x: x.sample(frac=0.25)) 
	df_batching = smTble.filter(regex='(Mean|Median|Slide)',axis=1)
	df_melted = pd.melt(df_batching, id_vars=["Slide"])

	fig, ax1 = plt.subplots(figsize=(24,8))
	origVals = sns.boxplot(x='Slide', y='value', color="#CD7F32", data=df_melted, ax=ax1, showfliers = False)
	plt.setp(ax1.get_xticklabels(), rotation=40, ha="right")
	ax1.set_title('Combined Marker Distribution (original values)')
	pdfOUT.savefig( origVals.get_figure() )
	
	df_batching = smTble.filter(regex='(Mean|Median|Image|Slide)',axis=1)
	df_melted = pd.melt(df_batching, id_vars=["Image","Slide"])

	fig, ax1 = plt.subplots(figsize=(24,8))
	origVals = sns.boxplot(x='Image', y='value', hue="Slide", data=df_melted, ax=ax1, showfliers = False)
	plt.setp(ax1.get_xticklabels(), rotation=40, ha="right")
	ax1.set_title('Combined Marker Distribution (original values)')
	pdfOUT.savefig( origVals.get_figure() )
	
	if qTyp == 'CellObject':
		df_batching2 = smTble.filter(regex='Cell: Mean',axis=1)
	else: 
		df_batching2 = smTble.filter(regex='Mean',axis=1)
	
	df_batching2.plot.density(figsize = (24, 6),linewidth = 3)
	plt.title("Marker Distributions (original values)")
	pdfOUT.savefig( plt.gcf() )
	
	
	### Do Box Cox on per batch level
	### Need to treat each statistical feature as it's own transformation distribution. 
	### (Avoid over skew correction) will calc lamda based on provided distribution.
	### Really should be using all of the data for transformation. 
	# https://machinelearningmastery.com/power-transforms-with-scikit-learn/
	# https://www.jstor.org/stable/2109981
	metrics = []
	bcDf = df.copy(deep=True).fillna(0)
	for fld in list(bcDf.filter(regex="(Min|Max|Median|Mean|StdDev)$")):
		preMu = bcDf[fld].mean()
		# print("{} -> {:.2f} mu".format(fld,preMu))
		try:
			nArr, mxLambda = boxcox(bcDf[fld].add(1).values)
			bcDf[fld] = nArr
			mxLambda = "{:.3f}".format(mxLambda)
		except:
			bcDf[fld] = 0
			mxLambda = 'Failed'
		metrics.append([fld,preMu,mxLambda,bcDf[fld].mean(),bcDf[fld].min(),bcDf[fld].max()])
		
	bxcxMetrics = pd.DataFrame(metrics, columns =['Feature','Pre_Mean','Lambda','Post_Mean','Post_Min','Post_Max'])
	bxcxMetrics.to_csv("BoxCoxRecord.csv")
	
	tmpPlot = bxcxMetrics.loc[bxcxMetrics['Lambda'] != 'Failed']
	bxcxMetrics.plot.scatter(x='Pre_Mean', y='Post_Mean', figsize = (12, 12) )
	plt.title("Feature Avg Pre v. Post (BoxCox)")
	pdfOUT.savefig( plt.gcf() )
	
	


if __name__ == "__main__":
	myData = pd.read_pickle("merged_dataframe.pkl")
	myFileIdx = "SET02"
	quantType = 'CellObject'
	#myData = pd.read_pickle("${pickleTable}")
	#myFileIdx = "${batchID}"
	
	pdfOUT = matplotlib.backends.backend_pdf.PdfPages("boxcox_report_{}.pdf".format(myFileIdx))
	collect_and_transform(myData, pdfOUT, quantType)
	pdfOUT.close()
	
	





