#!/usr/bin/env python3

import sys, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.backends.backend_pdf
from scipy.stats import boxcox


def collect_and_transform(df, pdfOUT, qTyp, nucMark):
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
	for fld in list(bcDf.filter(regex='(Min|Max|Median|Mean|StdDev)')):
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
	bxcxMetrics.plot.scatter(y='Pre_Mean', x='Post_Mean', figsize = (12, 12) )
	plt.title("Feature Avg Pre v. Post (BoxCox)")
	pdfOUT.savefig( plt.gcf() )
	
	
	smTble = bcDf.groupby('Slide').apply(lambda x: x.sample(frac=0.25)) 
	df_batching = smTble.filter(regex='(Mean|Median|Slide)',axis=1)
	df_melted = pd.melt(df_batching, id_vars=["Slide"])
	
	fig, ax1 = plt.subplots(figsize=(24,8))
	origVals = sns.boxplot(x='Slide', y='value', color="#50C878", data=df_melted, ax=ax1, showfliers = False)
	plt.setp(ax1.get_xticklabels(), rotation=40, ha="right")
	ax1.set_title('Combined Marker Distribution (boxcox values)')
	pdfOUT.savefig( origVals.get_figure() )
	
	
	colNames = list(filter(lambda x:'Mean' in x, df.columns.tolist()))[0:10]
	NucOnly = list(filter(lambda x:nucMark in x, colNames))[0]
	for i in range(0, len(colNames), 8):
		# Create a new figure for each page
		fig, axs = plt.subplots(8, 2, figsize=(8.5, 11))
		axs = axs.flatten()

		for j in range(8):
			if i + j < len(colNames):
				hd = colNames[(i + j)]
				nuc1 = pd.DataFrame({"Original_Value": df[NucOnly], "Transformed_Value":bcDf[NucOnly]})
				nuc1['Mark'] = nucMark
				mk2 = pd.DataFrame({"Original_Value": df[hd], "Transformed_Value":bcDf[hd]})
				mk2['Mark'] = hd.split(":")[0]
				qqDF = pd.concat([nuc1,mk2], ignore_index=True)

				# Plot on the j-th subplot
				ax2 = axs[j]

				sns.scatterplot(x='Original_Value', y='Transformed_Value', data=qqDF, hue="Mark", ax=ax2)
				ax2.set_title("BoxCox Transformation: {}".format(hd))
				ax2.axline((0, 0), (nuc1['Original_Value'].max(), nuc1['Transformed_Value'].max()), linewidth=2, color='r')

			else:
				axs[j].axis('off')  # Turn off axis if no data

		# Adjust layout and save the page to the PDF
		plt.tight_layout()
		pdfOUT.savefig(fig)
	
	
	
	return bcDf

if __name__ == "__main__":
	myData = pd.read_pickle("${pickleTable}")
	myFileIdx = "${batchID}"
	quantType = '${params.qupath_object_type}'
	nucMark = '${params.nucleus_marker}'
			
	pdfOUT = matplotlib.backends.backend_pdf.PdfPages("boxcox_report_{}.pdf".format(myFileIdx))
	trnsfTBL = collect_and_transform(myData, pdfOUT, quantType, nucMark)
	pdfOUT.close()
	
	trnsfTBL.to_csv("boxcox_transformed_{}.tsv".format(myFileIdx), sep="\t") 	
	
	





