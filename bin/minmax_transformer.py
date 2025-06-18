#!/usr/bin/env python3

import sys, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.backends.backend_pdf
from sklearn.preprocessing import MinMaxScaler

def collect_and_transform(df, pdfOUT, qTyp, nucMark):
	## Print original values figure
	df['Image'] = [e.replace('.ome.tiff', '') for e in df['Image'].tolist() ]

	#smTble = df.groupby('Slide').apply(lambda x: x.sample(frac=0.25))
	smTble = df.groupby('Slide', group_keys=False).apply(lambda x: x.sample(frac=0.25))
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
	
	# Drop columns with no variability (all values are the same)
	df_batching2 = df_batching2.loc[:, df_batching2.nunique() > 1]
	
	#print(df_batching2.describe())
	df_batching2.plot.density(figsize = (24, 6),linewidth = 3)
	plt.title("Marker Distributions (original values)")
	pdfOUT.savefig( plt.gcf() )
	
	
	# This estimator scales and translates each feature individually such that it is in the given range on the training set,
	# e.g. between zero and one.
	scaler = MinMaxScaler(feature_range=(-2,2))
	
	# grab just quant fields
	imgMets = df.filter(regex='(Min|Max|Median|Mean|StdDev)',axis=1)
	df_norm = pd.DataFrame(scaler.fit_transform(imgMets), columns=imgMets.columns)

	df_a = df[df.columns.difference(imgMets.columns)]
	bcDf = pd.concat([df_a.reset_index(drop=True), df_norm], axis=1).fillna(0)	
	
	
	#smTble = bcDf.groupby('Slide').apply(lambda x: x.sample(frac=0.25)) 
	smTble = bcDf.groupby('Slide', group_keys=False).apply(lambda x: x.sample(frac=0.25)) 
	df_batching = smTble.filter(regex='(Mean|Median|Slide)',axis=1)
	df_melted = pd.melt(df_batching, id_vars=["Slide"])
	
	fig, ax1 = plt.subplots(figsize=(24,8))
	origVals = sns.boxplot(x='Slide', y='value', color="#50C878", data=df_melted, ax=ax1, showfliers = False)
	plt.setp(ax1.get_xticklabels(), rotation=40, ha="right")
	ax1.set_title('Combined Marker Distribution (quantile values)')
	pdfOUT.savefig( origVals.get_figure() )
	
	
	colNames = list(filter(lambda x:'Mean' in x, df.columns.tolist()))
	NucOnly = list(filter(lambda x:nucMark in x, colNames))[0]
	for i in range(0, len(colNames), 8):
		# Create a new figure for each page
		fig, axs = plt.subplots(4, 4, figsize=(8.5, 11))
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
				ax2.set_title("Min-Max Transformation: {}".format(hd))
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
			
	pdfOUT = matplotlib.backends.backend_pdf.PdfPages("minmax_report_{}.pdf".format(myFileIdx))
	trnsfTBL = collect_and_transform(myData, pdfOUT, quantType, nucMark)
	pdfOUT.close()
	
	trnsfTBL.to_csv("minmax_transformed_{}.tsv".format(myFileIdx), sep="\t") 	
	
	





