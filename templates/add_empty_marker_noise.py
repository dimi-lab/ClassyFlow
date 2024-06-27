#!/usr/bin/env python3

import os, sys, math
import pandas as pd
from sklearn.datasets import make_blobs

staticColHead = 'Unnamed: 0'
objtype = '${params.qupath_object_type}'
#objtype = 'CellObject'
bitDepth = '${params.bit_depth}'



def getUniqueSets():
	uniqueSuffixes = []
	sts = ["Min","Max","Median","Mean","Std.Dev.","Variance"]

	if objtype == 'CellObject':
		comp = ["Nucleus","Cytoplasm","Membrane","Cell"]
		for c in comp:
			for s in sts:
				uniqueSuffixes.append(": "+c+": "+s)
	else:
		uniqueSuffixes = [": "+e for e in sts]

	return uniqueSuffixes

def findMissingFeatures(df, nom, designFile):
	panelDesign = pd.read_csv(designFile)
	#print( list(panelDesign.columns.values) )
	pdf2 = panelDesign.loc[panelDesign[nom] == 0]
	print(pdf2)
	if pdf2.shape[0] == 0:
		print("Skip this batch, no missing fields.")
		# Save the merged DataFrame as a pickle file
		df.to_pickle('merged_dataframe_{}_mod.pkl'.format(nom))
	
	else:
		missingMarks = pdf2[staticColHead].tolist()
		prt1DataT = df.copy(deep=True)
		for st in getUniqueSets():
			commonSetFeatures = df.filter(regex=st)
			print(st+'  '+str(commonSetFeatures.shape))
			descTbl = commonSetFeatures.describe([0.01,0.02,0.05,0.9])
			descTbl['avg'] = descTbl.mean(axis=1)
			mn = descTbl.loc['min','avg']
			mx = descTbl.loc['5%','avg']
			center_box = (mn, mx) # defines the box that cluster centres are allowed to be in
			standard_dev = math.ceil((mx-mn)/6) # defines the standard deviation of clusters

			theseMissingFields = [f+st for f in missingMarks]
			if len(theseMissingFields) == 0:
				sys.exit('Missing Fields Empty! ( {} )'.format(st))
			vals, lbs = make_blobs(n_samples=len(df), n_features=len(theseMissingFields), center_box=center_box, cluster_std=standard_dev)
			dfTmp = pd.DataFrame(vals, columns=theseMissingFields)
			dfTmp[dfTmp < 0] = 0
			prt1DataT = pd.concat([prt1DataT,dfTmp],axis=1)

		# Save the merged DataFrame as a pickle file
		prt1DataT.to_pickle('merged_dataframe_{}_mod.pkl'.format(nom))

if __name__ == "__main__":
	myDataFile = pd.read_pickle("${pickleTable}")
	myFileIdx = "${batchID}"
	panelCsvFile = "${designTable}"	

	findMissingFeatures(myDataFile, myFileIdx, panelCsvFile)
	


